#!/usr/bin/env python3
"""
Fill missing Design features using Dify LLM and CTG text blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from trial_agent.llm import DifyClient  # noqa: E402


WHITESPACE_RE = re.compile(r"\s+")
BIOMARKER_NAMES_FIELD = "Biomarker_names"
BIOMARKER_SLOT_FIELDS = ["Biomarker1", "Biomarker2", "Biomarker3"]
BIOMARKER_NAMES_ANNOTATION = (
    "List all biomarker names/criteria used for participant identification; separate with '; '."
)
ACTUAL_ENROLL_FIELD = "No_Subj_Actual"
ACTUAL_ENROLL_ANNOTATION = "Number of participants actually enrolled."
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")
SKIP_LLM_FIELDS = {
    "Molecule_Size",
    "SMILES",
    "ECFP6",
    "SELFIES",
    "F_bioav",
    "K_a",
    "C_max",
    "T_max",
    "AUC",
    "V_d",
    "V_ss",
    "Cl",
    "T_half",
    "K_el",
    "Cl_R",
    "Cl_H",
    *BIOMARKER_SLOT_FIELDS,
}


def load_env_file(env_path: Path) -> Dict[str, str]:
    if not env_path.exists():
        return {}
    values: Dict[str, str] = {}
    with env_path.open() as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("export "):
                raw = raw[len("export ") :].strip()
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def join_values(values: Iterable[object], delimiter: str = "; ") -> str:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return delimiter.join(ordered)


def stringify_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return join_values(value)
    return str(value).strip()


def stringify_evidence(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return join_values(value, delimiter=" | ")
    return str(value).strip()


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def detect_nct_id_column(fieldnames: Iterable[str], preferred: str = "") -> str:
    if preferred:
        preferred_key = normalize_header(preferred)
        for name in fieldnames:
            if normalize_header(name) == preferred_key:
                return name
        raise ValueError(f"NCT ID column '{preferred}' not found in CSV header.")

    normalized = {normalize_header(name): name for name in fieldnames if name}
    for candidate in NCT_ID_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("NCT ID column not found in CSV header; pass --nct-id-col.")


def load_nct_ids_from_csv(csv_path: Path, column: str = "", limit: int = 0) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing NCT CSV: {csv_path}")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"NCT CSV has no header: {csv_path}")
        col_name = detect_nct_id_column(reader.fieldnames, column)
        seen = set()
        ordered: List[str] = []
        for row in reader:
            raw = row.get(col_name, "")
            for nct_id in parse_nct_ids(raw or ""):
                if nct_id in seen:
                    continue
                seen.add(nct_id)
                ordered.append(nct_id)
                if limit and len(ordered) >= limit:
                    return ordered
    return ordered


def merge_nct_ids(primary: List[str], secondary: List[str], limit: int = 0) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for source in (primary, secondary):
        for nct_id in source:
            if nct_id in seen:
                continue
            seen.add(nct_id)
            ordered.append(nct_id)
            if limit and len(ordered) >= limit:
                return ordered
    return ordered


def parse_nct_ids(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw.strip())
    seen = set()
    ordered: List[str] = []
    for part in parts:
        if not part:
            continue
        value = part.upper()
        if not value.startswith("NCT"):
            value = f"NCT{value}"
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def cell_value(cell: ET.Element, shared_strings: List[str], ns: Dict[str, str]) -> Optional[str]:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", ns)
    if cell_type == "s" and value is not None:
        idx = int(value.text or "0")
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return None
    if cell_type == "inlineStr":
        inline = cell.find("a:is", ns)
        if inline is None:
            return None
        return "".join(t.text or "" for t in inline.findall(".//a:t", ns))
    if value is not None:
        return value.text
    return None


def load_design_schema(xlsx_path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(xlsx_path) as zf:
        wb_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

        rels_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels_xml}
        sheets = wb_xml.findall("a:sheets/a:sheet", ns)

        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        target = None
        for sheet in sheets:
            name = (sheet.attrib.get("name") or "").strip().lower()
            if name == "design":
                rid = sheet.attrib.get(f"{{{rel_ns}}}id")
                target = rels_map.get(rid)
                break
        if not target:
            raise ValueError("Design sheet not found in CSR-Vars.xlsx")

        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))
        rows = root.findall("a:sheetData/a:row", ns)

        parsed_rows: List[Dict[str, Optional[str]]] = []
        for row in rows:
            cells: Dict[str, Optional[str]] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                cells[col] = cell_value(cell, shared_strings, ns)
            if cells:
                parsed_rows.append(cells)

        header_idx = None
        header_map: Dict[str, str] = {}
        for i, cells in enumerate(parsed_rows):
            values = {v for v in cells.values() if v}
            if "Category" in values and "StudyID" in values:
                header_idx = i
                header_map = {str(v): k for k, v in cells.items() if v}
                break
        if header_idx is None:
            raise ValueError("Header row not found in Design sheet")

        col_category = header_map.get("Category")
        col_studyid = header_map.get("StudyID")
        col_annotation = header_map.get("Annotations")
        if not col_studyid:
            raise ValueError("StudyID column not found in Design sheet")

        fields: List[str] = []
        schema: Dict[str, Dict[str, str]] = {}
        for cells in parsed_rows[header_idx + 1 :]:
            study_id = (cells.get(col_studyid) or "").strip() if col_studyid else ""
            if not study_id:
                continue
            if study_id not in fields:
                fields.append(study_id)
            schema[study_id] = {
                "category": (cells.get(col_category) or "").strip() if col_category else "",
                "annotation": (cells.get(col_annotation) or "").strip() if col_annotation else "",
            }
        if not fields:
            raise ValueError("No StudyID fields found in Design sheet")
        return fields, schema


def ensure_biomarker_names_schema(schema: Dict[str, Dict[str, str]]) -> None:
    if BIOMARKER_NAMES_FIELD in schema:
        return
    schema[BIOMARKER_NAMES_FIELD] = {
        "category": "Design",
        "annotation": BIOMARKER_NAMES_ANNOTATION,
    }


def ensure_actual_enrollment_schema(schema: Dict[str, Dict[str, str]]) -> None:
    if ACTUAL_ENROLL_FIELD in schema:
        return
    schema[ACTUAL_ENROLL_FIELD] = {
        "category": "Design",
        "annotation": ACTUAL_ENROLL_ANNOTATION,
    }


def ensure_biomarker_names_fieldnames(fieldnames: List[str]) -> List[str]:
    if BIOMARKER_NAMES_FIELD in fieldnames:
        return fieldnames
    updated = list(fieldnames)
    insert_at = None
    for field in reversed(BIOMARKER_SLOT_FIELDS):
        if field in updated:
            insert_at = updated.index(field) + 1
            break
    if insert_at is None and "Biomarker" in updated:
        insert_at = updated.index("Biomarker") + 1
    if insert_at is None:
        insert_at = len(updated)
    updated.insert(insert_at, BIOMARKER_NAMES_FIELD)
    return updated


def ensure_actual_enrollment_fieldnames(fieldnames: List[str]) -> List[str]:
    if ACTUAL_ENROLL_FIELD in fieldnames:
        return fieldnames
    updated = list(fieldnames)
    insert_at = updated.index("No_Subj_Planned") + 1 if "No_Subj_Planned" in updated else len(updated)
    updated.insert(insert_at, ACTUAL_ENROLL_FIELD)
    return updated


def seed_biomarker_names(row: Dict[str, str]) -> None:
    if row.get(BIOMARKER_NAMES_FIELD):
        return
    values = [row.get(field) for field in BIOMARKER_SLOT_FIELDS if row.get(field)]
    if values:
        row[BIOMARKER_NAMES_FIELD] = join_values(values)


def load_text_blocks(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            nct_id = str(record.get("nct_id", "")).strip()
            if not nct_id:
                continue
            data[nct_id] = record
    return data


def is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def format_text_blocks(record: Dict[str, object], max_chars: int) -> str:
    sections: List[str] = []
    for label, key in [
        ("Brief Title", "brief_title"),
        ("Official Title", "official_title"),
        ("Brief Summary", "brief_summary"),
        ("Detailed Description", "detailed_description"),
        ("Eligibility Criteria", "eligibility_criteria"),
    ]:
        value = normalize_whitespace(str(record.get(key, "") or ""))
        if value:
            sections.append(f"{label}: {value}")

    for label, key in [
        ("Participant Flow", "participant_flow"),
        ("Baseline Results", "baseline_results"),
        ("Results Outcomes", "results_outcomes"),
        ("Arm Groups", "arm_groups"),
        ("Interventions", "interventions"),
        ("Primary Outcomes", "primary_outcomes"),
        ("Secondary Outcomes", "secondary_outcomes"),
    ]:
        value = record.get(key)
        if value:
            sections.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")

    for label, key in [
        ("Keywords", "keywords"),
        ("Conditions", "conditions"),
        ("Location Countries", "location_countries"),
    ]:
        value = record.get(key)
        if value:
            sections.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")

    text = "\n".join(sections)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text


def build_prompt(
    nct_id: str,
    missing_fields: List[str],
    schema: Dict[str, Dict[str, str]],
    text_blob: str,
) -> str:
    field_lines = []
    for field in missing_fields:
        info = schema.get(field, {})
        annotation = info.get("annotation", "")
        category = info.get("category", "")
        if annotation and category:
            field_lines.append(f"- {field}: {annotation} (Category: {category})")
        elif annotation:
            field_lines.append(f"- {field}: {annotation}")
        else:
            field_lines.append(f"- {field}")

    instructions = (
        "You extract clinical trial design variables from ClinicalTrials.gov text.\n"
        "Rules:\n"
        "- Use only the provided text.\n"
        "- Use each field's annotation as the authoritative definition.\n"
        "- If the text is ambiguous or conflicts with the annotation, return an empty string.\n"
        "- If a field is not explicitly stated, return an empty string.\n"
        "- Keep answers short and literal (Yes/No, numbers, or short phrases).\n"
        "- Output JSON only, with keys exactly as listed.\n"
        "- For each key, return an object: {\"value\": \"...\", \"evidence\": \"...\"}.\n"
        "- Evidence must be a short quote/snippet from the provided text; no paraphrasing.\n"
        "- If not explicitly stated, return {\"value\":\"\", \"evidence\":\"\"}.\n"
    )
    prompt = (
        f"{instructions}\nNCT ID: {nct_id}\n\nFields:\n"
        + "\n".join(field_lines)
        + "\n\nText:\n"
        + text_blob
        + "\n"
    )
    return prompt


def parse_json_response(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing Design features using Dify.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_design_extract/design_features.csv",
        help="Input CSV from the structured extractor (auto-suffixed when --nct-id is set and this default is used).",
    )
    parser.add_argument(
        "--text-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_design_extract/ctg_text_blocks.jsonl",
        help="Text blocks JSONL from the structured extractor (auto-suffixed when --nct-id is set and this default is used).",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=PROJECT_ROOT / "data/raw/CSR-Vars.xlsx",
        help="CSR-Vars.xlsx path for field annotations.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_design_extract/design_features_llm.csv",
        help="Output CSV with LLM-filled values (auto-suffixed when --nct-id is set and this default is used).",
    )
    parser.add_argument(
        "--responses-out",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_design_extract/llm_responses.jsonl",
        help="JSONL with raw LLM responses for auditing (auto-suffixed when --nct-id is set and this default is used).",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_design_extract/llm_evidence.jsonl",
        help="JSONL with per-field evidence (auto-suffixed when --nct-id is set and this default is used).",
    )
    parser.add_argument(
        "--nct-id",
        type=str,
        default="",
        help="Only process a single NCT ID (or comma/space-separated list).",
    )
    parser.add_argument(
        "--nct-csv",
        type=Path,
        default=None,
        help="CSV containing NCT IDs to process (uses --nct-id-col or auto-detect).",
    )
    parser.add_argument(
        "--nct-id-col",
        type=str,
        default="",
        help="Column name in --nct-csv that holds NCT IDs (default: auto-detect).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters from text blocks to include in the prompt.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on number of trials processed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of NCT IDs processed (applies after combining inputs).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between LLM calls.",
    )
    parser.add_argument(
        "--dify-api-key",
        type=str,
        default=None,
        help="Override DIFY_API_KEY.",
    )
    parser.add_argument(
        "--dify-base-url",
        type=str,
        default=None,
        help="Override DIFY_BASE_URL.",
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        default="",
        help="Optional Dify conversation_id.",
    )
    return parser.parse_args()


def suffix_for_ncts(nct_ids: List[str]) -> str:
    if not nct_ids:
        return ""
    if len(nct_ids) == 1:
        return nct_ids[0]
    if len(nct_ids) <= 3:
        return "_".join(nct_ids)
    return f"{nct_ids[0]}_plus{len(nct_ids) - 1}"


def main() -> None:
    args = parse_args()
    default_input = PROJECT_ROOT / "data/ctg_design_extract/design_features.csv"
    default_text = PROJECT_ROOT / "data/ctg_design_extract/ctg_text_blocks.jsonl"
    default_output = PROJECT_ROOT / "data/ctg_design_extract/design_features_llm.csv"
    default_responses = PROJECT_ROOT / "data/ctg_design_extract/llm_responses.jsonl"
    default_evidence = PROJECT_ROOT / "data/ctg_design_extract/llm_evidence.jsonl"

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)
    if nct_ids:
        suffix = suffix_for_ncts(nct_ids)
        if args.input_csv == default_input:
            args.input_csv = default_input.with_name(f"design_features_{suffix}.csv")
        if args.text_jsonl == default_text:
            args.text_jsonl = default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl")
        if args.output_csv == default_output:
            args.output_csv = default_output.with_name(f"design_features_{suffix}_llm.csv")
        if args.responses_out == default_responses:
            args.responses_out = default_responses.with_name(f"llm_responses_{suffix}.jsonl")
        if args.evidence_out == default_evidence:
            args.evidence_out = default_evidence.with_name(f"llm_evidence_{suffix}.jsonl")
    elif args.limit and args.max_trials == 0:
        args.max_trials = args.limit

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.text_jsonl.exists():
        raise FileNotFoundError(f"Missing text JSONL: {args.text_jsonl}")
    if not args.xlsx.exists():
        raise FileNotFoundError(f"Missing xlsx: {args.xlsx}")

    _, schema = load_design_schema(args.xlsx)
    ensure_biomarker_names_schema(schema)
    ensure_actual_enrollment_schema(schema)
    text_blocks = load_text_blocks(args.text_jsonl)
    allowed_nct_ids = set(nct_ids)

    if not args.dify_api_key:
        env_key = os.getenv("DIFY_API_KEY")
        if env_key:
            args.dify_api_key = env_key
        else:
            env_values = load_env_file(PROJECT_ROOT / ".env")
            if env_values.get("DIFY_API_KEY"):
                args.dify_api_key = env_values["DIFY_API_KEY"]
            if not args.dify_base_url and env_values.get("DIFY_BASE_URL"):
                args.dify_base_url = env_values["DIFY_BASE_URL"]

    if not args.dify_api_key:
        raise ValueError("DIFY_API_KEY is not set; add it to .env or pass --dify-api-key.")

    client = DifyClient(api_key=args.dify_api_key, base_url=args.dify_base_url)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.responses_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with args.input_csv.open() as f_in, \
        args.output_csv.open("w", newline="") as f_out, \
        args.responses_out.open("w") as f_resp, \
        args.evidence_out.open("w") as f_evidence:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        fieldnames = ensure_biomarker_names_fieldnames(list(reader.fieldnames))
        fieldnames = ensure_actual_enrollment_fieldnames(fieldnames)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            nct_id = (row.get("NCT_No") or "").strip()
            if allowed_nct_ids and nct_id not in allowed_nct_ids:
                writer.writerow(row)
                continue

            record = text_blocks.get(nct_id)
            if not record:
                writer.writerow(row)
                continue

            row.setdefault(BIOMARKER_NAMES_FIELD, "")
            seed_biomarker_names(row)

            missing_fields = [
                field
                for field in fieldnames
                if is_missing(row.get(field)) and field not in SKIP_LLM_FIELDS
            ]
            if not missing_fields:
                writer.writerow(row)
                continue

            text_blob = format_text_blocks(record, args.max_chars)
            prompt = build_prompt(nct_id, missing_fields, schema, text_blob)
            response = client.chat(
                prompt,
                inputs=None,
                conversation_id=args.conversation_id,
                user="ctg-design-extract",
            )
            parsed = parse_json_response(response or "")

            field_outputs: Dict[str, Dict[str, str]] = {
                field: {"value": "", "evidence": ""} for field in missing_fields
            }
            if parsed:
                for field in missing_fields:
                    payload = parsed.get(field)
                    if isinstance(payload, dict):
                        value = stringify_value(payload.get("value"))
                        evidence = stringify_evidence(payload.get("evidence"))
                    else:
                        value = stringify_value(payload)
                        evidence = ""
                    if value:
                        row[field] = value
                    field_outputs[field] = {
                        "value": value,
                        "evidence": evidence,
                    }

            f_resp.write(
                json.dumps(
                    {
                        "nct_id": nct_id,
                        "missing_fields": missing_fields,
                        "response": response,
                        "parsed": parsed,
                        "normalized": field_outputs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_evidence.write(
                json.dumps(
                    {"nct_id": nct_id, "fields": field_outputs},
                    ensure_ascii=False,
                )
                + "\n"
            )

            writer.writerow(row)
            processed += 1
            if args.max_trials and processed >= args.max_trials:
                break
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
