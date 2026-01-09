#!/usr/bin/env python3
"""
Fill CTG table fields using Dify LLM.
Per-table prompts can be supplied by wrapper scripts.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, TextIO, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from trial_agent.llm import DifyClient  # noqa: E402


WHITESPACE_RE = re.compile(r"\s+")
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")

DESIGN_TABLES = ("Design", "Stat_Reg", "TargetPop", "Drug", "Others")
RESULT_TABLES = ("Endpoints", "Groups")
ALL_TABLES = DESIGN_TABLES + RESULT_TABLES

BIOMARKER_NAMES_FIELD = "Biomarker_names"
BIOMARKER_SLOT_FIELDS = ["Biomarker1", "Biomarker2", "Biomarker3"]

STAT_REG_KNOWN_FIELDS_PRIORITY = [
    "Randomization",
    "Blinding",
    "Level_Blinding",
    "Rand_Ratio",
    "No_Arm",
    "Stratification",
    "No_Stratification",
    "Placebo_control",
    "Active_Control",
    "Single_Arm",
    "Hist_control",
    "No_Prim_EP",
    "Primary_EP",
    "Key_Second_EP",
    "Sample_Size",
]

DEFAULT_INSTRUCTIONS = "You extract clinical trial variables from ClinicalTrials.gov text."

ExtraRulesFn = Callable[[List[str]], List[str]]


@dataclass(frozen=True)
class PromptConfig:
    instructions: str
    notes: Optional[Dict[str, str]] = None
    extra_rules_fn: Optional[ExtraRulesFn] = None
    llm_fields: Optional[List[str]] = None
    text_modules: Optional[List[str]] = None

TABLE_SKIP_FIELDS = {
    "Design": {"StudyID", "NCT_No", "Prot_No", "EUCT_No", "Other_No", "Drug_Name"},
    "Stat_Reg": {"StudyID", "Drug_Name", "Control_Drug"},
    "TargetPop": {"StudyID", "Drug_Name"},
    "Drug": {"StudyID", "Drug_Name"},
    "Others": {"StudyID", "Drug_Name"},
    "Endpoints": {"StudyID", "Arm_ID", "Endpoint_Name", "Endpoint_Type", "Drug_Name", "EP_P_value"},
    "Groups": {"StudyID", "Arm_ID", "Drug_Name"},
}


def default_paths(table: str) -> Dict[str, Path]:
    if table in DESIGN_TABLES:
        base = PROJECT_ROOT / "data/ctg_extract/design"
        input_csv = base / f"design_{table}.csv"
        output_csv = base / f"design_{table}_llm.csv"
        responses = base / f"llm_responses_{table}.jsonl"
        evidence = base / f"llm_evidence_{table}.jsonl"
    else:
        base = PROJECT_ROOT / "data/ctg_extract/results"
        lower = table.lower()
        input_csv = base / f"results_{lower}.csv"
        output_csv = base / f"results_{lower}_llm.csv"
        responses = base / f"results_llm_responses_{table}.jsonl"
        evidence = base / f"results_llm_evidence_{table}.jsonl"
    return {
        "input": input_csv,
        "output": output_csv,
        "responses": responses,
        "evidence": evidence,
    }


def per_nct_paths(table: str, nct_id: str) -> Dict[str, Path]:
    base_dir = PROJECT_ROOT / "data/ctg_extract/by_nctid" / nct_id
    out_dir = base_dir / "llm"
    table_token = table.lower()
    return {
        "input": base_dir / f"{table}.csv",
        "output": out_dir / f"{table_token}_llm.csv",
        "responses": out_dir / f"{table_token}_llm_responses.jsonl",
        "evidence": out_dir / f"{table_token}_llm_evidence.jsonl",
    }


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


def normalize_table_name(name: str) -> str:
    if not name:
        raise ValueError("Missing --table (Design, Stat_Reg, TargetPop, Drug, Others, Endpoints, Groups).")
    for table in ALL_TABLES:
        if table.lower() == name.strip().lower():
            return table
    raise ValueError(f"Unknown table: {name}")


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


def load_sheet_schema(xlsx_path: Path, sheet_name: str) -> Dict[str, Dict[str, str]]:
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
            name = (sheet.attrib.get("name") or "").strip()
            if name == sheet_name:
                rid = sheet.attrib.get(f"{{{rel_ns}}}id")
                target = rels_map.get(rid)
                break
        if not target:
            raise ValueError(f"Sheet '{sheet_name}' not found in CSR-Vars.")

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
            if "Category" in values and ("StudyID" in values or "Variable" in values):
                header_idx = i
                header_map = {str(v): k for k, v in cells.items() if v}
                break
        if header_idx is None:
            raise ValueError(f"Header row not found in sheet '{sheet_name}'.")

        key_col = header_map.get("StudyID") or header_map.get("Variable")
        ann_col = header_map.get("Annotations")
        cat_col = header_map.get("Category")
        if not key_col:
            raise ValueError(f"StudyID/Variable column not found in sheet '{sheet_name}'.")

        schema: Dict[str, Dict[str, str]] = {}
        for cells in parsed_rows[header_idx + 1 :]:
            study_id = (cells.get(key_col) or "").strip()
            if not study_id:
                continue
            schema[study_id] = {
                "category": (cells.get(cat_col) or "").strip() if cat_col else "",
                "annotation": (cells.get(ann_col) or "").strip() if ann_col else "",
            }
        return schema


def is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def normalize_list_values(value: object) -> str:
    if isinstance(value, list):
        return join_values(value)
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_match_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return normalize_whitespace(cleaned)


def match_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    token_score = 0
    if tokens_a and tokens_b:
        token_score = int(round(len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b)) * 100))
    seq_score = int(round(difflib.SequenceMatcher(None, a, b).ratio() * 100))
    return max(token_score, seq_score)


def select_outcomes(
    outcomes: object,
    endpoint_name: str,
    max_candidates: int = 2,
    min_score: int = 80,
) -> List[Tuple[Dict[str, object], int]]:
    if not outcomes or not endpoint_name:
        return []
    if not isinstance(outcomes, list):
        return []
    target_norm = normalize_match_text(endpoint_name)
    if not target_norm:
        return []

    exact_matches: List[Tuple[Dict[str, object], int]] = []
    scored: List[Tuple[Dict[str, object], int]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        title = normalize_whitespace(str(outcome.get("title") or ""))
        title_norm = normalize_match_text(title)
        if not title_norm:
            continue
        if title_norm == target_norm:
            exact_matches.append((outcome, 100))
        else:
            scored.append((outcome, match_score(target_norm, title_norm)))

    if exact_matches:
        return exact_matches[:max_candidates]
    if not scored:
        return []
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored[0][1] < min_score:
        return []
    return scored[:max_candidates]


def format_outcome(outcome: Dict[str, object], score: Optional[int] = None) -> str:
    lines: List[str] = []
    title = normalize_whitespace(str(outcome.get("title") or ""))
    header = f"- Outcome Title: {title}" if title else "- Outcome Title: (missing)"
    if score is not None:
        header += f" [match score: {score}]"
    lines.append(header)

    for label, key in [
        ("Type", "type"),
        ("Time frame", "time_frame"),
        ("Description", "description"),
        ("Population", "population"),
    ]:
        value = normalize_whitespace(str(outcome.get(key) or ""))
        if value:
            lines.append(f"  {label}: {value}")

    groups = outcome.get("groups")
    if isinstance(groups, list) and groups:
        lines.append("  Groups:")
        for group in groups:
            if not isinstance(group, dict):
                value = normalize_whitespace(str(group))
                if value:
                    lines.append(f"    - {value}")
                continue
            group_id = normalize_whitespace(str(group.get("group_id") or ""))
            group_title = normalize_whitespace(str(group.get("title") or ""))
            group_desc = normalize_whitespace(str(group.get("description") or ""))
            parts = []
            if group_id:
                parts.append(f"id={group_id}")
            if group_title:
                parts.append(group_title)
            if parts:
                lines.append(f"    - {'; '.join(parts)}")
            if group_desc:
                lines.append(f"      Desc: {group_desc}")

    analyses = outcome.get("analysis_list")
    if isinstance(analyses, list) and analyses:
        lines.append("  Analyses:")
        for analysis in analyses:
            if not isinstance(analysis, dict):
                value = normalize_whitespace(str(analysis))
                if value:
                    lines.append(f"    - {value}")
                continue
            details: List[str] = []
            for key, label in [
                ("groups_desc", "Groups"),
                ("method", "Method"),
                ("method_desc", "Method desc"),
                ("estimate_desc", "Estimate desc"),
                ("p_value_desc", "P value desc"),
                ("param_type", "Param type"),
                ("non_inferiority_type", "Non-inferiority"),
            ]:
                value = normalize_whitespace(str(analysis.get(key) or ""))
                if value:
                    details.append(f"{label}: {value}")
            if details:
                lines.append(f"    - {details[0]}")
                for extra in details[1:]:
                    lines.append(f"      {extra}")

    return "\n".join(lines)


def format_arm_groups(arm_groups: object) -> str:
    if not arm_groups:
        return ""
    if not isinstance(arm_groups, list):
        return normalize_whitespace(str(arm_groups))
    lines: List[str] = []
    for arm in arm_groups:
        if not isinstance(arm, dict):
            text = normalize_whitespace(str(arm))
            if text:
                lines.append(f"- Arm: {text}")
            continue
        label = normalize_whitespace(str(arm.get("label", "") or ""))
        arm_type = normalize_whitespace(str(arm.get("type", "") or ""))
        desc = normalize_whitespace(str(arm.get("description", "") or ""))
        header = f"Arm: {label}" if label else "Arm: (unnamed)"
        if arm_type:
            header = f"{header} ({arm_type})"
        lines.append(f"- {header}")
        if desc:
            lines.append(f"  Desc: {desc}")
    return "\n".join(lines)


def format_interventions(interventions: object) -> str:
    if not interventions:
        return ""
    if not isinstance(interventions, list):
        return normalize_whitespace(str(interventions))
    lines: List[str] = []
    for intervention in interventions:
        if not isinstance(intervention, dict):
            text = normalize_whitespace(str(intervention))
            if text:
                lines.append(f"- Intervention: {text}")
            continue
        name = normalize_whitespace(str(intervention.get("name", "") or ""))
        iv_type = normalize_whitespace(str(intervention.get("type", "") or ""))
        desc = normalize_whitespace(str(intervention.get("description", "") or ""))
        labels = normalize_list_values(intervention.get("labels", []))
        other_names = normalize_list_values(intervention.get("other_names", []))
        header = f"Intervention: {name}" if name else "Intervention: (unnamed)"
        if iv_type:
            header = f"{header} ({iv_type})"
        lines.append(f"- {header}")
        if labels:
            lines.append(f"  Arms: {labels}")
        if desc:
            lines.append(f"  Desc: {desc}")
        if other_names:
            lines.append(f"  Other names: {other_names}")
    return "\n".join(lines)


def format_outcome_definitions(outcomes: object) -> str:
    if not outcomes:
        return ""
    if not isinstance(outcomes, list):
        return normalize_whitespace(str(outcomes))
    lines: List[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            text = normalize_whitespace(str(outcome))
            if text:
                lines.append(f"- Outcome: {text}")
            continue
        title = normalize_whitespace(str(outcome.get("title") or ""))
        header = f"- Outcome: {title}" if title else "- Outcome: (missing)"
        lines.append(header)
        time_frame = normalize_whitespace(str(outcome.get("time_frame") or ""))
        if time_frame:
            lines.append(f"  Time frame: {time_frame}")
        description = normalize_whitespace(str(outcome.get("description") or ""))
        if description:
            lines.append(f"  Description: {description}")
    return "\n".join(lines)


STUDY_INFO_FIELDS = [
    ("Brief Title", "brief_title"),
    ("Official Title", "official_title"),
    ("Brief Summary", "brief_summary"),
    ("Detailed Description", "detailed_description"),
]

TEXT_BLOCK_MODULES = {
    "study_info": "Brief/official title, brief summary, detailed description",
    "eligibility": "Eligibility criteria",
    "design_info": "Structured design metadata (phase, allocation, masking, etc.)",
    "arm_groups": "Arm groups",
    "interventions": "Interventions",
    "primary_outcomes": "Primary outcomes",
    "secondary_outcomes": "Secondary outcomes",
    "participant_flow": "Participant flow (milestones/drop-withdraw reasons)",
    "baseline_results": "Baseline results (population/groups)",
    "baseline_measures": "Baseline measures (categories/values)",
    "results_outcomes": "Results outcomes (analyses/measures)",
    "reported_events": "Serious/other reported events",
    "keywords": "Keywords",
    "conditions": "Conditions",
    "location_countries": "Location countries",
    "endpoint_target": "Endpoint target fields from the row",
    "endpoint_matches": "Matched outcomes from results",
}

DEFAULT_TEXT_MODULES_GENERIC = [
    "study_info",
    "eligibility",
    "participant_flow",
    "baseline_results",
    "baseline_measures",
    "results_outcomes",
    "reported_events",
    "arm_groups",
    "interventions",
    "primary_outcomes",
    "secondary_outcomes",
    "keywords",
    "conditions",
    "location_countries",
]

DEFAULT_TEXT_MODULES = {
    "Design": [
        "design_info",
        "study_info",
        "eligibility",
        "arm_groups",
        "interventions",
    ],
    "Stat_Reg": [
        "design_info",
        "study_info",
        "arm_groups",
        "interventions",
        "primary_outcomes",
        "secondary_outcomes",
    ],
    "Endpoints": [
        "endpoint_target",
        "endpoint_matches",
    ],
    "TargetPop": [
        "study_info",
        "eligibility",
    ],
}


def resolve_text_modules(table: str, text_modules: Optional[List[str]]) -> List[str]:
    modules = text_modules or DEFAULT_TEXT_MODULES.get(table, DEFAULT_TEXT_MODULES_GENERIC)
    return [module for module in modules if module in TEXT_BLOCK_MODULES]


def format_text_blocks(
    record: Dict[str, object],
    max_chars: int,
    table: str,
    row: Optional[Dict[str, str]] = None,
    text_modules: Optional[List[str]] = None,
) -> str:
    sections: List[str] = []
    modules = resolve_text_modules(table, text_modules)
    for module in modules:
        if module == "design_info":
            design_info = record.get("design_info")
            if design_info:
                sections.append("Structured Design:\n" + json.dumps(design_info, ensure_ascii=False))
        elif module == "study_info":
            for label, key in STUDY_INFO_FIELDS:
                value = normalize_whitespace(str(record.get(key, "") or ""))
                if value:
                    sections.append(f"{label}: {value}")
        elif module == "eligibility":
            value = normalize_whitespace(str(record.get("eligibility_criteria", "") or ""))
            if value:
                sections.append(f"Eligibility Criteria: {value}")
        elif module == "arm_groups":
            arm_groups = format_arm_groups(record.get("arm_groups"))
            if arm_groups:
                sections.append("Arm Groups:\n" + arm_groups)
        elif module == "interventions":
            interventions = format_interventions(record.get("interventions"))
            if interventions:
                sections.append("Interventions:\n" + interventions)
        elif module == "primary_outcomes":
            primary_outcomes = format_outcome_definitions(record.get("primary_outcomes"))
            if primary_outcomes:
                sections.append("Primary Outcomes:\n" + primary_outcomes)
        elif module == "secondary_outcomes":
            secondary_outcomes = format_outcome_definitions(record.get("secondary_outcomes"))
            if secondary_outcomes:
                sections.append("Secondary Outcomes:\n" + secondary_outcomes)
        elif module == "participant_flow":
            value = record.get("participant_flow")
            if value:
                sections.append(f"Participant Flow: {json.dumps(value, ensure_ascii=False)}")
        elif module == "baseline_results":
            value = record.get("baseline_results")
            if value:
                sections.append(f"Baseline Results: {json.dumps(value, ensure_ascii=False)}")
        elif module == "baseline_measures":
            value = record.get("baseline_measures")
            if value:
                sections.append(f"Baseline Measures: {json.dumps(value, ensure_ascii=False)}")
        elif module == "results_outcomes":
            value = record.get("results_outcomes")
            if value:
                sections.append(f"Results Outcomes: {json.dumps(value, ensure_ascii=False)}")
        elif module == "reported_events":
            value = record.get("reported_events")
            if value:
                sections.append(f"Reported Events: {json.dumps(value, ensure_ascii=False)}")
        elif module == "keywords":
            value = record.get("keywords")
            if value:
                sections.append(f"Keywords: {json.dumps(value, ensure_ascii=False)}")
        elif module == "conditions":
            value = record.get("conditions")
            if value:
                sections.append(f"Conditions: {json.dumps(value, ensure_ascii=False)}")
        elif module == "location_countries":
            value = record.get("location_countries")
            if value:
                sections.append(f"Location Countries: {json.dumps(value, ensure_ascii=False)}")
        elif module == "endpoint_target":
            target_name = normalize_whitespace(str((row or {}).get("Endpoint_Name") or ""))
            target_type = normalize_whitespace(str((row or {}).get("Endpoint_Type") or ""))
            target_arm = normalize_whitespace(str((row or {}).get("Arm_ID") or ""))
            if target_name:
                sections.append(f"Target Endpoint_Name: {target_name}")
            if target_type:
                sections.append(f"Target Endpoint_Type: {target_type}")
            if target_arm:
                sections.append(f"Target Arm_ID: {target_arm}")
        elif module == "endpoint_matches":
            target_name = normalize_whitespace(str((row or {}).get("Endpoint_Name") or ""))
            selected = select_outcomes(record.get("results_outcomes"), target_name, max_candidates=1)
            if selected:
                sections.append("Matched Outcomes:")
                for outcome, score in selected:
                    sections.append(format_outcome(outcome, score))
            else:
                sections.append(
                    "No matching outcome found for Target Endpoint_Name; return empty for all fields."
                )

    text = "\n".join(sections)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text


def field_lines(
    missing_fields: List[str],
    schema: Dict[str, Dict[str, str]],
    notes: Dict[str, str],
) -> List[str]:
    lines = []
    for field in missing_fields:
        info = schema.get(field, {})
        annotation = info.get("annotation", "")
        category = info.get("category", "")
        note = notes.get(field, "")
        if annotation and category:
            line = f"- {field}: {annotation} (Category: {category})"
        elif annotation:
            line = f"- {field}: {annotation}"
        elif note:
            line = f"- {field}: {note}"
        else:
            line = f"- {field}"
        if note and annotation:
            line += f" Note: {note}"
        lines.append(line)
    return lines


def output_template(missing_fields: List[str]) -> str:
    template = {field: {"value": "", "evidence": ""} for field in missing_fields}
    return json.dumps(template, ensure_ascii=False)


def build_prompt(
    table: str,
    nct_id: str,
    row: Dict[str, str],
    missing_fields: List[str],
    schema: Dict[str, Dict[str, str]],
    prompt_config: Optional[PromptConfig],
    text_blob: str,
) -> str:
    if table == "Stat_Reg":
        context_pairs = prioritize_context_pairs(
            row, STAT_REG_KNOWN_FIELDS_PRIORITY, limit=20
        )
    else:
        context_pairs = [f"{k}={v}" for k, v in row.items() if v]
    context_text = "; ".join(context_pairs[:20])

    instructions = DEFAULT_INSTRUCTIONS
    notes: Dict[str, str] = {}
    extra_rules: List[str] = []
    if prompt_config:
        instructions = prompt_config.instructions or instructions
        if prompt_config.notes:
            notes = prompt_config.notes
        if prompt_config.extra_rules_fn:
            extra_rules = prompt_config.extra_rules_fn(missing_fields)

    instructions = (
        f"{instructions}\n"
        "You are an information extraction engine.\n"
        "Return ONLY a valid JSON object (no markdown, no commentary).\n"
        "\n"
        "Hard rules:\n"
        "- Use ONLY the content inside TEXT_START ... TEXT_END to infer values.\n"
        "- Evidence MUST be exact contiguous substrings copied from TEXT_START...TEXT_END ONLY.\n"
        "- Do NOT use CONTEXT as evidence.\n"
        "- If the text is ambiguous or conflicts with the field meaning, return empty.\n"
        "- If a field is not explicitly stated in TEXT, return empty.\n"
        "- You may normalize output ONLY when Extra rules explicitly allow it; evidence must quote the original phrase.\n"
        "- Evidence can be multiple substrings separated by ' | ' (each must appear in TEXT).\n"
        "- Keep evidence short (<=240 chars per segment). No ellipses.\n"
        "- Keys MUST match exactly and MUST include ALL keys listed in the template.\n"
        "- Output must match this schema for every key: {\"value\":\"...\",\"evidence\":\"...\"}.\n"
        "- Do NOT add extra keys.\n"
    )
    if extra_rules:
        instructions += "Extra rules:\n" + "\n".join(extra_rules) + "\n"

    prompt = (
        f"{instructions}\nNCT ID: {nct_id}\nTable: {table}\n\n"
        "CONTEXT_START (reference only; DO NOT cite as evidence)\n"
        f"{context_text}\n"
        "CONTEXT_END\n\n"
        "FIELDS_START (definitions)\n"
        + "\n".join(field_lines(missing_fields, schema, notes))
        + "\nFIELDS_END\n\n"
        "OUTPUT_TEMPLATE_START\n"
        + output_template(missing_fields)
        + "\nOUTPUT_TEMPLATE_END\n\n"
        "TEXT_START\n"
        + text_blob
        + "\nTEXT_END\n"
    )
    return prompt


def parse_json_response(text: str) -> Optional[Dict[str, object]]:
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


def split_evidence_segments(evidence: str) -> List[str]:
    segments = [segment.strip() for segment in evidence.split("|")]
    return [segment for segment in segments if segment]


def validate_evidence(
    value: str,
    evidence: str,
    text_blob: str,
    max_len: int = 240,
) -> Tuple[str, str]:
    if not value:
        return "", ""
    if not evidence:
        return "", ""
    segments = split_evidence_segments(evidence)
    if not segments:
        return "", ""
    for segment in segments:
        if len(segment) > max_len or segment not in text_blob:
            return "", ""
    return value, " | ".join(segments)


def prioritize_context_pairs(
    row: Dict[str, str], priority: List[str], limit: int = 20
) -> List[str]:
    pairs: List[str] = []
    seen = set()
    for key in priority:
        value = row.get(key)
        if not value:
            continue
        pairs.append(f"{key}={value}")
        seen.add(key)
        if len(pairs) >= limit:
            return pairs
    for key, value in row.items():
        if not value or key in seen:
            continue
        pairs.append(f"{key}={value}")
        if len(pairs) >= limit:
            break
    return pairs


def seed_biomarker_names(row: Dict[str, str]) -> None:
    if BIOMARKER_NAMES_FIELD not in row or row.get(BIOMARKER_NAMES_FIELD):
        return
    values = [row.get(field) for field in BIOMARKER_SLOT_FIELDS if row.get(field)]
    if values:
        row[BIOMARKER_NAMES_FIELD] = join_values(values)


def suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def load_text_blocks(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    raw = jsonl_path.read_text().strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        return index_text_blocks(parsed)

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


class JsonArrayWriter:
    def __init__(self, handle: TextIO, indent: int = 2) -> None:
        self.handle = handle
        self.indent = indent
        self.first = True
        self.handle.write("[\n")

    def write(self, payload: Dict[str, object]) -> None:
        if not self.first:
            self.handle.write(",\n")
        entry = json.dumps(payload, ensure_ascii=False, indent=self.indent)
        prefix = " " * self.indent
        indented = "\n".join(f"{prefix}{line}" if line else line for line in entry.splitlines())
        self.handle.write(indented)
        self.first = False

    def close(self) -> None:
        if not self.first:
            self.handle.write("\n")
        self.handle.write("]\n")
        self.handle.close()


def index_text_blocks(payload: object) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    if isinstance(payload, dict):
        if "nct_id" in payload:
            nct_id = str(payload.get("nct_id") or "").strip()
            if nct_id:
                data[nct_id] = payload
            return data
        for key, record in payload.items():
            if not isinstance(record, dict):
                continue
            nct_id = str(record.get("nct_id") or key).strip()
            if not nct_id:
                continue
            if "nct_id" not in record:
                record = dict(record)
                record["nct_id"] = nct_id
            data[nct_id] = record
        return data
    if isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            nct_id = str(record.get("nct_id") or "").strip()
            if not nct_id:
                continue
            data[nct_id] = record
    return data


def order_fields_for_table(table: str, fields: List[str]) -> List[str]:
    if table == "Design":
        priority = [
            "Route_Admin",
            "Treat_Duration",
            "Add_On_Treat",
            "Adherence_Treat",
            "Enroll_Duration_Plan",
            "FU_Duration_Plan",
            "Central_Lab",
            "Run_in",
            "GCP_Compliance",
            "Data_Cutoff_Date",
        ]
    elif table == "Stat_Reg":
        priority = [
            "Central_Random",
            "Rand_Ratio",
            "No_Stratification",
            "IRC",
            "Subgroup",
            "Adaptive_Design",
            "Interim",
            "Timing_IA",
            "Alpha",
            "Sided",
            "Power",
            "Alpha_Spend_Func",
            "Gatekeeping_Strategy",
            "Consistency_Sens_Ana_PE",
            "Consistency_Sens_Ana_SE",
            "Post_Hoc_Ana",
            "Intercurrent_Events",
            "Success_Criteria_Text",
            "Reg_Alignment",
            "Reg_Audit",
            "Consistency_MRCT",
            "Fast_Track",
            "Breakthrough",
            "Priority_Review",
            "Accelerated_App",
            "Orphan_Drug",
            "Pediatric",
            "Rare_Disease",
        ]
    elif table == "Endpoints":
        priority = [
            "Strategy",
            "Missing_Imput",
            "Covariate_Adjust",
            "MCP",
            "Subgroup_Ana",
            "EP_Value",
            "EP_Unit",
            "EP_Point",
            "EP_95CI",
            "ARR",
            "NNT",
            "Med_OS",
            "OS_YrX",
            "Med_PFS",
            "ORR",
            "pCR",
            "Med_DOR",
            "RMST",
            "PRO",
            "QoL",
        ]
    else:
        return list(fields)
    fields_set = set(fields)
    ordered = [field for field in priority if field in fields_set]
    seen = set(ordered)
    for field in fields:
        if field in seen:
            continue
        ordered.append(field)
        seen.add(field)
    return ordered


def split_fields(fields: List[str], max_fields: int) -> List[List[str]]:
    if max_fields <= 0 or len(fields) <= max_fields:
        return [fields]
    return [fields[i : i + max_fields] for i in range(0, len(fields), max_fields)]


def process_table(
    table: str,
    input_path: Path,
    output_path: Path,
    text_blocks: Dict[str, Dict[str, object]],
    client: DifyClient,
    allowed_nct_ids: set,
    max_chars: int,
    max_rows: int,
    conversation_id: str,
    responses_file,
    evidence_file,
    sleep_seconds: float,
    schema: Dict[str, Dict[str, str]],
    prompt_config: Optional[PromptConfig],
    llm_fields: Optional[set],
    skip_fields: set,
    max_fields_per_call: int,
    pretty_evidence: Optional[JsonArrayWriter] = None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    with input_path.open() as f_in, output_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_path}")
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        for row in reader:
            nct_id = (row.get("StudyID") or row.get("NCT_No") or "").strip()
            if allowed_nct_ids and nct_id not in allowed_nct_ids:
                writer.writerow(row)
                continue

            record = text_blocks.get(nct_id)
            if not record:
                writer.writerow(row)
                continue

            seed_biomarker_names(row)

            missing_fields = []
            for field in fieldnames:
                if field in skip_fields:
                    continue
                if llm_fields is not None and field not in llm_fields:
                    continue
                if is_missing(row.get(field)):
                    missing_fields.append(field)
            if not missing_fields:
                writer.writerow(row)
                continue

            ordered_fields = order_fields_for_table(table, missing_fields)
            batches = split_fields(ordered_fields, max_fields_per_call)
            text_modules = prompt_config.text_modules if prompt_config else None
            text_blob = format_text_blocks(record, max_chars, table, row, text_modules)

            for batch_index, batch_fields in enumerate(batches):
                prompt = build_prompt(
                    table=table,
                    nct_id=nct_id,
                    row=row,
                    missing_fields=batch_fields,
                    schema=schema,
                    prompt_config=prompt_config,
                    text_blob=text_blob,
                )
                response = client.chat(
                    prompt,
                    inputs=None,
                    conversation_id=conversation_id,
                    user=f"ctg-{table.lower()}-extract",
                )
                parsed = parse_json_response(response or "")

                field_outputs: Dict[str, Dict[str, str]] = {
                    field: {"value": "", "evidence": ""} for field in batch_fields
                }
                if parsed:
                    for field in batch_fields:
                        payload = parsed.get(field)
                        if isinstance(payload, dict):
                            value = stringify_value(payload.get("value"))
                            evidence = stringify_evidence(payload.get("evidence"))
                        else:
                            value = stringify_value(payload)
                            evidence = ""
                        value, evidence = validate_evidence(value, evidence, text_blob)
                        if value:
                            row[field] = value
                        field_outputs[field] = {
                            "value": value,
                            "evidence": evidence,
                        }

                response_payload = {
                    "table": table,
                    "nct_id": nct_id,
                    "missing_fields": batch_fields,
                    "response": response,
                    "parsed": parsed,
                    "normalized": field_outputs,
                }
                evidence_payload = {
                    "table": table,
                    "nct_id": nct_id,
                    "fields": field_outputs,
                }
                if len(batches) > 1:
                    response_payload["batch_index"] = batch_index + 1
                    response_payload["batch_total"] = len(batches)
                    evidence_payload["batch_index"] = batch_index + 1
                    evidence_payload["batch_total"] = len(batches)

                responses_file.write(json.dumps(response_payload, ensure_ascii=False) + "\n")
                evidence_file.write(json.dumps(evidence_payload, ensure_ascii=False) + "\n")
                if pretty_evidence is not None:
                    pretty_evidence.write(evidence_payload)

            writer.writerow(row)
            processed += 1
            if max_rows and processed >= max_rows:
                break
            if sleep_seconds:
                time.sleep(sleep_seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill CTG table fields using Dify.")
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Table name (Design, Stat_Reg, TargetPop, Drug, Others, Endpoints, Groups).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV from structured extractor.",
    )
    parser.add_argument(
        "--text-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/ctg_extract/ctg_text_blocks.jsonl",
        help="Text blocks JSONL/JSON from structured extractor.",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=PROJECT_ROOT / "data/raw/CSR-Vars 2026-01-08.xlsx",
        help="CSR-Vars 2026-01-08.xlsx path for field annotations.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV with LLM-filled values.",
    )
    parser.add_argument(
        "--responses-out",
        type=Path,
        default=None,
        help="JSONL with raw LLM responses for auditing.",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=None,
        help="JSONL with per-field evidence.",
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
        "--limit",
        type=int,
        default=0,
        help="Limit number of NCT IDs processed (applies after combining inputs).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters from text blocks to include in the prompt.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on number of rows processed.",
    )
    parser.add_argument(
        "--max-fields-per-call",
        type=int,
        default=0,
        help="Max fields per LLM call (0 means no batching).",
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None, prompt_config: Optional[PromptConfig] = None) -> None:
    args = parse_args(argv)
    table = normalize_table_name(args.table)
    paths = default_paths(table)

    if args.input_csv is None:
        args.input_csv = paths["input"]
    if args.output_csv is None:
        args.output_csv = paths["output"]
    if args.responses_out is None:
        args.responses_out = paths["responses"]
    if args.evidence_out is None:
        args.evidence_out = paths["evidence"]

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)

    if nct_ids:
        default_text = PROJECT_ROOT / "data/ctg_extract/ctg_text_blocks.jsonl"
        if len(nct_ids) == 1:
            nct_id = nct_ids[0]
            nct_paths = per_nct_paths(table, nct_id)
            if args.input_csv == paths["input"]:
                args.input_csv = nct_paths["input"]
            if args.output_csv == paths["output"]:
                args.output_csv = nct_paths["output"]
            if args.responses_out == paths["responses"]:
                args.responses_out = nct_paths["responses"]
            if args.evidence_out == paths["evidence"]:
                args.evidence_out = nct_paths["evidence"]
            if args.text_jsonl == default_text:
                per_nct_text = PROJECT_ROOT / "data/ctg_extract/by_nctid" / nct_id / "ctg_text_blocks.jsonl"
                if per_nct_text.exists():
                    args.text_jsonl = per_nct_text
                else:
                    suffix_text = suffix_path(default_text, nct_id)
                    args.text_jsonl = suffix_text if suffix_text.exists() else per_nct_text
        else:
            suffix = f"{nct_ids[0]}_plus{len(nct_ids) - 1}"
            if args.input_csv == paths["input"]:
                args.input_csv = suffix_path(paths["input"], suffix)
            if args.output_csv == paths["output"]:
                args.output_csv = suffix_path(paths["output"], suffix)
            if args.responses_out == paths["responses"]:
                args.responses_out = suffix_path(paths["responses"], suffix)
            if args.evidence_out == paths["evidence"]:
                args.evidence_out = suffix_path(paths["evidence"], suffix)
            if args.text_jsonl == default_text:
                args.text_jsonl = suffix_path(default_text, suffix)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.text_jsonl.exists():
        suffix = args.text_jsonl.suffix.lower()
        candidates: List[Path] = []
        if suffix == ".jsonl":
            candidates.append(args.text_jsonl.with_suffix(".json"))
        elif suffix == ".json":
            candidates.append(args.text_jsonl.with_suffix(".jsonl"))
        else:
            candidates.append(args.text_jsonl.with_suffix(".jsonl"))
            candidates.append(args.text_jsonl.with_suffix(".json"))
        for candidate in candidates:
            if candidate.exists():
                args.text_jsonl = candidate
                break
        else:
            raise FileNotFoundError(f"Missing text JSON/JSONL: {args.text_jsonl}")

    schema: Dict[str, Dict[str, str]] = {}
    if not args.xlsx.exists():
        raise FileNotFoundError(f"Missing xlsx: {args.xlsx}")
    schema = load_sheet_schema(args.xlsx, table)

    skip_fields = TABLE_SKIP_FIELDS.get(table, set())
    llm_fields = None
    if prompt_config and prompt_config.llm_fields is not None:
        llm_fields = set(prompt_config.llm_fields)
    else:
        raise ValueError(
            f"LLM fields must be defined in the table prompt config (LLM_FIELDS in fill_*.py): {table}"
        )

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

    pretty_evidence = None
    if nct_ids and len(nct_ids) == 1 and args.nct_id and not args.nct_csv:
        if args.evidence_out.suffix.lower() == ".jsonl":
            pretty_path = args.evidence_out.with_suffix(".json")
            pretty_evidence = JsonArrayWriter(pretty_path.open("w"))

    try:
        with args.responses_out.open("w") as f_resp, args.evidence_out.open("w") as f_evidence:
            process_table(
                table=table,
                input_path=args.input_csv,
                output_path=args.output_csv,
                text_blocks=text_blocks,
                client=client,
                allowed_nct_ids=allowed_nct_ids,
                max_chars=args.max_chars,
                max_rows=args.max_rows,
                conversation_id=args.conversation_id,
                responses_file=f_resp,
                evidence_file=f_evidence,
                sleep_seconds=args.sleep,
                schema=schema,
                prompt_config=prompt_config,
                llm_fields=llm_fields,
                skip_fields=skip_fields,
                max_fields_per_call=args.max_fields_per_call,
                pretty_evidence=pretty_evidence,
            )
    finally:
        if pretty_evidence is not None:
            pretty_evidence.close()


if __name__ == "__main__":
    main()
