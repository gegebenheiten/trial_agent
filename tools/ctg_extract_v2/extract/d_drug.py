from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import re
import xml.etree.ElementTree as ET

from .common import is_drug_like, join_values, normalize_whitespace, parse_interventions, set_if_present

COMBO_SPLIT_RE = re.compile(r"\s*(?:\+|/|&|\band\b|\bplus\b)\s*", re.IGNORECASE)
PAREN_CONTENT_RE = re.compile(r"\(([^)]+)\)")


def extract_d_drug_rows(root: ET.Element, fields: List[str], nct_id: str) -> Iterable[Dict[str, str]]:
    interventions = parse_interventions(root)
    for iv in interventions:
        if not is_drug_like(str(iv.get("type") or "")):
            continue
        row = {field: "" for field in fields}
        set_if_present(row, "StudyID", nct_id)
        set_if_present(row, "NCT_No", nct_id)
        set_if_present(row, "NCT_ID", nct_id)

        name = str(iv.get("name") or "")
        iv_type = str(iv.get("type") or "")
        other_names = join_values(iv.get("other_names", []))
        for field in ("Drug_Name", "Drug", "Intervention_Name"):
            set_if_present(row, field, name)
        for field in ("Intervention_Type", "Drug_Type"):
            set_if_present(row, field, iv_type)
        for field in ("Other_Names", "Alt_Names"):
            set_if_present(row, field, other_names)
        yield row


def normalize_drug_name(name: str) -> str:
    if not name:
        return ""
    text = normalize_whitespace(name).lower().replace("&", "and")
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def split_combo_name(name: str) -> List[str]:
    text = normalize_whitespace(name)
    if not text:
        return []
    parts = [part.strip() for part in COMBO_SPLIT_RE.split(text) if part.strip()]
    return parts or [text]


def name_variants(name: str) -> List[str]:
    text = normalize_whitespace(name)
    if not text:
        return []
    variants = [text]
    stripped = normalize_whitespace(re.sub(r"\([^)]*\)", " ", text))
    if stripped and stripped not in variants:
        variants.append(stripped)
    for alias in PAREN_CONTENT_RE.findall(text):
        alias_text = normalize_whitespace(alias)
        if alias_text and alias_text not in variants:
            variants.append(alias_text)
    return variants


def score_drugbank_entry(entry: Dict[str, object]) -> int:
    groups = [str(group).lower() for group in (entry.get("groups") or [])]
    return 1 if "approved" in groups else 0


def load_drugbank_minimal(path: Optional[Path]) -> List[Dict[str, object]]:
    if not path or not path.exists():
        return []
    entries: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def build_drugbank_index(entries: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for entry in entries:
        names = [entry.get("name")] + list(entry.get("synonyms") or [])
        for name in names:
            key = normalize_drug_name(str(name or ""))
            if not key:
                continue
            current = index.get(key)
            if current is None or score_drugbank_entry(entry) > score_drugbank_entry(current):
                index[key] = entry
    return index


def match_drugbank(name: str, index: Dict[str, Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not index or not name:
        return None
    best_entry = None
    best_score = -1
    for variant in name_variants(name):
        key = normalize_drug_name(variant)
        if not key:
            continue
        entry = index.get(key)
        if not entry:
            continue
        score = score_drugbank_entry(entry)
        if score > best_score:
            best_entry = entry
            best_score = score
    return best_entry


def index_properties(entries: List[Dict[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in entries or []:
        kind = normalize_whitespace(str(item.get("kind") or ""))
        value = normalize_whitespace(str(item.get("value") or ""))
        if kind and value and kind not in out:
            out[kind] = value
    return out


def format_targets(targets: List[Dict[str, object]]) -> str:
    entries = []
    for target in targets or []:
        name = normalize_whitespace(str(target.get("name") or ""))
        genes = [normalize_whitespace(g) for g in (target.get("genes") or []) if normalize_whitespace(g)]
        if name and genes:
            entries.append(f"{name} ({'; '.join(genes)})")
        elif name:
            entries.append(name)
        elif genes:
            entries.append("; ".join(genes))
    return join_values(entries)


def dedupe_names(values: Iterable[str], exclude: str = "") -> List[str]:
    seen = set()
    output = []
    exclude_key = normalize_drug_name(exclude)
    for value in values:
        text = normalize_whitespace(value)
        if not text:
            continue
        key = normalize_drug_name(text)
        if not key or key == exclude_key or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


@dataclass
class DrugRecord:
    study_ids: set = field(default_factory=set)
    ctg_names: set = field(default_factory=set)
    primary_names: set = field(default_factory=set)
    drugbank_entry: Optional[Dict[str, object]] = None


def sort_key_for_record(record: DrugRecord) -> str:
    entry = record.drugbank_entry or {}
    name = normalize_whitespace(str(entry.get("name") or ""))
    if name:
        return normalize_drug_name(name)
    if record.primary_names:
        primary = sorted(record.primary_names, key=lambda n: (len(n), n.lower()))[0]
        return normalize_drug_name(primary)
    return ""


def build_drug_row(
    fields: List[str],
    record: DrugRecord,
    study_id_value: str = "",
    list_mode: bool = True,
) -> Dict[str, str]:
    row = {field: "" for field in fields}

    if list_mode:
        study_ids = sorted(record.study_ids)
        set_if_present(row, "StudyID", json.dumps(study_ids, ensure_ascii=False))
    else:
        set_if_present(row, "StudyID", study_id_value)

    entry = record.drugbank_entry or {}
    drugbank_id = normalize_whitespace(str(entry.get("drugbank_id") or ""))
    set_if_present(row, "DrugBank_ID", drugbank_id)

    primary_candidates = sorted(record.primary_names, key=lambda n: (len(n), n.lower()))
    generic_name = primary_candidates[0] if primary_candidates else ""
    if not generic_name:
        ctg_candidates = sorted(record.ctg_names, key=lambda n: (len(n), n.lower()))
        generic_name = ctg_candidates[0] if ctg_candidates else ""
    set_if_present(row, "Generic_Name", generic_name)

    ctg_synonyms = dedupe_names(record.ctg_names, exclude=generic_name)
    set_if_present(row, "Brand_Name", join_values(ctg_synonyms))

    generic_name_db = normalize_whitespace(str(entry.get("name") or ""))
    set_if_present(row, "Generic_Name_DB", generic_name_db)
    db_synonyms = dedupe_names(entry.get("synonyms") or [], exclude=generic_name_db)
    set_if_present(row, "Brand_Name_DB", join_values(db_synonyms))

    properties = entry.get("properties") or {}
    experimental = index_properties(properties.get("experimental") or [])
    calculated = index_properties(properties.get("calculated") or [])

    molecular_weight = experimental.get("Molecular Weight") or calculated.get("Molecular Weight") or ""
    set_if_present(row, "Molecule_Size_DB", molecular_weight)
    smiles = calculated.get("SMILES") or experimental.get("SMILES") or ""
    set_if_present(row, "SMILES_DB", smiles)
    bioavailability = calculated.get("Bioavailability") or experimental.get("Bioavailability") or ""
    set_if_present(row, "F_bioav_DB", bioavailability)

    pharmacology = entry.get("pharmacology") or {}
    set_if_present(row, "V_d_DB", normalize_whitespace(str(pharmacology.get("volume_of_distribution") or "")))
    set_if_present(row, "Cl_DB", normalize_whitespace(str(pharmacology.get("clearance") or "")))
    set_if_present(row, "T_half_DB", normalize_whitespace(str(pharmacology.get("half_life") or "")))
    set_if_present(row, "Tox_General_DB", normalize_whitespace(str(pharmacology.get("toxicity") or "")))

    set_if_present(
        row,
        "Target_Engage_DB",
        format_targets(entry.get("targets") or []),
    )
    set_if_present(row, "Biochem_Change_DB", normalize_whitespace(str(entry.get("mechanism_of_action") or "")))
    set_if_present(row, "Physio_Change_DB", normalize_whitespace(str(pharmacology.get("pharmacodynamics") or "")))

    return row


class DrugTableBuilder:
    def __init__(self, fields: List[str], drugbank_jsonl: Optional[Path]) -> None:
        self.fields = fields
        self.drugbank_index = build_drugbank_index(load_drugbank_minimal(drugbank_jsonl))
        self.records: Dict[tuple[str, str], DrugRecord] = {}

    def add_from_root(self, root: ET.Element, nct_id: str) -> None:
        interventions = parse_interventions(root)
        for iv in interventions:
            if not is_drug_like(str(iv.get("type") or "")):
                continue
            name = normalize_whitespace(str(iv.get("name") or ""))
            other_names = [normalize_whitespace(n) for n in (iv.get("other_names") or []) if normalize_whitespace(n)]
            primary_parts = split_combo_name(name) if name else []
            if not primary_parts and other_names:
                primary_parts = split_combo_name(other_names[0])
            if not primary_parts:
                continue

            multi = len(primary_parts) > 1
            for part in primary_parts:
                candidate_names = [part]
                if not multi and other_names:
                    for other in other_names:
                        candidate_names.extend(split_combo_name(other))
                candidate_names = [n for n in candidate_names if n]
                if not candidate_names:
                    continue

                matched_entry = None
                for candidate in candidate_names:
                    matched_entry = match_drugbank(candidate, self.drugbank_index)
                    if matched_entry:
                        break

                if matched_entry:
                    key = ("db", str(matched_entry.get("drugbank_id") or ""))
                else:
                    key = ("ctg", normalize_drug_name(part))

                if not key[1]:
                    continue
                record = self.records.setdefault(key, DrugRecord())
                record.study_ids.add(nct_id)
                record.primary_names.add(part)
                record.ctg_names.update(candidate_names)
                if matched_entry and record.drugbank_entry is None:
                    record.drugbank_entry = matched_entry

    def _row_from_record(self, record: DrugRecord) -> Dict[str, str]:
        return build_drug_row(self.fields, record, list_mode=True)

    def _sorted_records(self) -> List[DrugRecord]:
        return sorted(self.records.values(), key=sort_key_for_record)

    def iter_rows(self) -> Iterable[Dict[str, str]]:
        for record in self._sorted_records():
            yield self._row_from_record(record)

    def iter_rows_for_nct(self, nct_id: str) -> Iterable[Dict[str, str]]:
        for record in self._sorted_records():
            if nct_id in record.study_ids:
                yield build_drug_row(self.fields, record, study_id_value=nct_id, list_mode=False)


def extract_d_drug_rows_with_index(
    root: ET.Element,
    fields: List[str],
    nct_id: str,
    drugbank_index: Dict[str, Dict[str, object]],
) -> Iterable[Dict[str, str]]:
    interventions = parse_interventions(root)
    records: Dict[tuple[str, str], DrugRecord] = {}

    for iv in interventions:
        if not is_drug_like(str(iv.get("type") or "")):
            continue
        name = normalize_whitespace(str(iv.get("name") or ""))
        other_names = [normalize_whitespace(n) for n in (iv.get("other_names") or []) if normalize_whitespace(n)]
        primary_parts = split_combo_name(name) if name else []
        if not primary_parts and other_names:
            primary_parts = split_combo_name(other_names[0])
        if not primary_parts:
            continue

        multi = len(primary_parts) > 1
        for part in primary_parts:
            candidate_names = [part]
            if not multi and other_names:
                for other in other_names:
                    candidate_names.extend(split_combo_name(other))
            candidate_names = [n for n in candidate_names if n]
            if not candidate_names:
                continue

            matched_entry = None
            for candidate in candidate_names:
                matched_entry = match_drugbank(candidate, drugbank_index)
                if matched_entry:
                    break

            if matched_entry:
                key = ("db", str(matched_entry.get("drugbank_id") or ""))
            else:
                key = ("ctg", normalize_drug_name(part))
            if not key[1]:
                continue

            record = records.setdefault(key, DrugRecord())
            record.study_ids.add(nct_id)
            record.primary_names.add(part)
            record.ctg_names.update(candidate_names)
            if matched_entry and record.drugbank_entry is None:
                record.drugbank_entry = matched_entry

    for record in sorted(records.values(), key=sort_key_for_record):
        yield build_drug_row(fields, record, study_id_value=nct_id, list_mode=False)
