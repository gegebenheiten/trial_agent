from __future__ import annotations

from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET
import json

from .common import normalize_whitespace, set_if_present, xml_text


def normalize_tag(tag: str) -> str:
    if not tag:
        return ""
    return tag.split("}", 1)[-1]


def outcome_type_label(text: str) -> str:
    lower = (text or "").strip().lower()
    if not lower:
        return ""
    if "key" in lower and "secondary" in lower:
        return "Key Secondary"
    if "primary" in lower:
        return "Primary"
    if "secondary" in lower:
        return "Secondary"
    return ""


CANONICAL_ANALYSIS_FIELDS = {
    "analysis_id": "analysis_id",
    "groups_desc": "groups_desc",
    "non_inferiority_type": "non_inferiority_type",
    "method": "method",
    "param_type": "param_type",
    "param_value": "param_value",
    "ci_percent": "ci_percent",
    "ci_n_sides": "ci_n_sides",
    "ci_lower_limit": "ci_lower_limit",
    "ci_upper_limit": "ci_upper_limit",
    "p_value": "p_value",
    "p_value_modifier": "p_value_modifier",
    "p_value_desc": "p_value_desc",
    "p_value_description": "p_value_description",
}


def append_value(target: Dict[str, object], key: str, value: str) -> None:
    if key in target:
        existing = target[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            target[key] = [existing, value]
    else:
        target[key] = value


def analysis_to_dict(analysis: ET.Element) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    extra: Dict[str, object] = {}

    for attr, value in analysis.attrib.items():
        clean = normalize_whitespace(value)
        if not clean:
            continue
        key = CANONICAL_ANALYSIS_FIELDS.get(attr)
        if key:
            payload[key] = clean
        else:
            extra[f"@{attr}"] = clean

    group_ids = []
    for group in analysis.findall("group_id_list/group_id"):
        value = normalize_whitespace(group.text or "")
        if value:
            group_ids.append(value)
    if group_ids:
        payload["group_id_list"] = group_ids

    for child in list(analysis):
        tag = normalize_tag(child.tag)
        if tag == "group_id_list":
            continue
        value = normalize_whitespace(child.text or "")
        if not value:
            continue
        key = CANONICAL_ANALYSIS_FIELDS.get(tag)
        if key:
            append_value(payload, key, value)
        else:
            append_value(extra, tag, value)

    if extra:
        payload["extra"] = extra
    return payload


def extract_r_study_rows(root: ET.Element, fields: List[str], nct_id: str) -> Iterable[Dict[str, str]]:
    outcomes = root.findall("clinical_results/outcome_list/outcome")

    start_date = xml_text(root, "start_date")
    completion_date = xml_text(root, "completion_date")
    primary_completion_date = xml_text(root, "primary_completion_date")
    overall_status = xml_text(root, "overall_status")
    why_stopped = xml_text(root, "why_stopped")

    for outcome in outcomes:
        row = {field: "" for field in fields}
        set_if_present(row, "StudyID", nct_id)

        set_if_present(row, "Start_Date", start_date)
        set_if_present(row, "Complet_Date", completion_date)
        if "terminated" in overall_status.lower():
            set_if_present(row, "Termin_Date", completion_date or primary_completion_date)
        set_if_present(row, "Date_End", completion_date or primary_completion_date)
        set_if_present(row, "Study_Status", overall_status)
        set_if_present(row, "Termination", why_stopped)

        outcome_title = xml_text(outcome, "title")
        set_if_present(row, "Outcome", outcome_title)
        set_if_present(row, "EP_Name", outcome_title)
        set_if_present(row, "EP_type", outcome_type_label(xml_text(outcome, "type")))
        set_if_present(row, "EP_Pop", xml_text(outcome, "population"))
        set_if_present(row, "EP_time_frame", xml_text(outcome, "time_frame"))
        set_if_present(row, "EP_description", xml_text(outcome, "description"))

        analysis_list = []
        for analysis in outcome.findall("analysis_list/analysis"):
            entry = analysis_to_dict(analysis)
            if entry:
                analysis_list.append(entry)
        if analysis_list:
            set_if_present(row, "SA_json", json.dumps(analysis_list, ensure_ascii=False))

        yield row
