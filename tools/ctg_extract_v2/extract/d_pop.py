from __future__ import annotations

from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET

from .common import (
    join_values,
    normalize_gender,
    parse_age_value,
    set_if_present,
    xml_text,
    xml_texts,
)


COMMON_FIELD_MAP = {
    "StudyID": "",
    "NCT_No": "",
    "NCT_ID": "",
    "Gender": "eligibility/gender",
    "Age_Min": "eligibility/minimum_age",
    "Age_Max": "eligibility/maximum_age",
}


def extract_d_pop_rows(root: ET.Element, fields: List[str], nct_id: str) -> Iterable[Dict[str, str]]:
    row = {field: "" for field in fields}

    for field, path in COMMON_FIELD_MAP.items():
        if field in {"StudyID", "NCT_No", "NCT_ID"}:
            set_if_present(row, field, nct_id)
            continue
        if field in {"Age_Min", "Age_Max"}:
            set_if_present(row, field, parse_age_value(xml_text(root, path)))
            continue
        if field == "Gender":
            set_if_present(row, field, normalize_gender(xml_text(root, path)))
            continue
        set_if_present(row, field, xml_text(root, path))

    conditions = xml_texts(root, "condition")
    if conditions:
        set_if_present(row, "Disease", join_values(conditions))

    criteria = xml_text(root, "eligibility/criteria/textblock")
    for field in ("Eligibility", "Eligibility_Criteria", "Inclusion_Exclusion"):
        set_if_present(row, field, criteria)

    yield row
