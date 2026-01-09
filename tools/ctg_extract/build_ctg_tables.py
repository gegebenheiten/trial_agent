#!/usr/bin/env python3
"""
Build CTG tables from ClinicalTrials.gov XML files.
Outputs Design/Stat_Reg/TargetPop/Drug/Others and Results Endpoints/Groups.
Optionally save text blocks for later LLM extraction as JSONL.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WHITESPACE_RE = re.compile(r"\s+")
EUDRACT_RE = re.compile(r"\b\d{4}-\d{6}-\d{2}\b")
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")

DESIGN_TABLES = ("Design", "Stat_Reg", "TargetPop", "Drug", "Others")
RESULT_TABLES = ("Endpoints", "Groups")
ALL_TABLES = DESIGN_TABLES + RESULT_TABLES

CONTROL_DRUG_TYPES = {
    "drug",
    "biological",
    "combination product",
}
CONTROL_ARM_KEYWORDS = (
    "placebo",
    "active",
    "sham",
    "no intervention",
    "histor",
)
EXPERIMENTAL_ARM_KEYWORDS = ("experimental",)
PLACEBO_LIKE_KEYWORDS = ("placebo", "sham", "vehicle")
BACKGROUND_THERAPY_PATTERNS = (
    r"\bstandard of care\b",
    r"\bbest supportive care\b",
    r"\bsupportive care\b",
    r"\bsoc\b",
    r"\bbsc\b",
)
ACADEMIC_SPONSOR_KEYWORDS = (
    "university",
    "college",
    "school",
    "hospital",
    "academy",
    "institute",
)

TITLE_MATCH_THRESHOLD = 90
DESC_MATCH_THRESHOLD = 80
FUZZY_MATCH_MARGIN = 3

ENDPOINT_MATCH_FIELDS = [
    "endpoint_group_id",
    "endpoint_group_title_raw",
    "endpoint_group_desc_raw",
    "endpoint_group_title_norm",
    "endpoint_group_desc_norm",
    "matched_arm_id",
    "arm_group_title_raw",
    "arm_group_desc_raw",
    "arm_group_title_norm",
    "arm_group_desc_norm",
    "match_type",
    "title_score",
    "desc_score",
    "debug_notes",
    "top_candidates_json",
]

GROUP_ARM_MAPPING_FIELDS = [
    "nct_id",
    "endpoint_name",
    "endpoint_type",
    "endpoint_group_id",
    "endpoint_group_title_raw",
    "endpoint_group_desc_raw",
    "endpoint_group_title_norm",
    "endpoint_group_desc_norm",
    "matched_arm_id",
    "arm_group_title_raw",
    "arm_group_desc_raw",
    "arm_group_title_norm",
    "arm_group_desc_norm",
    "match_type",
    "title_score",
    "desc_score",
    "debug_notes",
    "top_candidates_json",
]

GROUPS_EXTRA_FIELDS = [
    "group_id_raw",
    "group_title_raw",
    "group_desc_raw",
    "N_group",
    "count_male",
    "count_female",
    "count_white",
    "count_black",
    "count_asian",
    "count_other",
    "Missing_K",
]


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_match_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_whitespace(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def normalize_country(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


def join_values(values: Iterable[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return "; ".join(ordered)


def extend_fields(fields: List[str], extra_fields: Iterable[str]) -> List[str]:
    seen = set(fields)
    for field in extra_fields:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    return fields


def safe_int(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def safe_float(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def format_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    ratio = numerator / denominator
    return f"{ratio:.4f}".rstrip("0").rstrip(".")


def parse_range_values(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None, None
    try:
        return float(numbers[0]), float(numbers[1])
    except ValueError:
        return None, None


def parse_age_value(text: str) -> str:
    if not text:
        return ""
    lowered = text.strip().lower()
    if lowered in {"n/a", "na", "not applicable", "none"} or "no limit" in lowered:
        return ""
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return ""
    value = float(match.group(0))
    if "month" in lowered:
        years = value / 12
    elif "week" in lowered:
        years = value / 52
    elif "day" in lowered:
        years = value / 365.25
    else:
        years = value
    if abs(years - round(years)) < 1e-6:
        return str(int(round(years)))
    return f"{years:.2f}".rstrip("0").rstrip(".")


def normalize_gender(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown")):
        return ""
    if "male" in lower and "female" in lower:
        return "Both"
    if lower == "male":
        return "Male"
    if lower == "female":
        return "Female"
    if lower in {"all", "both"}:
        return "Both"
    return ""


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


def load_sheet_fields(xlsx_path: Path, sheet_name: str) -> List[str]:
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
        if not key_col:
            raise ValueError(f"StudyID/Variable column not found in sheet '{sheet_name}'.")

        fields: List[str] = []
        seen = set()
        for cells in parsed_rows[header_idx + 1 :]:
            raw = cells.get(key_col)
            if raw is None:
                continue
            name = str(raw).strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                fields.append(name)
        if not fields:
            raise ValueError(f"No fields found in sheet '{sheet_name}'.")
        rename = {
            "Report_Date": "Study_First_Posted_Date",
            "Date_Report": "Results_Posted_Or_Updated_Date",
            "Fomulation": "Formulation",
        }
        fields = [rename.get(name, name) for name in fields]
        return fields


def region_sets() -> Dict[str, set]:
    regions: Dict[str, List[str]] = {
        "NA": ["United States", "Canada"],
        "WEU": [
            "Austria",
            "Belgium",
            "France",
            "Germany",
            "Greece",
            "Italy",
            "Liechtenstein",
            "Luxembourg",
            "Monaco",
            "Netherlands",
            "Portugal",
            "Spain",
            "Switzerland",
        ],
        "EEU": [
            "Albania",
            "Belarus",
            "Bulgaria",
            "Czech Republic",
            "Czechia",
            "Estonia",
            "Hungary",
            "Latvia",
            "Lithuania",
            "Moldova",
            "Poland",
            "Romania",
            "Russia",
            "Russian Federation",
            "Serbia",
            "Slovakia",
            "Ukraine",
        ],
        "AF": [
            "Algeria",
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cabo Verde",
            "Cape Verde",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Comoros",
            "Congo",
            "Congo, Republic of the",
            "Congo, The Democratic Republic of the",
            "Democratic Republic of the Congo",
            "Djibouti",
            "Egypt",
            "Equatorial Guinea",
            "Eritrea",
            "Eswatini",
            "Swaziland",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Ivory Coast",
            "Cote d'Ivoire",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Mauritius",
            "Morocco",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "Sao Tome and Principe",
            "Senegal",
            "Seychelles",
            "Sierra Leone",
            "Somalia",
            "South Africa",
            "South Sudan",
            "Sudan",
            "Tanzania",
            "Tanzania, United Republic of",
            "Togo",
            "Tunisia",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ],
        "AP": [
            "Afghanistan",
            "Armenia",
            "Azerbaijan",
            "Bahrain",
            "Bangladesh",
            "Bhutan",
            "Brunei",
            "Brunei Darussalam",
            "Cambodia",
            "China",
            "Cyprus",
            "Georgia",
            "Hong Kong",
            "Macao",
            "India",
            "Indonesia",
            "Iran",
            "Iran, Islamic Republic of",
            "Iraq",
            "Israel",
            "Japan",
            "Jordan",
            "Kazakhstan",
            "Kuwait",
            "Kyrgyzstan",
            "Laos",
            "Lao People's Democratic Republic",
            "Lebanon",
            "Malaysia",
            "Maldives",
            "Mongolia",
            "Myanmar",
            "Burma",
            "Nepal",
            "North Korea",
            "Korea, Democratic People's Republic of",
            "Oman",
            "Pakistan",
            "Palestine",
            "Palestinian Territory",
            "Philippines",
            "Qatar",
            "Saudi Arabia",
            "Singapore",
            "South Korea",
            "Korea, Republic of",
            "Sri Lanka",
            "Syria",
            "Syrian Arab Republic",
            "Taiwan",
            "Taiwan, Province of China",
            "Tajikistan",
            "Thailand",
            "Timor-Leste",
            "Turkey",
            "Turkmenistan",
            "United Arab Emirates",
            "Uzbekistan",
            "Vietnam",
            "Viet Nam",
            "Yemen",
            "Australia",
            "New Zealand",
            "Papua New Guinea",
            "Fiji",
            "Solomon Islands",
            "Vanuatu",
            "Samoa",
            "Tonga",
            "Tuvalu",
            "Kiribati",
            "Nauru",
            "Palau",
            "Micronesia",
            "Micronesia, Federated States of",
            "Marshall Islands",
            "American Samoa",
            "Guam",
            "Northern Mariana Islands",
        ],
    }

    normalized: Dict[str, set] = {}
    for region, names in regions.items():
        normalized[region] = {normalize_country(n) for n in names}
    return normalized


def region_for_country(country: str, region_map: Dict[str, set]) -> str:
    key = normalize_country(country)
    if not key:
        return ""
    for region, names in region_map.items():
        if key in names:
            return region
    return ""


def normalize_intervention_type(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    lowered = lower.replace("_", " ").replace("-", " ").replace("/", " ")
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    if "combination" in lowered and "product" in lowered:
        return "combination product"
    if "biologic" in lowered or "biological" in lowered or "vaccine" in lowered:
        return "biological"
    if "drug" in lowered:
        return "drug"
    return lowered


def normalize_intervention_model(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown", "not provided")):
        return ""
    return re.sub(r"[\s\-_]+", "", lower)


def arm_type_matches(arm_type: str, keywords: Iterable[str]) -> bool:
    lower = (arm_type or "").lower()
    return any(keyword in lower for keyword in keywords)


def is_drug_like(iv_type: str) -> bool:
    return normalize_intervention_type(iv_type) in CONTROL_DRUG_TYPES


def is_placebo_like_name(name: str) -> bool:
    lower = (name or "").lower()
    return any(keyword in lower for keyword in PLACEBO_LIKE_KEYWORDS)


def is_background_like_name(name: str) -> bool:
    lower = (name or "").lower()
    return any(re.search(pattern, lower) for pattern in BACKGROUND_THERAPY_PATTERNS)


def xml_text(root: ET.Element, path: str) -> str:
    return normalize_whitespace(root.findtext(path) or "")


def xml_texts(root: ET.Element, path: str) -> List[str]:
    values = []
    for el in root.findall(path):
        value = normalize_whitespace(el.text or "")
        if value:
            values.append(value)
    return values


def extract_eudract_id(text: str) -> str:
    if not text:
        return ""
    match = EUDRACT_RE.search(text)
    return match.group(0) if match else ""


def parse_secondary_ids(root: ET.Element) -> List[str]:
    return xml_texts(root, "id_info/secondary_id")


def parse_interventions(root: ET.Element) -> List[Dict[str, List[str] | str]]:
    interventions = []
    for iv in root.findall("intervention"):
        name = xml_text(iv, "intervention_name")
        other_names = [
            normalize_whitespace(o.text or "")
            for o in iv.findall("other_name")
            if normalize_whitespace(o.text or "")
        ]
        labels = [
            normalize_whitespace(l.text or "")
            for l in iv.findall("arm_group_label")
            if normalize_whitespace(l.text or "")
        ]
        interventions.append(
            {
                "type": normalize_intervention_type(xml_text(iv, "intervention_type")),
                "name": name,
                "description": xml_text(iv, "description"),
                "other_names": other_names,
                "labels": labels,
            }
        )
    return interventions


def parse_arm_groups(root: ET.Element) -> List[Dict[str, str]]:
    arms = []
    for arm in root.findall("arm_group"):
        arms.append(
            {
                "label": xml_text(arm, "arm_group_label"),
                "type": xml_text(arm, "arm_group_type"),
                "description": xml_text(arm, "description"),
            }
        )
    return arms


def parse_location_centers(root: ET.Element) -> List[Tuple[str, str, str]]:
    centers: List[Tuple[str, str, str]] = []
    for loc in root.findall("location"):
        name = xml_text(loc, "facility/name")
        city = xml_text(loc, "facility/address/city")
        if not city:
            city = xml_text(loc, "facility/address/state") or xml_text(
                loc, "facility/address/state_province"
            )
        country = xml_text(loc, "facility/address/country")
        if not (name or city or country):
            continue
        centers.append((name, city, country))
    seen = set()
    unique: List[Tuple[str, str, str]] = []
    for center in centers:
        if center in seen:
            continue
        seen.add(center)
        unique.append(center)
    return unique


def parse_locations(root: ET.Element) -> List[str]:
    countries = []
    for loc in root.findall("location"):
        country = xml_text(loc, "facility/address/country")
        if country:
            countries.append(country)
    return countries


def parse_officials(root: ET.Element) -> List[Dict[str, str]]:
    officials = []
    for official in root.findall("overall_official"):
        role = xml_text(official, "role")
        parts = [
            xml_text(official, "first_name"),
            xml_text(official, "middle_name"),
            xml_text(official, "last_name"),
        ]
        combined = normalize_whitespace(" ".join(p for p in parts if p))
        if not combined:
            combined = xml_text(official, "last_name")
        if combined:
            officials.append({"name": combined, "role": role})
    return officials


def parse_sponsors(root: ET.Element) -> List[str]:
    sponsors = []
    lead = xml_text(root, "sponsors/lead_sponsor/agency")
    if lead:
        sponsors.append(lead)
    for col in root.findall("sponsors/collaborator/agency"):
        name = normalize_whitespace(col.text or "")
        if name:
            sponsors.append(name)
    return sponsors


def parse_sponsor_entries(root: ET.Element) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    lead_node = root.find("sponsors/lead_sponsor")
    lead = {
        "agency": xml_text(lead_node, "agency") if lead_node is not None else "",
        "agency_class": xml_text(lead_node, "agency_class") if lead_node is not None else "",
    }
    collaborators = []
    for col in root.findall("sponsors/collaborator"):
        entry = {
            "agency": xml_text(col, "agency"),
            "agency_class": xml_text(col, "agency_class"),
        }
        if entry["agency"] or entry["agency_class"]:
            collaborators.append(entry)
    return lead, collaborators


def classify_sponsor_type(agency: str, agency_class: str) -> str:
    class_lower = normalize_whitespace(agency_class).lower()
    if class_lower in {"nih", "u.s. fed", "us fed", "u.s. federal", "federal"}:
        return "Government"
    if class_lower == "industry":
        return "Industry"
    name_lower = normalize_whitespace(agency).lower()
    if any(keyword in name_lower for keyword in ACADEMIC_SPONSOR_KEYWORDS):
        return "Academic"
    return ""


def parse_outcomes(root: ET.Element, tag: str) -> List[Dict[str, str]]:
    outcomes = []
    for outcome in root.findall(tag):
        outcomes.append(
            {
                "measure": xml_text(outcome, "measure"),
                "time_frame": xml_text(outcome, "time_frame"),
                "description": xml_text(outcome, "description"),
            }
        )
    return outcomes


def parse_results_participant_flow(root: ET.Element) -> Dict[str, object]:
    results = root.find("clinical_results")
    if results is None:
        return {}
    flow = results.find("participant_flow")
    if flow is None:
        return {}
    groups = parse_results_group_list(flow)
    return {
        "recruitment_details": xml_text(flow, "recruitment_details"),
        "pre_assignment_details": xml_text(flow, "pre_assignment_details"),
        "groups": groups,
    }


def parse_results_group_list(section: ET.Element) -> List[Dict[str, str]]:
    groups = []
    for group in section.findall("group_list/group"):
        groups.append(
            {
                "group_id": (group.attrib.get("group_id") or "").strip(),
                "title": xml_text(group, "title"),
                "description": xml_text(group, "description"),
            }
        )
    return groups


def group_dedup_key(group_id: str, title: str) -> str:
    key = normalize_label(title)
    if key:
        return key
    return group_id.strip() or "overall"


def dedupe_groups(groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for group in groups:
        group_id = group.get("group_id", "")
        title = group.get("title", "")
        key = group_dedup_key(group_id, title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(group)
    return deduped


def parse_results_groups_preferred(root: ET.Element) -> List[Dict[str, str]]:
    results = root.find("clinical_results")
    if results is None:
        return []
    flow = results.find("participant_flow")
    groups: List[Dict[str, str]] = []
    if flow is not None:
        groups.extend(parse_results_group_list(flow))
    baseline = results.find("baseline")
    if baseline is not None:
        groups.extend(parse_results_group_list(baseline))
    if not groups:
        return []
    return dedupe_groups(groups)


def parse_results_groups_union(root: ET.Element) -> List[Dict[str, str]]:
    results = root.find("clinical_results")
    if results is None:
        return []
    groups: List[Dict[str, str]] = []
    flow = results.find("participant_flow")
    if flow is not None:
        groups.extend(parse_results_group_list(flow))
    baseline = results.find("baseline")
    if baseline is not None:
        groups.extend(parse_results_group_list(baseline))
    if not groups:
        return []
    return dedupe_groups(groups)


def count_results_groups(root: ET.Element) -> int:
    groups = parse_results_groups_preferred(root)
    if not groups:
        return 0
    return len(dedupe_groups(groups))


def parse_participant_flow_started_counts(root: ET.Element) -> Dict[str, str]:
    results = root.find("clinical_results")
    if results is None:
        return {}
    flow = results.find("participant_flow")
    if flow is None:
        return {}

    def counts_from_milestone(milestone: ET.Element) -> Dict[str, str]:
        counts: Dict[str, str] = {}
        for participant in milestone.findall("participants_list/participants"):
            group_id = (participant.attrib.get("group_id") or "").strip()
            count = (participant.attrib.get("count") or "").strip()
            if not count:
                count = normalize_whitespace(participant.text or "")
            if group_id and count:
                counts[group_id] = count
        return counts

    milestones = flow.findall("period_list/period/milestone_list/milestone")
    priorities = ("started", "enrolled", "randomized", "assigned")
    for label in priorities:
        for milestone in milestones:
            title = xml_text(milestone, "title").lower()
            if label in title:
                counts = counts_from_milestone(milestone)
                if counts:
                    return counts
    return {}


def parse_results_baseline(root: ET.Element) -> Dict[str, object]:
    results = root.find("clinical_results")
    if results is None:
        return {}
    baseline = results.find("baseline")
    if baseline is None:
        return {}
    groups = parse_results_group_list(baseline)
    return {
        "population": xml_text(baseline, "population"),
        "groups": groups,
    }


def collect_measure_entries(measure: ET.Element) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen: set = set()

    def add_entry(category_title: str, measurement: ET.Element) -> None:
        if id(measurement) in seen:
            return
        seen.add(id(measurement))
        group_id = (measurement.attrib.get("group_id") or "").strip()
        value = (measurement.attrib.get("value") or "").strip()
        if not value:
            value = normalize_whitespace(measurement.text or "")
        spread = (measurement.attrib.get("spread") or "").strip()
        lower = (measurement.attrib.get("lower_limit") or "").strip()
        upper = (measurement.attrib.get("upper_limit") or "").strip()
        entries.append(
            {
                "category": normalize_whitespace(category_title or ""),
                "group_id": group_id,
                "value": value,
                "spread": spread,
                "lower": lower,
                "upper": upper,
            }
        )

    for cls in measure.findall("class_list/class"):
        cls_title = xml_text(cls, "title")
        categories = cls.findall("category_list/category")
        if categories:
            for cat in categories:
                cat_title = xml_text(cat, "title") or cls_title
                for measurement in cat.findall("measurement_list/measurement"):
                    add_entry(cat_title, measurement)
        else:
            for measurement in cls.findall("measurement_list/measurement"):
                add_entry(cls_title, measurement)

    for cat in measure.findall("category_list/category"):
        cat_title = xml_text(cat, "title")
        for measurement in cat.findall("measurement_list/measurement"):
            add_entry(cat_title, measurement)

    for measurement in measure.findall("measurement_list/measurement"):
        add_entry("", measurement)

    return entries


def classify_sex_category(text: str) -> str:
    key = normalize_label(text)
    if "female" in key:
        return "female"
    if "male" in key:
        return "male"
    return ""


def classify_race_category(text: str) -> str:
    key = normalize_label(text)
    if not key:
        return ""
    if "white" in key:
        return "white"
    if "black" in key or "african" in key:
        return "black"
    if "asian" in key:
        return "asian"
    if any(
        token in key
        for token in (
            "other",
            "american indian",
            "alaska native",
            "native hawaiian",
            "pacific islander",
            "multiple",
            "unknown",
            "not reported",
            "not specified",
        )
    ):
        return "other"
    return ""


def classify_region_category(text: str, region_map: Optional[Dict[str, set]] = None) -> str:
    key = normalize_label(text)
    if "north america" in key or key == "na":
        return "na"
    if "europe" in key or key == "eu" or "western europe" in key or "eastern europe" in key:
        return "eu"
    if "asia" in key or "pacific" in key or "apac" in key:
        return "ap"
    if "africa" in key:
        return "af"
    if region_map:
        region = region_for_country(text, region_map)
        if region in {"WEU", "EEU"}:
            return "eu"
        if region == "NA":
            return "na"
        if region == "AP":
            return "ap"
        if region == "AF":
            return "af"
    return ""


def classify_ecog_category(text: str) -> Optional[int]:
    key = normalize_label(text)
    if not key:
        return None
    digits = re.findall(r"\b[0-5]\b", key)
    if not digits:
        return None
    unique_digits = sorted(set(digits))
    if len(unique_digits) != 1:
        return None
    return int(unique_digits[0])


def is_total_group_title(title: str) -> bool:
    key = normalize_label(title)
    return key in {"total", "overall", "all participants", "total participants"}


def extract_group_demographics(root: ET.Element) -> List[Dict[str, object]]:
    results = root.find("clinical_results")
    if results is None:
        return []
    baseline = results.find("baseline")
    if baseline is None:
        return []
    region_map = region_sets()

    groups = dedupe_groups(parse_results_group_list(baseline))
    if len(groups) > 1:
        groups = [group for group in groups if not is_total_group_title(group.get("title", ""))]

    entries: List[Dict[str, object]] = []
    entries_by_id: Dict[str, Dict[str, object]] = {}
    for group in groups:
        group_id = group.get("group_id", "")
        entry: Dict[str, object] = {
            "group_id_raw": group_id,
            "group_title_raw": group.get("title", ""),
            "group_desc_raw": group.get("description", ""),
            "count_male": None,
            "count_female": None,
            "count_white": None,
            "count_black": None,
            "count_asian": None,
            "count_other": None,
            "count_na": None,
            "count_ap": None,
            "count_eu": None,
            "count_af": None,
            "count_ecog0": None,
            "count_ecog1": None,
            "count_ecog2": None,
            "count_ecog3": None,
            "count_ecog4": None,
            "count_ecog5": None,
            "med_age": None,
            "min_age": None,
            "max_age": None,
        }
        entries.append(entry)
        if group_id:
            entries_by_id[group_id] = entry

    min_age_elig = parse_age_value(xml_text(root, "eligibility/minimum_age"))
    max_age_elig = parse_age_value(xml_text(root, "eligibility/maximum_age"))

    for measure in baseline.findall("measure_list/measure"):
        title = normalize_label(xml_text(measure, "title"))
        param = normalize_label(xml_text(measure, "param"))
        dispersion = normalize_label(xml_text(measure, "dispersion"))
        entries_list = collect_measure_entries(measure)

        if "sex" in title:
            for item in entries_list:
                category = classify_sex_category(item["category"])
                if not category:
                    continue
                count = safe_int(item["value"])
                if count is None:
                    continue
                entry = entries_by_id.get(item["group_id"])
                if not entry:
                    continue
                key = f"count_{category}"
                entry[key] = (entry.get(key) or 0) + count
        elif "race" in title or "ethnic" in title:
            has_core_race = any(
                classify_race_category(item["category"]) in {"white", "black", "asian"}
                for item in entries_list
            )
            if not has_core_race:
                continue
            for item in entries_list:
                category = classify_race_category(item["category"])
                if not category:
                    continue
                count = safe_int(item["value"])
                if count is None:
                    continue
                entry = entries_by_id.get(item["group_id"])
                if not entry:
                    continue
                key = f"count_{category}"
                entry[key] = (entry.get(key) or 0) + count
        elif "region" in title or "geograph" in title or "country" in title:
            for item in entries_list:
                category = classify_region_category(item["category"], region_map)
                if not category:
                    continue
                count = safe_int(item["value"])
                if count is None:
                    continue
                entry = entries_by_id.get(item["group_id"])
                if not entry:
                    continue
                key = f"count_{category}"
                entry[key] = (entry.get(key) or 0) + count
        elif "ecog" in title or "eastern cooperative oncology group" in title:
            for item in entries_list:
                score = classify_ecog_category(item["category"])
                if score is None:
                    continue
                count = safe_int(item["value"])
                if count is None:
                    continue
                entry = entries_by_id.get(item["group_id"])
                if not entry:
                    continue
                key = f"count_ecog{score}"
                entry[key] = (entry.get(key) or 0) + count
        elif "age" in title:
            for item in entries_list:
                entry = entries_by_id.get(item["group_id"])
                if not entry:
                    continue
                value = safe_float(item["value"])
                if value is None:
                    continue
                if "median" in param:
                    entry["med_age"] = value
                elif "mean" in param and entry.get("med_age") is None:
                    entry["med_age"] = value
                if "minimum" in param or param == "min":
                    entry["min_age"] = value
                if "maximum" in param or param == "max":
                    entry["max_age"] = value

                if "range" in dispersion or "range" in param or "range" in title:
                    lower = safe_float(item["lower"])
                    upper = safe_float(item["upper"])
                    if lower is None or upper is None:
                        lower, upper = parse_range_values(item["spread"])
                    if lower is not None and entry.get("min_age") is None:
                        entry["min_age"] = lower
                    if upper is not None and entry.get("max_age") is None:
                        entry["max_age"] = upper

    for entry in entries:
        male = entry.get("count_male")
        female = entry.get("count_female")
        if male is not None and female is not None:
            n_group = male + female
        else:
            n_group = None

        race_counts = [
            entry.get("count_white"),
            entry.get("count_black"),
            entry.get("count_asian"),
            entry.get("count_other"),
        ]
        race_total = sum(count for count in race_counts if count is not None)
        denom_race = n_group if n_group and n_group > 0 else (race_total if race_total > 0 else None)

        region_counts = [
            entry.get("count_na"),
            entry.get("count_ap"),
            entry.get("count_eu"),
            entry.get("count_af"),
        ]
        region_total = sum(count for count in region_counts if count is not None)
        denom_region = n_group if n_group and n_group > 0 else (region_total if region_total > 0 else None)

        ecog_counts = [entry.get(f"count_ecog{score}") for score in range(6)]
        ecog_total = sum(count for count in ecog_counts if count is not None)
        denom_ecog = n_group if n_group and n_group > 0 else (ecog_total if ecog_total > 0 else None)

        entry["N_group"] = str(n_group) if n_group is not None else ""
        entry["Prop_Male"] = format_ratio(male, n_group) if male is not None else ""
        entry["Prop_White"] = format_ratio(entry.get("count_white"), denom_race)
        entry["Prop_Black"] = format_ratio(entry.get("count_black"), denom_race)
        entry["Prop_Asian"] = format_ratio(entry.get("count_asian"), denom_race)
        entry["Prop_NA"] = format_ratio(entry.get("count_na"), denom_region)
        entry["Prop_AP"] = format_ratio(entry.get("count_ap"), denom_region)
        entry["Prop_EU"] = format_ratio(entry.get("count_eu"), denom_region)
        entry["Prop_AF"] = format_ratio(entry.get("count_af"), denom_region)
        entry["Prop_ECOG0"] = format_ratio(entry.get("count_ecog0"), denom_ecog)
        entry["Prop_ECOG1"] = format_ratio(entry.get("count_ecog1"), denom_ecog)
        entry["Prop_ECOG2"] = format_ratio(entry.get("count_ecog2"), denom_ecog)
        entry["Prop_ECOG3"] = format_ratio(entry.get("count_ecog3"), denom_ecog)

        if entry.get("med_age") is not None:
            entry["Med_Age"] = format_number(entry.get("med_age"))
        else:
            entry["Med_Age"] = ""
        if entry.get("min_age") is not None:
            entry["Min_Age"] = format_number(entry.get("min_age"))
        else:
            entry["Min_Age"] = min_age_elig
        if entry.get("max_age") is not None:
            entry["Max_Age"] = format_number(entry.get("max_age"))
        else:
            entry["Max_Age"] = max_age_elig

        heter_values = []
        for key in ("count_white", "count_black", "count_asian", "count_other"):
            count = entry.get(key)
            if count is None or denom_race is None or denom_race == 0:
                continue
            heter_values.append(count / denom_race)
        if heter_values and denom_race:
            heter_index = 1.0 - sum(value * value for value in heter_values)
            entry["Heter_Index"] = format_number(heter_index)
        else:
            entry["Heter_Index"] = ""

    return entries


def index_group_demographics(
    entries: List[Dict[str, object]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    by_id: Dict[str, Dict[str, object]] = {}
    by_title: Dict[str, Dict[str, object]] = {}
    for entry in entries:
        group_id = str(entry.get("group_id_raw") or "").strip()
        if group_id and group_id not in by_id:
            by_id[group_id] = entry
        title_key = normalize_label(str(entry.get("group_title_raw") or ""))
        if title_key and title_key not in by_title:
            by_title[title_key] = entry
    return by_id, by_title


def match_group_demographics(
    group: Dict[str, str],
    entries: List[Dict[str, object]],
    by_id: Dict[str, Dict[str, object]],
    by_title: Dict[str, Dict[str, object]],
    threshold: int = 90,
) -> Optional[Dict[str, object]]:
    group_id = str(group.get("group_id") or "").strip()
    if group_id and group_id in by_id:
        return by_id[group_id]
    title_key = normalize_label(str(group.get("title") or ""))
    if title_key and title_key in by_title:
        return by_title[title_key]
    if not title_key:
        return None
    best_entry = None
    best_score = 0
    for entry in entries:
        entry_title = normalize_label(str(entry.get("group_title_raw") or ""))
        if not entry_title:
            continue
        score = fuzzy_score(title_key, entry_title)
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry and best_score >= threshold:
        return best_entry
    return None


def parse_results_outcomes(root: ET.Element) -> List[Dict[str, object]]:
    results = root.find("clinical_results")
    if results is None:
        return []
    outcomes = []
    for outcome in results.findall("outcome_list/outcome"):
        groups = []
        for group in outcome.findall("group_list/group"):
            groups.append(
                {
                    "group_id": (group.attrib.get("group_id") or "").strip(),
                    "title": xml_text(group, "title"),
                    "description": xml_text(group, "description"),
                }
            )
        analyses = []
        for analysis in outcome.findall("analysis_list/analysis"):
            analyses.append(
                {
                    "non_inferiority_type": xml_text(analysis, "non_inferiority_type"),
                    "method": xml_text(analysis, "method"),
                    "param_type": xml_text(analysis, "param_type"),
                    "groups_desc": xml_text(analysis, "groups_desc"),
                    "method_desc": xml_text(analysis, "method_desc"),
                    "estimate_desc": xml_text(analysis, "estimate_desc"),
                    "p_value_desc": xml_text(analysis, "p_value_desc"),
                }
            )
        outcomes.append(
            {
                "type": xml_text(outcome, "type"),
                "title": xml_text(outcome, "title"),
                "description": xml_text(outcome, "description"),
                "time_frame": xml_text(outcome, "time_frame"),
                "population": xml_text(outcome, "population"),
                "groups": groups,
                "analysis_list": analyses,
            }
        )
    return outcomes


def infer_blinding(masking: str) -> Tuple[str, str]:
    if not masking:
        return "", ""
    lower = masking.lower()
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown")):
        return "", ""
    if "none" in lower or "open" in lower:
        return "No", "0"
    if "single" in lower:
        return "Yes", "1"
    if "double" in lower or "triple" in lower or "quadruple" in lower:
        if "triple" in lower:
            return "Yes", "3"
        if "quad" in lower:
            return "Yes", "4"
        return "Yes", "2"
    return "Yes", ""


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def normalize_study_status(status: str) -> str:
    lower = (status or "").strip().lower()
    if not lower:
        return ""
    if "completed" in lower:
        return "Completed"
    if "terminated" in lower:
        return "Terminated"
    if any(term in lower for term in ("recruiting", "active", "enrolling", "not yet recruiting", "suspended")):
        return "Ongoing"
    return "Unknown"


def parse_outcome_groups(outcome: ET.Element) -> List[Dict[str, str]]:
    groups = []
    for group in outcome.findall("group_list/group"):
        groups.append(
            {
                "group_id": (group.attrib.get("group_id") or "").strip(),
                "title": xml_text(group, "title"),
                "description": xml_text(group, "description"),
            }
        )
    return groups


def parse_outcome_measures(outcome: ET.Element) -> List[Dict[str, object]]:
    measures = []
    measure_nodes = outcome.findall("measure_list/measure")
    if not measure_nodes:
        single_measure = outcome.find("measure")
        if single_measure is not None:
            measure_nodes = [single_measure]

    for measure in measure_nodes:
        entry: Dict[str, object] = {}
        title = xml_text(measure, "title")
        units = xml_text(measure, "units")
        param = xml_text(measure, "param")
        dispersion = xml_text(measure, "dispersion")
        if title:
            entry["measure_title"] = title
        if units:
            entry["units"] = units
        if param:
            entry["param"] = param
        if dispersion:
            entry["dispersion"] = dispersion

        values: List[Dict[str, str]] = []
        for measurement in measure.findall(".//measurement_list/measurement"):
            group_id = (measurement.attrib.get("group_id") or "").strip()
            value = (measurement.attrib.get("value") or "").strip()
            if not value:
                value = normalize_whitespace(measurement.text or "")
            spread = (measurement.attrib.get("spread") or "").strip()
            lower = (measurement.attrib.get("lower_limit") or "").strip()
            upper = (measurement.attrib.get("upper_limit") or "").strip()

            item: Dict[str, str] = {}
            if group_id:
                item["group_id"] = group_id
            if value:
                item["value"] = value
            if spread:
                item["spread"] = spread
            if lower:
                item["lower"] = lower
            if upper:
                item["upper"] = upper
            if item:
                values.append(item)

        if values:
            entry["values"] = values
        if entry:
            measures.append(entry)
    return measures


def filter_measures_for_group(
    measures: List[Dict[str, object]],
    group_id: str,
) -> List[Dict[str, object]]:
    gid = (group_id or "").strip()
    if not gid or gid.lower() == "overall":
        return measures
    filtered: List[Dict[str, object]] = []
    for measure in measures:
        values = [
            value
            for value in (measure.get("values") or [])
            if isinstance(value, dict) and value.get("group_id") == gid
        ]
        if not values:
            continue
        updated = dict(measure)
        updated["values"] = values
        filtered.append(updated)
    return filtered


def units_from_measures(measures: List[Dict[str, object]]) -> str:
    units: List[str] = []
    for measure in measures:
        unit = str(measure.get("units") or "").strip()
        if unit:
            units.append(unit)
    return join_values(units)


def normalize_label(text: str) -> str:
    return normalize_whitespace(text).lower()


def fuzzy_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return int(round(difflib.SequenceMatcher(None, a, b).ratio() * 100))


def arm_code_from_index(index: int) -> str:
    return f"A{index}"


def arm_id_from_group(nct_id: str, group_id: str, group_title: str) -> str:
    group_key = (group_id or "").strip()
    if group_key:
        return f"{nct_id}_{group_key}"
    title_key = normalize_label(group_title)
    if title_key:
        return f"{nct_id}_{title_key.replace(' ', '_')}"
    return f"{nct_id}_overall"


def is_informative_arm_group(arm: Dict[str, str]) -> bool:
    label = normalize_whitespace(str(arm.get("label") or ""))
    desc = normalize_whitespace(str(arm.get("description") or ""))
    if desc:
        return True
    if label and not label.isdigit():
        return True
    return False


def has_informative_arm_groups(arms: List[Dict[str, str]]) -> bool:
    return any(is_informative_arm_group(arm) for arm in arms)


def build_arm_candidates(arms: List[Dict[str, str]], nct_id: str) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for idx, arm in enumerate(arms, start=1):
        title_raw = normalize_whitespace(str(arm.get("label") or ""))
        desc_raw = normalize_whitespace(str(arm.get("description") or ""))
        title_norm = normalize_match_text(title_raw)
        desc_norm = normalize_match_text(desc_raw)
        aliases_norm = [title_norm] if title_norm else []
        arm_code = arm_code_from_index(idx)
        candidates.append(
            {
                "arm_id": f"{nct_id}_{arm_code}",
                "arm_code": arm_code,
                "title_raw": title_raw,
                "desc_raw": desc_raw,
                "title_norm": title_norm,
                "desc_norm": desc_norm,
                "aliases_norm": aliases_norm,
            }
        )
    return candidates


def build_group_candidates(groups: List[Dict[str, str]], nct_id: str) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for idx, group in enumerate(groups, start=1):
        group_id = (group.get("group_id") or "").strip()
        title_raw = normalize_whitespace(str(group.get("title") or ""))
        desc_raw = normalize_whitespace(str(group.get("description") or ""))
        title_norm = normalize_match_text(title_raw)
        desc_norm = normalize_match_text(desc_raw)
        aliases_norm: List[str] = []
        if title_norm:
            aliases_norm.append(title_norm)
        if group_id:
            group_id_norm = normalize_match_text(group_id)
            if group_id_norm and group_id_norm not in aliases_norm:
                aliases_norm.append(group_id_norm)
        arm_code = group_id or arm_code_from_index(idx)
        candidates.append(
            {
                "arm_id": arm_id_from_group(nct_id, group_id, title_raw),
                "arm_code": arm_code,
                "title_raw": title_raw,
                "desc_raw": desc_raw,
                "title_norm": title_norm,
                "desc_norm": desc_norm,
                "aliases_norm": aliases_norm,
            }
        )
    return candidates


def merge_arm_candidates(
    primary: List[Dict[str, object]],
    secondary: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen: set = set()
    for candidate in primary + secondary:
        key = str(candidate.get("title_norm") or "").strip()
        if not key:
            key = str(candidate.get("arm_id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged


def build_arm_candidates_with_results_fallback(
    root: ET.Element,
    nct_id: str,
) -> List[Dict[str, object]]:
    arms = parse_arm_groups(root)
    candidates = build_arm_candidates(arms, nct_id)
    result_groups = parse_results_groups_union(root)
    if result_groups:
        result_candidates = build_group_candidates(result_groups, nct_id)
        if candidates:
            candidates = merge_arm_candidates(candidates, result_candidates)
        else:
            candidates = result_candidates
    return candidates


def score_arm_candidates(
    endpoint_title_norm: str,
    endpoint_desc_norm: str,
    arm_candidates: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    for candidate in arm_candidates:
        aliases = candidate.get("aliases_norm") or []
        title_score = 0
        if endpoint_title_norm and aliases:
            title_score = max(fuzzy_score(endpoint_title_norm, alias) for alias in aliases if alias)
        desc_score = None
        if endpoint_desc_norm and candidate.get("desc_norm"):
            desc_score = fuzzy_score(endpoint_desc_norm, str(candidate.get("desc_norm") or ""))
        min_score = title_score if desc_score is None else min(title_score, desc_score)
        scored.append(
            {
                "candidate": candidate,
                "title_score": title_score,
                "desc_score": desc_score,
                "min_score": min_score,
            }
        )
    scored.sort(
        key=lambda item: (
            item["min_score"],
            item["title_score"],
            item["desc_score"] or 0,
        ),
        reverse=True,
    )
    return scored


def top_candidates_payload(scored: List[Dict[str, object]], limit: int = 3) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for item in scored[:limit]:
        candidate = item["candidate"]
        payload.append(
            {
                "arm_id": candidate.get("arm_id") or "",
                "arm_group_title_raw": candidate.get("title_raw") or "",
                "arm_group_desc_raw": candidate.get("desc_raw") or "",
                "title_score": item.get("title_score"),
                "desc_score": item.get("desc_score"),
            }
        )
    return payload


def match_endpoint_group(
    group: Dict[str, str],
    arm_candidates: List[Dict[str, object]],
) -> Dict[str, object]:
    endpoint_group_id = (group.get("group_id") or "").strip()
    endpoint_group_title_raw = normalize_whitespace(str(group.get("title") or ""))
    endpoint_group_desc_raw = normalize_whitespace(str(group.get("description") or ""))
    endpoint_group_title_norm = normalize_match_text(endpoint_group_title_raw)
    endpoint_group_desc_norm = normalize_match_text(endpoint_group_desc_raw)

    result: Dict[str, object] = {
        "endpoint_group_id": endpoint_group_id,
        "endpoint_group_title_raw": endpoint_group_title_raw,
        "endpoint_group_desc_raw": endpoint_group_desc_raw,
        "endpoint_group_title_norm": endpoint_group_title_norm,
        "endpoint_group_desc_norm": endpoint_group_desc_norm,
        "matched_arm_id": "",
        "arm_group_title_raw": "",
        "arm_group_desc_raw": "",
        "arm_group_title_norm": "",
        "arm_group_desc_norm": "",
        "match_type": "",
        "title_score": "",
        "desc_score": "",
        "debug_notes": "",
        "top_candidates": [],
    }

    debug_notes: List[str] = []
    if not arm_candidates:
        debug_notes.append("no_arm_groups")
        result["match_type"] = "unmatched_low_score"
        result["debug_notes"] = "; ".join(debug_notes)
        return result

    scored = score_arm_candidates(endpoint_group_title_norm, endpoint_group_desc_norm, arm_candidates)
    scored_for_candidates = scored
    if endpoint_group_desc_norm:
        scored_for_candidates = sorted(
            scored,
            key=lambda item: (
                item.get("desc_score") if item.get("desc_score") is not None else -1,
                item.get("title_score") or 0,
            ),
            reverse=True,
        )
    result["top_candidates"] = top_candidates_payload(scored_for_candidates)

    precise_matches = []
    if endpoint_group_title_norm:
        for candidate in arm_candidates:
            aliases = candidate.get("aliases_norm") or []
            if endpoint_group_title_norm in aliases:
                precise_matches.append(candidate)

    if precise_matches:
        if len(precise_matches) == 1:
            matched = precise_matches[0]
            result["match_type"] = "precise_title"
            result["matched_arm_id"] = matched.get("arm_id") or ""
            result["arm_group_title_raw"] = matched.get("title_raw") or ""
            result["arm_group_desc_raw"] = matched.get("desc_raw") or ""
            result["arm_group_title_norm"] = matched.get("title_norm") or ""
            result["arm_group_desc_norm"] = matched.get("desc_norm") or ""
            result["title_score"] = "100"
            if endpoint_group_desc_norm and matched.get("desc_norm"):
                result["desc_score"] = str(fuzzy_score(endpoint_group_desc_norm, str(matched.get("desc_norm"))))
            result["debug_notes"] = "; ".join(debug_notes)
            return result
        debug_notes.append("multiple_precise_title_matches")
        best = precise_matches[0]
        best_desc_score = None
        if endpoint_group_desc_norm:
            for candidate in precise_matches:
                candidate_desc = candidate.get("desc_norm")
                if not candidate_desc:
                    continue
                score = fuzzy_score(endpoint_group_desc_norm, str(candidate_desc))
                if best_desc_score is None or score > best_desc_score:
                    best_desc_score = score
                    best = candidate
        result["match_type"] = "precise_title_tiebreak"
        result["matched_arm_id"] = best.get("arm_id") or ""
        result["arm_group_title_raw"] = best.get("title_raw") or ""
        result["arm_group_desc_raw"] = best.get("desc_raw") or ""
        result["arm_group_title_norm"] = best.get("title_norm") or ""
        result["arm_group_desc_norm"] = best.get("desc_norm") or ""
        result["title_score"] = "100"
        if best_desc_score is not None:
            result["desc_score"] = str(best_desc_score)
        result["debug_notes"] = "; ".join(debug_notes)
        return result

    if not endpoint_group_desc_norm:
        debug_notes.append("endpoint_desc_missing")
        result["match_type"] = "unmatched_missing_desc"
        result["debug_notes"] = "; ".join(debug_notes)
        if scored:
            result["title_score"] = str(scored[0].get("title_score") or 0)
        return result

    scored_with_desc = [item for item in scored if item.get("desc_score") is not None]
    if not scored_with_desc:
        debug_notes.append("arm_desc_missing")
        result["match_type"] = "unmatched_missing_desc"
        result["debug_notes"] = "; ".join(debug_notes)
        if scored:
            result["title_score"] = str(scored[0].get("title_score") or 0)
        return result

    passing = [
        item
        for item in scored_with_desc
        if (item.get("desc_score") or 0) >= DESC_MATCH_THRESHOLD
    ]
    if not passing:
        result["match_type"] = "unmatched_low_score"
        result["debug_notes"] = "; ".join(debug_notes)
        if scored_for_candidates:
            result["title_score"] = str(scored_for_candidates[0].get("title_score") or 0)
            result["desc_score"] = str(scored_for_candidates[0].get("desc_score") or 0)
        return result

    passing.sort(
        key=lambda item: (
            item.get("desc_score") or 0,
            item.get("title_score") or 0,
        ),
        reverse=True,
    )
    fuzzy_tie = False
    if len(passing) > 1:
        top_gap = (passing[0].get("desc_score") or 0) - (passing[1].get("desc_score") or 0)
        if top_gap < FUZZY_MATCH_MARGIN:
            debug_notes.append("fuzzy_margin_too_close")
            fuzzy_tie = True

    top = passing[0]
    matched = top["candidate"]
    result["match_type"] = "fuzzy_desc_tiebreak" if fuzzy_tie else "fuzzy_desc"
    result["matched_arm_id"] = matched.get("arm_id") or ""
    result["arm_group_title_raw"] = matched.get("title_raw") or ""
    result["arm_group_desc_raw"] = matched.get("desc_raw") or ""
    result["arm_group_title_norm"] = matched.get("title_norm") or ""
    result["arm_group_desc_norm"] = matched.get("desc_norm") or ""
    result["title_score"] = str(top.get("title_score") or 0)
    result["desc_score"] = str(top.get("desc_score") or 0)
    result["debug_notes"] = "; ".join(debug_notes)
    return result


def ensure_matched_arm_id(
    match_info: Dict[str, object],
    nct_id: str,
    group: Dict[str, str],
) -> None:
    if match_info.get("matched_arm_id"):
        return
    fallback_id = arm_id_from_group(
        nct_id,
        str(group.get("group_id") or ""),
        str(group.get("title") or ""),
    )
    if not fallback_id:
        return
    match_info["matched_arm_id"] = fallback_id
    debug_notes = str(match_info.get("debug_notes") or "")
    note = "fallback_group_id"
    match_info["debug_notes"] = f"{debug_notes}; {note}" if debug_notes else note
    match_type = str(match_info.get("match_type") or "")
    if not match_type or match_type.startswith("unmatched"):
        match_info["match_type"] = "fallback_group_id"


def endpoint_type_from_outcome(outcome_type: str) -> str:
    lower = outcome_type.lower()
    if "primary" in lower:
        return "Primary"
    if "secondary" in lower:
        return "Secondary"
    if "other" in lower:
        return "Other"
    return ""


def resolve_group_drug_name(
    title: str,
    arm_map: Dict[str, Dict[str, object]],
) -> str:
    key = normalize_label(title)
    if key in arm_map:
        names = arm_map[key].get("interventions", [])
        if names:
            return join_values(names)
    return ""


def resolve_group_dosage(title: str, description: str, arm_map: Dict[str, Dict[str, object]]) -> str:
    key = normalize_label(title)
    if key in arm_map:
        arm_desc = str(arm_map[key].get("description") or "").strip()
        if arm_desc:
            return arm_desc
    return normalize_whitespace(description)


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


def xml_path_for_id(xml_root: Path, nct_id: str) -> Path:
    prefix = nct_id[:7] + "xxxx"
    return xml_root / prefix / f"{nct_id}.xml"


def iter_xml_files(xml_root: Path, nct_ids: List[str]) -> Iterable[Path]:
    if nct_ids:
        for nct_id in nct_ids:
            path = xml_path_for_id(xml_root, nct_id)
            if path.exists():
                yield path
            else:
                print(f"[WARN] XML not found for {nct_id}: {path}")
        return
    for path in xml_root.rglob("NCT*.xml"):
        yield path


def suffix_for_ncts(nct_ids: List[str]) -> str:
    if not nct_ids:
        return ""
    if len(nct_ids) == 1:
        return nct_ids[0]
    if len(nct_ids) <= 3:
        return "_".join(nct_ids)
    return f"{nct_ids[0]}_plus{len(nct_ids) - 1}"


def ensure_nct_id(root: ET.Element, nct_id: str = "") -> str:
    xml_id = xml_text(root, "id_info/nct_id") or xml_text(root, "nct_id")
    if nct_id:
        if xml_id and xml_id != nct_id:
            print(f"[WARN] NCT mismatch: input {nct_id} != xml {xml_id}")
        return nct_id
    return xml_id


def extract_design_rows(
    root: ET.Element,
    fields: List[str],
    region_map: Dict[str, set],
    nct_id: str = "",
) -> List[Dict[str, str]]:
    row = {field: "" for field in fields}
    nct_id = ensure_nct_id(root, nct_id)

    if "StudyID" in row:
        row["StudyID"] = nct_id
    if "NCT_No" in row:
        row["NCT_No"] = nct_id

    if "Prot_No" in row:
        row["Prot_No"] = xml_text(root, "id_info/org_study_id")

    secondary_ids = parse_secondary_ids(root)
    euct_ids = []
    other_ids = []
    for sid in secondary_ids:
        eudract_id = extract_eudract_id(sid)
        if eudract_id:
            euct_ids.append(eudract_id)
            continue
        other_ids.append(sid)
    if "EUCT_No" in row:
        row["EUCT_No"] = join_values(euct_ids)
    if "Other_No" in row:
        row["Other_No"] = join_values(other_ids)

    interventions = parse_interventions(root)
    arms = parse_arm_groups(root)
    drug_like = [iv for iv in interventions if iv["name"] and is_drug_like(iv["type"])]
    drug_like_names = [iv["name"] for iv in drug_like]
    if "Drug_Name" in row:
        row["Drug_Name"] = join_values(drug_like_names)
    if "Inv_Prod" in row:
        experimental_labels = {
            normalize_label(arm["label"])
            for arm in arms
            if arm["label"] and arm_type_matches(arm["type"], EXPERIMENTAL_ARM_KEYWORDS)
        }
        control_labels = {
            normalize_label(arm["label"])
            for arm in arms
            if arm["label"] and arm_type_matches(arm["type"], CONTROL_ARM_KEYWORDS)
        }
        interventions_by_label: Dict[str, List[str]] = {}
        for iv in drug_like:
            for label in iv["labels"]:
                key = normalize_label(label)
                if not key:
                    continue
                interventions_by_label.setdefault(key, []).append(iv["name"])
        exp_interventions: set = set()
        for label in experimental_labels:
            exp_interventions.update(interventions_by_label.get(label, []))
        ctrl_interventions: set = set()
        for label in control_labels:
            ctrl_interventions.update(interventions_by_label.get(label, []))
        inv_prod_names = [
            name
            for name in drug_like_names
            if name in exp_interventions
            and name not in ctrl_interventions
            and not is_placebo_like_name(name)
            and not is_background_like_name(name)
        ]
        non_placebo_drug_like = [
            name
            for name in drug_like_names
            if not is_placebo_like_name(name) and not is_background_like_name(name)
        ]
        if not inv_prod_names and len(non_placebo_drug_like) == 1:
            inv_prod_names = non_placebo_drug_like
        row["Inv_Prod"] = join_values(inv_prod_names)
    if "Arm_Description" in row:
        descriptions: List[str] = []
        for arm in arms:
            desc = normalize_whitespace(str(arm.get("description") or ""))
            if not desc:
                continue
            label = normalize_whitespace(str(arm.get("label") or ""))
            descriptions.append(f"{label}: {desc}" if label else desc)
        row["Arm_Description"] = join_values(descriptions)

    if "Study_Phase" in row:
        row["Study_Phase"] = xml_text(root, "phase")
    if "Start_Date" in row:
        row["Start_Date"] = xml_text(root, "start_date")

    completion_date = xml_text(root, "completion_date")
    primary_completion_date = xml_text(root, "primary_completion_date")
    if "Complet_Date" in row:
        row["Complet_Date"] = completion_date

    overall_status = xml_text(root, "overall_status")
    if "Termin_Date" in row:
        if "terminated" in overall_status.lower():
            row["Termin_Date"] = completion_date or primary_completion_date
        else:
            row["Termin_Date"] = ""
    if "Study_Status" in row:
        row["Study_Status"] = normalize_study_status(overall_status)

    if "Study_First_Posted_Date" in row:
        row["Study_First_Posted_Date"] = xml_text(root, "study_first_posted")
    if "Date_End" in row:
        row["Date_End"] = completion_date or primary_completion_date
    if "Results_Posted_Or_Updated_Date" in row:
        row["Results_Posted_Or_Updated_Date"] = xml_text(root, "results_first_posted") or xml_text(
            root, "last_update_posted"
        )

    officials = parse_officials(root)
    pi_officials = [
        official["name"]
        for official in officials
        if "principal investigator" in (official.get("role") or "").lower()
    ]
    rp_name = xml_text(root, "responsible_party/investigator_full_name")
    if "Name_PI" in row:
        if pi_officials:
            row["Name_PI"] = join_values(pi_officials)
        elif rp_name:
            row["Name_PI"] = rp_name

    sponsors = parse_sponsors(root)
    if "Sponsor" in row:
        row["Sponsor"] = join_values(sponsors)

    enrollment = root.find("enrollment")
    enrollment_text = normalize_whitespace(enrollment.text or "") if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    enrollment_value = safe_int(enrollment_text)
    actual_enrollment = enrollment_value if enrollment_type == "actual" else None
    if "No_Sub_Enroll" in row:
        row["No_Sub_Enroll"] = str(actual_enrollment) if actual_enrollment is not None else ""

    if "DMC" in row:
        dmc = xml_text(root, "oversight_info/has_dmc")
        if dmc.lower() in {"yes", "no"}:
            row["DMC"] = dmc.title()
        else:
            row["DMC"] = dmc

    centers = parse_location_centers(root)
    location_countries = [country for _, _, country in centers if country]
    country_list = xml_texts(root, "location_countries/country") or location_countries
    unique_countries = []
    seen = set()
    for country in country_list:
        key = normalize_country(country)
        if key and key not in seen:
            seen.add(key)
            unique_countries.append(country)

    if "MRCT" in row and unique_countries:
        region_hits = set()
        for country in unique_countries:
            region = region_for_country(country, region_map)
            if region:
                region_hits.add(region)
        if region_hits:
            row["MRCT"] = yes_no(len(region_hits) >= 2)

    total_centers = len(centers)
    if "No_Center" in row:
        row["No_Center"] = str(total_centers) if total_centers > 0 else ""

    if total_centers > 0:
        counts = {"NA": 0, "AP": 0, "WEU": 0, "EEU": 0, "AF": 0}
        for _, _, country in centers:
            region = region_for_country(country, region_map)
            if region:
                counts[region] += 1
        if "No_Center_NA" in row:
            row["No_Center_NA"] = str(counts["NA"])
        if "No_Center_AP" in row:
            row["No_Center_AP"] = str(counts["AP"])
        if "No_Center_WEU" in row:
            row["No_Center_WEU"] = str(counts["WEU"])
        if "No_Center_EEU" in row:
            row["No_Center_EEU"] = str(counts["EEU"])
        if "No_Center_AF" in row:
            row["No_Center_AF"] = str(counts["AF"])

    return [row]


def extract_stat_reg_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str = "",
) -> List[Dict[str, str]]:
    row = {field: "" for field in fields}
    nct_id = ensure_nct_id(root, nct_id)

    if "StudyID" in row:
        row["StudyID"] = nct_id

    interventions = parse_interventions(root)
    drug_names = [iv["name"] for iv in interventions if iv["name"] and is_drug_like(iv["type"])]
    if "Drug_Name" in row:
        row["Drug_Name"] = join_values(drug_names)

    study_type = xml_text(root, "study_type")
    is_interventional = "interventional" in study_type.lower()

    allocation = xml_text(root, "study_design_info/allocation")
    intervention_model = xml_text(root, "study_design_info/intervention_model")
    masking = xml_text(root, "study_design_info/masking")

    if "Randomization" in row and is_interventional and allocation:
        lower = allocation.lower()
        if not any(token in lower for token in ("n/a", "na", "not applicable", "unknown")):
            if re.search(r"\bnon[- ]?random", lower):
                row["Randomization"] = "No"
            elif "random" in lower:
                row["Randomization"] = "Yes"
    model_key = normalize_intervention_model(intervention_model)
    if "Random_Parallel" in row and is_interventional and model_key:
        row["Random_Parallel"] = "Yes" if "parallel" in model_key else "No"
    if "Random_Crossover" in row and is_interventional and model_key:
        row["Random_Crossover"] = "Yes" if "crossover" in model_key else "No"
    if "Random_Fact" in row and is_interventional and model_key:
        row["Random_Fact"] = "Yes" if "factorial" in model_key else "No"

    if "Stratification" in row and is_interventional:
        strat = xml_text(root, "study_design_info/stratification")
        if strat:
            lower = strat.lower()
            if not any(
                token in lower
                for token in ("n/a", "na", "not applicable", "unknown", "not provided")
            ):
                if lower in {"no", "none"} or "not stratified" in lower or "no stratification" in lower:
                    row["Stratification"] = "No"
                else:
                    row["Stratification"] = "Yes"
    if "No_Stratification" in row and row.get("Stratification") == "No":
        row["No_Stratification"] = "0"

    blinding, level = infer_blinding(masking)
    if "Blinding" in row and is_interventional:
        row["Blinding"] = blinding
    if "Level_Blinding" in row and is_interventional:
        row["Level_Blinding"] = level

    arms = parse_arm_groups(root)
    arm_types = [a["type"] for a in arms if a["type"]]
    arm_type_lowers = [t.lower() for t in arm_types]
    has_decisive_arm_types = any("other" not in t for t in arm_type_lowers)
    if has_decisive_arm_types and is_interventional:
        if "Placebo_control" in row:
            row["Placebo_control"] = yes_no(any("placebo" in t for t in arm_type_lowers))
        if "Active_Control" in row:
            row["Active_Control"] = yes_no(any("active" in t for t in arm_type_lowers))
        if "Hist_control" in row:
            row["Hist_control"] = yes_no(any("histor" in t for t in arm_type_lowers))

    arm_labels_by_type = {
        normalize_label(a["label"]): a["type"] for a in arms if a["label"] and a["type"]
    }
    control_labels = {
        label
        for label, arm_type in arm_labels_by_type.items()
        if arm_type_matches(arm_type, CONTROL_ARM_KEYWORDS)
    }
    control_drugs = []
    for iv in interventions:
        if not iv["labels"]:
            continue
        normalized_labels = {normalize_label(label) for label in iv["labels"]}
        if normalized_labels & control_labels:
            if not is_drug_like(str(iv["type"] or "")):
                continue
            if iv["name"]:
                if not is_placebo_like_name(iv["name"]):
                    control_drugs.append(iv["name"])
    if "Control_Drug" in row:
        row["Control_Drug"] = join_values(control_drugs)

    num_arms = safe_int(xml_text(root, "number_of_arms"))
    if num_arms is None and arms:
        num_arms = len(arms)
    if (num_arms is None or num_arms <= 1) and not has_informative_arm_groups(arms):
        results_count = count_results_groups(root)
        if results_count:
            num_arms = results_count
    if "No_Arm" in row:
        row["No_Arm"] = str(num_arms) if num_arms is not None else ""
    if "Single_Arm" in row and is_interventional and num_arms is not None:
        row["Single_Arm"] = yes_no(num_arms == 1)

    primary_outcomes = parse_outcomes(root, "primary_outcome")
    secondary_outcomes = parse_outcomes(root, "secondary_outcome")
    if "Primary_EP" in row and primary_outcomes:
        measures = [outcome["measure"] for outcome in primary_outcomes if outcome.get("measure")]
        row["Primary_EP"] = join_values(measures)
    if "No_Prim_EP" in row and primary_outcomes:
        row["No_Prim_EP"] = str(len(primary_outcomes))
    if "Key_Second_EP" in row and secondary_outcomes:
        row["Key_Second_EP"] = secondary_outcomes[0]["measure"]

    enrollment = root.find("enrollment")
    enrollment_text = normalize_whitespace(enrollment.text or "") if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    enrollment_value = safe_int(enrollment_text)
    planned_enrollment = enrollment_value if enrollment_type == "anticipated" else None
    if "Sample_Size" in row:
        row["Sample_Size"] = str(planned_enrollment) if planned_enrollment is not None else ""

    return [row]


def extract_targetpop_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str = "",
) -> List[Dict[str, str]]:
    row = {field: "" for field in fields}
    nct_id = ensure_nct_id(root, nct_id)

    if "StudyID" in row:
        row["StudyID"] = nct_id

    interventions = parse_interventions(root)
    drug_names = [iv["name"] for iv in interventions if iv["name"] and is_drug_like(iv["type"])]
    if "Drug_Name" in row:
        row["Drug_Name"] = join_values(drug_names)

    if "Disease" in row:
        row["Disease"] = join_values(xml_texts(root, "condition"))

    if "Gender_Criteria" in row:
        row["Gender_Criteria"] = normalize_gender(xml_text(root, "eligibility/gender"))

    if "Age_Min" in row:
        row["Age_Min"] = parse_age_value(xml_text(root, "eligibility/minimum_age"))
    if "Age_Max" in row:
        row["Age_Max"] = parse_age_value(xml_text(root, "eligibility/maximum_age"))

    return [row]


def extract_drug_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str = "",
) -> List[Dict[str, str]]:
    nct_id = ensure_nct_id(root, nct_id)
    interventions = parse_interventions(root)
    drug_names = [iv["name"] for iv in interventions if iv["name"] and is_drug_like(iv["type"])]
    unique_names = list(dict.fromkeys(drug_names))

    rows: List[Dict[str, str]] = []
    for name in unique_names:
        row = {field: "" for field in fields}
        if "StudyID" in row:
            row["StudyID"] = nct_id
        if "Drug_Name" in row:
            row["Drug_Name"] = name
        rows.append(row)
    return rows


def extract_others_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str = "",
) -> List[Dict[str, str]]:
    row = {field: "" for field in fields}
    nct_id = ensure_nct_id(root, nct_id)

    if "StudyID" in row:
        row["StudyID"] = nct_id

    interventions = parse_interventions(root)
    drug_names = [iv["name"] for iv in interventions if iv["name"] and is_drug_like(iv["type"])]
    if "Drug_Name" in row:
        row["Drug_Name"] = join_values(drug_names)

    if "Sponsor_Type" in row:
        lead_entry, collaborators = parse_sponsor_entries(root)
        sponsor_type = classify_sponsor_type(lead_entry["agency"], lead_entry["agency_class"])
        if not sponsor_type:
            for entry in collaborators:
                sponsor_type = classify_sponsor_type(entry["agency"], entry["agency_class"])
                if sponsor_type:
                    break
        row["Sponsor_Type"] = sponsor_type

    return [row]


def extract_text_blocks(root: ET.Element, nct_id: str) -> Dict[str, object]:
    enrollment = root.find("enrollment")
    enrollment_text = normalize_whitespace(enrollment.text or "") if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    design_info = {
        "study_type": xml_text(root, "study_type"),
        "phase": xml_text(root, "phase"),
        "allocation": xml_text(root, "study_design_info/allocation"),
        "intervention_model": xml_text(root, "study_design_info/intervention_model"),
        "masking": xml_text(root, "study_design_info/masking"),
        "masking_description": xml_text(root, "study_design_info/masking_description"),
        "primary_purpose": xml_text(root, "study_design_info/primary_purpose"),
        "number_of_arms": xml_text(root, "number_of_arms"),
    }
    if enrollment_text:
        design_info["enrollment"] = enrollment_text
    if enrollment_type:
        design_info["enrollment_type"] = enrollment_type
    design_info = {key: value for key, value in design_info.items() if value}
    return {
        "nct_id": nct_id,
        "brief_title": xml_text(root, "brief_title"),
        "official_title": xml_text(root, "official_title"),
        "brief_summary": xml_text(root, "brief_summary/textblock"),
        "detailed_description": xml_text(root, "detailed_description/textblock"),
        "eligibility_criteria": xml_text(root, "eligibility/criteria/textblock"),
        "design_info": design_info,
        "arm_groups": parse_arm_groups(root),
        "interventions": parse_interventions(root),
        "primary_outcomes": parse_outcomes(root, "primary_outcome"),
        "secondary_outcomes": parse_outcomes(root, "secondary_outcome"),
        "participant_flow": parse_results_participant_flow(root),
        "baseline_results": parse_results_baseline(root),
        "results_outcomes": parse_results_outcomes(root),
        "keywords": xml_texts(root, "keyword"),
        "conditions": xml_texts(root, "condition"),
        "location_countries": xml_texts(root, "location_countries/country"),
    }


def build_results_context(root: ET.Element, nct_id: str) -> Optional[Dict[str, object]]:
    nct_id = ensure_nct_id(root, nct_id)
    results = root.find("clinical_results")
    if results is None:
        return None
    outcome_list = results.find("outcome_list")
    if outcome_list is None:
        return None

    interventions = [iv for iv in parse_interventions(root) if iv.get("name") and is_drug_like(iv.get("type", ""))]
    arms = parse_arm_groups(root)
    arm_map: Dict[str, Dict[str, object]] = {}
    for arm in arms:
        key = normalize_label(arm["label"])
        if not key:
            continue
        arm_map[key] = {"label": arm["label"], "description": arm["description"], "interventions": []}
    for iv in interventions:
        for label in iv["labels"]:
            key = normalize_label(label)
            if not key or key not in arm_map:
                continue
            name = str(iv.get("name") or "").strip()
            if name:
                arm_map[key]["interventions"].append(name)
    for data in arm_map.values():
        data["interventions"] = list(dict.fromkeys(data.get("interventions", [])))

    return {
        "nct_id": nct_id,
        "outcomes": outcome_list.findall("outcome"),
        "arm_map": arm_map,
    }


def extract_endpoints_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str,
    match_report: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    context = build_results_context(root, nct_id)
    if context is None:
        return rows

    outcomes = context["outcomes"]
    arm_map = context["arm_map"]
    nct_id = context["nct_id"]
    arm_candidates = build_arm_candidates_with_results_fallback(root, nct_id)

    first_secondary_idx = None
    for idx, outcome in enumerate(outcomes):
        outcome_type = xml_text(outcome, "type").lower()
        if "secondary" in outcome_type:
            first_secondary_idx = idx
            break

    for idx, outcome in enumerate(outcomes):
        outcome_type = xml_text(outcome, "type")
        endpoint_type = ""
        lower = outcome_type.lower()
        if "primary" in lower:
            endpoint_type = "Primary"
        elif "secondary" in lower and first_secondary_idx is not None and idx == first_secondary_idx:
            endpoint_type = "Key Secondary"
        if not endpoint_type:
            continue
        endpoint_name = xml_text(outcome, "title")

        analyses = outcome.findall("analysis_list/analysis")
        p_list: List[Dict[str, object]] = []
        point_list: List[Dict[str, object]] = []
        ci_list: List[Dict[str, object]] = []
        for analysis in analyses:
            analysis_id = analysis.attrib.get("analysis_id") or xml_text(analysis, "analysis_id")
            groups_desc = xml_text(analysis, "groups_desc")
            method = xml_text(analysis, "method")
            p_val = xml_text(analysis, "p_value")
            modifier = xml_text(analysis, "p_value_modifier")
            param_type = xml_text(analysis, "param_type")
            param_value = xml_text(analysis, "param_value")
            ci_lower = xml_text(analysis, "ci_lower_limit")
            ci_upper = xml_text(analysis, "ci_upper_limit")
            ci_pct = xml_text(analysis, "ci_percent")

            if p_val or modifier:
                item: Dict[str, object] = {}
                if analysis_id:
                    item["analysis_id"] = analysis_id
                if groups_desc:
                    item["groups"] = groups_desc
                if method:
                    item["method"] = method
                item["p"] = {"value": p_val, "modifier": modifier}
                p_list.append(item)
            if param_type or param_value:
                item = {}
                if analysis_id:
                    item["analysis_id"] = analysis_id
                if groups_desc:
                    item["groups"] = groups_desc
                item["effect"] = {"type": param_type, "value": param_value}
                point_list.append(item)
            if ci_lower or ci_upper or ci_pct:
                item = {}
                if analysis_id:
                    item["analysis_id"] = analysis_id
                if groups_desc:
                    item["groups"] = groups_desc
                item["ci"] = {"pct": ci_pct, "lower": ci_lower, "upper": ci_upper}
                ci_list.append(item)
        p_json = json.dumps(p_list, ensure_ascii=False)
        point_json = json.dumps(point_list, ensure_ascii=False)
        ci_json = json.dumps(ci_list, ensure_ascii=False)
        needs_measure = "EP_Value" in fields or "EP_Unit" in fields
        measure_payload = parse_outcome_measures(outcome) if needs_measure else []

        groups = parse_outcome_groups(outcome)
        if not groups:
            groups = [{"group_id": "overall", "title": "Overall", "description": ""}]

        for group in groups:
            group_id = group.get("group_id", "")
            group_title = group.get("title", "")
            drug_name = resolve_group_drug_name(group_title, arm_map)
            group_measures = filter_measures_for_group(measure_payload, group_id) if measure_payload else []
            measure_json = json.dumps(group_measures, ensure_ascii=False)
            match_info = match_endpoint_group(group, arm_candidates)
            ensure_matched_arm_id(match_info, nct_id, group)
            top_candidates_json = json.dumps(match_info.get("top_candidates") or [], ensure_ascii=False)

            row = {field: "" for field in fields}
            if "StudyID" in row:
                row["StudyID"] = nct_id
            if "Drug_Name" in row:
                row["Drug_Name"] = drug_name
            if "Arm_ID" in row:
                row["Arm_ID"] = match_info.get("matched_arm_id", "")
            if "Endpoint_Name" in row:
                row["Endpoint_Name"] = endpoint_name
            if "Endpoint_Type" in row:
                row["Endpoint_Type"] = endpoint_type
            if "EP_P_value" in row:
                row["EP_P_value"] = p_json
            if "EP_Point" in row:
                row["EP_Point"] = point_json
            if "EP_95CI" in row:
                row["EP_95CI"] = ci_json
            if "EP_Value" in row:
                row["EP_Value"] = measure_json
            if "EP_Unit" in row:
                row["EP_Unit"] = units_from_measures(group_measures)
            if "endpoint_group_id" in row:
                row["endpoint_group_id"] = match_info.get("endpoint_group_id", "")
            if "endpoint_group_title_raw" in row:
                row["endpoint_group_title_raw"] = match_info.get("endpoint_group_title_raw", "")
            if "endpoint_group_desc_raw" in row:
                row["endpoint_group_desc_raw"] = match_info.get("endpoint_group_desc_raw", "")
            if "endpoint_group_title_norm" in row:
                row["endpoint_group_title_norm"] = match_info.get("endpoint_group_title_norm", "")
            if "endpoint_group_desc_norm" in row:
                row["endpoint_group_desc_norm"] = match_info.get("endpoint_group_desc_norm", "")
            if "matched_arm_id" in row:
                row["matched_arm_id"] = match_info.get("matched_arm_id", "")
            if "arm_group_title_raw" in row:
                row["arm_group_title_raw"] = match_info.get("arm_group_title_raw", "")
            if "arm_group_desc_raw" in row:
                row["arm_group_desc_raw"] = match_info.get("arm_group_desc_raw", "")
            if "arm_group_title_norm" in row:
                row["arm_group_title_norm"] = match_info.get("arm_group_title_norm", "")
            if "arm_group_desc_norm" in row:
                row["arm_group_desc_norm"] = match_info.get("arm_group_desc_norm", "")
            if "match_type" in row:
                row["match_type"] = match_info.get("match_type", "")
            if "title_score" in row:
                row["title_score"] = match_info.get("title_score", "")
            if "desc_score" in row:
                row["desc_score"] = match_info.get("desc_score", "")
            if "debug_notes" in row:
                row["debug_notes"] = match_info.get("debug_notes", "")
            if "top_candidates_json" in row:
                row["top_candidates_json"] = top_candidates_json
            rows.append(row)

            if match_report is not None:
                match_report.append(
                    {
                        "nct_id": nct_id,
                        "endpoint_name": endpoint_name,
                        "endpoint_type": endpoint_type,
                        "endpoint_group_id": match_info.get("endpoint_group_id", ""),
                        "endpoint_group_title_raw": match_info.get("endpoint_group_title_raw", ""),
                        "endpoint_group_desc_raw": match_info.get("endpoint_group_desc_raw", ""),
                        "endpoint_group_title_norm": match_info.get("endpoint_group_title_norm", ""),
                        "endpoint_group_desc_norm": match_info.get("endpoint_group_desc_norm", ""),
                        "matched_arm_id": match_info.get("matched_arm_id", ""),
                        "arm_group_title_raw": match_info.get("arm_group_title_raw", ""),
                        "arm_group_desc_raw": match_info.get("arm_group_desc_raw", ""),
                        "arm_group_title_norm": match_info.get("arm_group_title_norm", ""),
                        "arm_group_desc_norm": match_info.get("arm_group_desc_norm", ""),
                        "match_type": match_info.get("match_type", ""),
                        "title_score": match_info.get("title_score", ""),
                        "desc_score": match_info.get("desc_score", ""),
                        "debug_notes": match_info.get("debug_notes", ""),
                        "top_candidates": match_info.get("top_candidates", []),
                    }
                )

    return rows


def extract_groups_rows(
    root: ET.Element,
    fields: List[str],
    nct_id: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    context = build_results_context(root, nct_id)
    if context is None:
        return rows

    outcomes = context["outcomes"]
    arm_map = context["arm_map"]
    nct_id = context["nct_id"]
    arm_candidates = build_arm_candidates_with_results_fallback(root, nct_id)

    completion_date = xml_text(root, "completion_date")
    primary_completion_date = xml_text(root, "primary_completion_date")
    overall_status = xml_text(root, "overall_status")
    results_posted = xml_text(root, "results_first_posted") or xml_text(
        root, "last_update_posted"
    )

    enrollment = root.find("enrollment")
    enrollment_text = normalize_whitespace(enrollment.text or "") if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    enrollment_value = safe_int(enrollment_text)
    actual_enrollment = enrollment_value if enrollment_type == "actual" else None

    groups = parse_results_groups_preferred(root)
    if groups:
        groups_seen = set()
        deduped: List[Dict[str, str]] = []
        for group in groups:
            group_id = group.get("group_id", "")
            group_title = group.get("title", "")
            key = group_dedup_key(group_id, group_title)
            if key in groups_seen:
                continue
            groups_seen.add(key)
            deduped.append(group)
        groups = deduped
    else:
        groups = []
        groups_seen = set()
        for outcome in outcomes:
            outcome_groups = parse_outcome_groups(outcome)
            if not outcome_groups:
                outcome_groups = [{"group_id": "overall", "title": "Overall", "description": ""}]
            for group in outcome_groups:
                group_id = group.get("group_id", "")
                group_title = group.get("title", "")
                key = group_dedup_key(group_id, group_title)
                if key in groups_seen:
                    continue
                groups_seen.add(key)
                groups.append(group)

    if not groups:
        groups = [{"group_id": "overall", "title": "Overall", "description": ""}]

    enroll_by_gid = parse_participant_flow_started_counts(root)
    demographics = extract_group_demographics(root)
    demo_by_id, demo_by_title = index_group_demographics(demographics)
    single_group = len(groups) == 1
    for group in groups:
        group_id = group.get("group_id", "")
        group_title = group.get("title", "")
        group_description = group.get("description", "")
        drug_name = resolve_group_drug_name(group_title, arm_map)
        demo_entry = match_group_demographics(group, demographics, demo_by_id, demo_by_title)

        row = {field: "" for field in fields}
        if "StudyID" in row:
            row["StudyID"] = nct_id
        if "Drug_Name" in row:
            row["Drug_Name"] = drug_name
        if "Arm_ID" in row:
            match_info = match_endpoint_group(group, arm_candidates)
            ensure_matched_arm_id(match_info, nct_id, group)
            row["Arm_ID"] = match_info.get("matched_arm_id", "")
        if "Date_End" in row:
            row["Date_End"] = completion_date or primary_completion_date
        if "Results_Posted_Or_Updated_Date" in row:
            row["Results_Posted_Or_Updated_Date"] = results_posted
        if "Date_Report" in row:
            row["Date_Report"] = results_posted
        if "Study_Status" in row:
            row["Study_Status"] = normalize_study_status(overall_status)
        if "No_Sub_Enroll" in row:
            if group_id and group_id in enroll_by_gid:
                row["No_Sub_Enroll"] = enroll_by_gid[group_id]
            elif single_group and actual_enrollment is not None:
                row["No_Sub_Enroll"] = str(actual_enrollment)
        if "group_id_raw" in row:
            row["group_id_raw"] = (
                str(demo_entry.get("group_id_raw") or "") if demo_entry else str(group_id or "")
            )
        if "group_title_raw" in row:
            row["group_title_raw"] = (
                str(demo_entry.get("group_title_raw") or "") if demo_entry else str(group_title or "")
            )
        if "group_desc_raw" in row:
            row["group_desc_raw"] = (
                str(demo_entry.get("group_desc_raw") or "")
                if demo_entry
                else str(group_description or "")
            )
        if demo_entry:
            for field in (
                "N_group",
                "count_male",
                "count_female",
                "count_white",
                "count_black",
                "count_asian",
                "count_other",
                "Med_Age",
                "Min_Age",
                "Max_Age",
                "Prop_Male",
                "Prop_White",
                "Prop_Black",
                "Prop_Asian",
                "Prop_NA",
                "Prop_AP",
                "Prop_EU",
                "Prop_AF",
                "Prop_ECOG0",
                "Prop_ECOG1",
                "Prop_ECOG2",
                "Prop_ECOG3",
                "Heter_Index",
            ):
                if field in row:
                    value = demo_entry.get(field)
                    if value is not None and value != "":
                        row[field] = str(value)
        if "Missing_K" in row:
            missing_keys = [
                "Med_Age",
                "Min_Age",
                "Max_Age",
                "Prop_Male",
                "Prop_White",
                "Prop_Black",
                "Prop_Asian",
                "Prop_NA",
                "Prop_AP",
                "Prop_EU",
                "Prop_AF",
            ]
            missing_count = 0
            for field in missing_keys:
                if field in row and not row.get(field):
                    missing_count += 1
            row["Missing_K"] = str(missing_count)
        rows.append(row)

    return rows


def parse_tables(raw: str) -> List[str]:
    if not raw.strip():
        return list(ALL_TABLES)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    normalized = {name.lower(): name for name in ALL_TABLES}
    tables: List[str] = []
    invalid: List[str] = []
    for item in requested:
        key = item.lower()
        if key in normalized:
            tables.append(normalized[key])
        else:
            invalid.append(item)
    if invalid:
        raise ValueError(f"Unknown tables requested: {', '.join(invalid)}")
    return tables


def output_path_for_table(output_root: Path, table: str, nct_id: str) -> Path:
    out_dir = output_root / "by_nctid" / nct_id
    filename = f"{table}.csv"
    return out_dir / filename


def write_matching_report(entries: List[Dict[str, object]], output_root: Path) -> None:
    if not entries:
        return
    summary: Dict[str, int] = {}
    for entry in entries:
        match_type = str(entry.get("match_type") or "")
        summary[match_type] = summary.get(match_type, 0) + 1

    ambiguous_types = {
        "ambiguous_precise",
        "ambiguous_fuzzy",
        "precise_title_tiebreak",
        "fuzzy_desc_tiebreak",
    }
    unmatched_types = {"unmatched_missing_desc", "unmatched_low_score"}
    ambiguous = [entry for entry in entries if entry.get("match_type") in ambiguous_types]
    unmatched = [entry for entry in entries if entry.get("match_type") in unmatched_types]

    report = {
        "summary": summary,
        "settings": {
            "title_threshold": TITLE_MATCH_THRESHOLD,
            "desc_threshold": DESC_MATCH_THRESHOLD,
            "margin": FUZZY_MATCH_MARGIN,
        },
        "ambiguous": ambiguous,
        "unmatched": unmatched,
    }
    report_path = output_root / "matching_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def write_group_arm_mapping(
    entries: List[Dict[str, object]],
    output_root: Path,
    nct_id: str = "",
) -> None:
    if not entries:
        return
    if nct_id:
        output_path = output_root / "by_nctid" / nct_id / "GroupArmMapping.csv"
    else:
        output_path = output_root / "GroupArmMapping.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_ARM_MAPPING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["top_candidates_json"] = json.dumps(entry.get("top_candidates") or [], ensure_ascii=False)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build CTG tables from CT.gov XML.")
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=project_root / "data/raw/CSR-Vars 2026-01-08.xlsx",
        help="Path to CSR-Vars 2026-01-08.xlsx.",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=project_root / "data/raw_data",
        help="Root directory containing CT.gov XML files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data/ctg_extract",
        help="Root output directory for CTG tables (by_nctid/<NCTID>/<table>.csv).",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default="",
        help="Comma-separated list of tables to output.",
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
        "--text-out",
        type=str,
        default="",
        help=(
            "Optional output path for text blocks (default: auto for --nct-id/--nct-csv; "
            "set a path to enable; use 'none' to disable)."
        ),
    )
    parser.add_argument(
        "--text-format",
        type=str,
        default="auto",
        choices=("auto", "jsonl", "json", "both"),
        help=(
            "Text block format: auto (single NCT -> json+jsonl, multiple -> jsonl), "
            "jsonl (newline-delimited), json (pretty JSON array), both."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    default_text = project_root / "data/ctg_extract/ctg_text_blocks.jsonl"

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Missing xlsx: {args.xlsx}")
    if not args.xml_root.exists():
        raise FileNotFoundError(f"Missing xml root: {args.xml_root}")

    tables = parse_tables(args.tables)
    table_fields = {table: load_sheet_fields(args.xlsx, table) for table in tables}
    if "Endpoints" in table_fields:
        table_fields["Endpoints"] = extend_fields(table_fields["Endpoints"], ENDPOINT_MATCH_FIELDS)
    if "Groups" in table_fields:
        table_fields["Groups"] = extend_fields(table_fields["Groups"], GROUPS_EXTRA_FIELDS)

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)
    suffix = suffix_for_ncts(nct_ids) if nct_ids else ""

    text_out_arg = (args.text_out or "").strip()
    text_out_lower = text_out_arg.lower()
    text_format = (args.text_format or "auto").lower()

    if nct_ids:
        if len(nct_ids) == 1:
            nct_id = nct_ids[0]
            per_nct_text = project_root / "data/ctg_extract/by_nctid" / nct_id / "ctg_text_blocks.jsonl"
            if text_out_lower in {"", "auto"}:
                args.text_out = str(per_nct_text)
            elif text_out_lower in {"none", "null"}:
                pass
            elif Path(text_out_arg) == default_text:
                args.text_out = str(per_nct_text)
        else:
            if text_out_lower in {"", "auto"}:
                args.text_out = str(default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl"))
            elif text_out_lower not in {"none", "null"} and Path(text_out_arg) == default_text:
                args.text_out = str(default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl"))

    text_out_arg = (args.text_out or "").strip()
    if text_out_arg.lower() in {"", "none", "null"}:
        text_out_path = None
    else:
        text_out_path = Path(text_out_arg)

    def normalize_text_paths(base: Path) -> Tuple[Path, Path]:
        if base.suffix == ".jsonl":
            return base, base.with_suffix(".json")
        if base.suffix == ".json":
            return base.with_suffix(".jsonl"), base
        return base.with_suffix(".jsonl"), base.with_suffix(".json")

    write_jsonl = False
    write_json = False
    if text_format == "auto":
        if nct_ids and len(nct_ids) == 1:
            write_jsonl = True
            write_json = True
        elif nct_ids:
            write_jsonl = True
    elif text_format == "jsonl":
        write_jsonl = True
    elif text_format == "json":
        write_json = True
    elif text_format == "both":
        write_jsonl = True
        write_json = True

    text_jsonl_file = None
    text_json_file = None
    text_json_first = True
    if text_out_path and (write_jsonl or write_json):
        jsonl_path, json_path = normalize_text_paths(text_out_path)
        if write_jsonl:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            text_jsonl_file = jsonl_path.open("w")
        if write_json:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            text_json_file = json_path.open("w")
            text_json_file.write("[\n")

    region_map = region_sets() if "Design" in tables else {}
    match_report: List[Dict[str, object]] = [] if "Endpoints" in tables else []

    for xml_path in iter_xml_files(args.xml_root, nct_ids):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            print(f"[WARN] Failed to parse {xml_path}: {exc}")
            continue

        nct_id = xml_text(root, "id_info/nct_id") or xml_text(root, "nct_id") or xml_path.stem
        output_paths = {table: output_path_for_table(args.output_root, table, nct_id) for table in tables}
        for path in output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        match_report_local: List[Dict[str, object]] = [] if "Endpoints" in tables else []
        with ExitStack() as stack:
            writers: Dict[str, csv.DictWriter] = {}
            for table in tables:
                handle = stack.enter_context(output_paths[table].open("w", newline=""))
                writer = csv.DictWriter(handle, fieldnames=table_fields[table], extrasaction="ignore")
                writer.writeheader()
                writers[table] = writer

            if "Design" in tables:
                for row in extract_design_rows(root, table_fields["Design"], region_map, nct_id):
                    writers["Design"].writerow(row)
            if "Stat_Reg" in tables:
                for row in extract_stat_reg_rows(root, table_fields["Stat_Reg"], nct_id):
                    writers["Stat_Reg"].writerow(row)
            if "TargetPop" in tables:
                for row in extract_targetpop_rows(root, table_fields["TargetPop"], nct_id):
                    writers["TargetPop"].writerow(row)
            if "Drug" in tables:
                for row in extract_drug_rows(root, table_fields["Drug"], nct_id):
                    writers["Drug"].writerow(row)
            if "Others" in tables:
                for row in extract_others_rows(root, table_fields["Others"], nct_id):
                    writers["Others"].writerow(row)
            if "Endpoints" in tables:
                for row in extract_endpoints_rows(root, table_fields["Endpoints"], nct_id, match_report_local):
                    writers["Endpoints"].writerow(row)
            if "Groups" in tables:
                for row in extract_groups_rows(root, table_fields["Groups"], nct_id):
                    writers["Groups"].writerow(row)

        if match_report_local:
            write_group_arm_mapping(match_report_local, args.output_root, nct_id)
            match_report.extend(match_report_local)

        if text_jsonl_file is not None or text_json_file is not None:
            text_blocks = extract_text_blocks(root, nct_id)
            if text_jsonl_file is not None:
                text_jsonl_file.write(json.dumps(text_blocks, ensure_ascii=False) + "\n")
            if text_json_file is not None:
                if not text_json_first:
                    text_json_file.write(",\n")
                entry = json.dumps(text_blocks, ensure_ascii=False, indent=2)
                indented = "\n".join(
                    f"  {line}" if line else line for line in entry.splitlines()
                )
                text_json_file.write(indented)
                text_json_first = False

    if text_jsonl_file is not None:
        text_jsonl_file.close()
    if text_json_file is not None:
        if not text_json_first:
            text_json_file.write("\n")
        text_json_file.write("]\n")
        text_json_file.close()

    if match_report:
        write_matching_report(match_report, args.output_root)


if __name__ == "__main__":
    main()
