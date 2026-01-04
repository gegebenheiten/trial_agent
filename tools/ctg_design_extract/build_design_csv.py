#!/usr/bin/env python3
"""
Create a Design-feature CSV from ClinicalTrials.gov XML files.
Optionally save text blocks for later LLM extraction as JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WHITESPACE_RE = re.compile(r"\s+")
EUDRACT_RE = re.compile(r"\b\d{4}-\d{6}-\d{2}\b")
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")
ACTUAL_ENROLL_FIELD = "No_Subj_Actual"


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


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


def safe_int(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


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


def load_design_fields(xlsx_path: Path) -> List[str]:
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

        studyid_col = header_map.get("StudyID")
        if not studyid_col:
            raise ValueError("StudyID column not found in Design sheet")

        fields: List[str] = []
        seen = set()
        for cells in parsed_rows[header_idx + 1 :]:
            raw = cells.get(studyid_col)
            if raw is None:
                continue
            name = str(raw).strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                fields.append(name)
        if not fields:
            raise ValueError("No StudyID fields found in Design sheet")
        return fields


def ensure_extra_fields(fields: List[str]) -> List[str]:
    if ACTUAL_ENROLL_FIELD in fields:
        return fields
    updated = list(fields)
    insert_at = updated.index("No_Subj_Planned") + 1 if "No_Subj_Planned" in updated else len(updated)
    updated.insert(insert_at, ACTUAL_ENROLL_FIELD)
    return updated


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


def xml_text(root: ET.Element, path: str) -> str:
    return normalize_whitespace(root.findtext(path) or "")


def xml_texts(root: ET.Element, path: str) -> List[str]:
    values = []
    for el in root.findall(path):
        value = normalize_whitespace(el.text or "")
        if value:
            values.append(value)
    return values


def parse_secondary_ids(root: ET.Element) -> List[str]:
    return xml_texts(root, "id_info/secondary_id")


def parse_interventions(root: ET.Element) -> List[Dict[str, List[str] | str]]:
    interventions = []
    for iv in root.findall("intervention"):
        name = xml_text(iv, "intervention_name")
        other_names = [normalize_whitespace(o.text or "") for o in iv.findall("other_name") if normalize_whitespace(o.text or "")]
        labels = [normalize_whitespace(l.text or "") for l in iv.findall("arm_group_label") if normalize_whitespace(l.text or "")]
        interventions.append(
            {
                "type": xml_text(iv, "intervention_type"),
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


def parse_locations(root: ET.Element) -> List[str]:
    countries = []
    for loc in root.findall("location"):
        country = xml_text(loc, "facility/address/country")
        if country:
            countries.append(country)
    return countries


def parse_officials(root: ET.Element) -> List[str]:
    names = []
    for official in root.findall("overall_official"):
        last_name = xml_text(official, "last_name")
        if last_name:
            names.append(last_name)
            continue
        parts = [
            xml_text(official, "first_name"),
            xml_text(official, "middle_name"),
            xml_text(official, "last_name"),
        ]
        combined = normalize_whitespace(" ".join(p for p in parts if p))
        if combined:
            names.append(combined)
    return names


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
    groups = []
    for group in flow.findall("group_list/group"):
        groups.append(
            {
                "title": xml_text(group, "title"),
                "description": xml_text(group, "description"),
            }
        )
    return {
        "recruitment_details": xml_text(flow, "recruitment_details"),
        "pre_assignment_details": xml_text(flow, "pre_assignment_details"),
        "groups": groups,
    }


def parse_results_baseline(root: ET.Element) -> Dict[str, object]:
    results = root.find("clinical_results")
    if results is None:
        return {}
    baseline = results.find("baseline")
    if baseline is None:
        return {}
    groups = []
    for group in baseline.findall("group_list/group"):
        groups.append(
            {
                "title": xml_text(group, "title"),
                "description": xml_text(group, "description"),
            }
        )
    return {
        "population": xml_text(baseline, "population"),
        "groups": groups,
    }


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
    if "none" in lower or "open" in lower:
        return "No", "0"
    if "single" in lower:
        return "Yes", "1"
    if "double" in lower or "triple" in lower or "quadruple" in lower:
        return "Yes", "2"
    return "Yes", ""


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def extract_features(root: ET.Element, fields: List[str], region_map: Dict[str, set]) -> Dict[str, str]:
    row = {field: "" for field in fields}

    nct_id = xml_text(root, "id_info/nct_id")
    if not nct_id:
        nct_id = xml_text(root, "nct_id")
    if "NCT_No" in row:
        row["NCT_No"] = nct_id

    org_study_id = xml_text(root, "id_info/org_study_id")
    if "Prot_No" in row:
        row["Prot_No"] = org_study_id

    secondary_ids = parse_secondary_ids(root)
    euct_ids = []
    other_ids = []
    for sid in secondary_ids:
        if EUDRACT_RE.search(sid) or "eudract" in sid.lower():
            euct_ids.append(sid)
        else:
            other_ids.append(sid)
    if "EUCT_No" in row:
        row["EUCT_No"] = join_values(euct_ids)
    if "Other_No" in row:
        row["Other_No"] = join_values(other_ids)

    interventions = parse_interventions(root)
    inv_names = []
    for iv in interventions:
        if iv["name"]:
            inv_names.append(iv["name"])
        for other in iv["other_names"]:
            inv_names.append(other)
    if "Inv_Prod" in row:
        row["Inv_Prod"] = join_values(inv_names)

    conditions = xml_texts(root, "condition")
    if "Indication" in row:
        row["Indication"] = join_values(conditions)

    if "Study_Phase" in row:
        row["Study_Phase"] = xml_text(root, "phase")

    if "Start_Date" in row:
        row["Start_Date"] = xml_text(root, "start_date")

    completion_date = xml_text(root, "completion_date")
    if "Complet_Date" in row:
        row["Complet_Date"] = completion_date

    overall_status = xml_text(root, "overall_status")
    if "Termin_Date" in row:
        row["Termin_Date"] = completion_date if overall_status.lower() == "terminated" else ""

    if "Report_Date" in row:
        row["Report_Date"] = xml_text(root, "study_first_posted")

    officials = parse_officials(root)
    if not officials:
        rp_name = xml_text(root, "responsible_party/investigator_full_name")
        if rp_name:
            officials.append(rp_name)
    if "Name_PI" in row:
        row["Name_PI"] = join_values(officials)

    sponsors = parse_sponsors(root)
    if "Sponsor" in row:
        row["Sponsor"] = join_values(sponsors)

    enrollment = root.find("enrollment")
    enrollment_text = normalize_whitespace(enrollment.text or "") if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    enrollment_value = safe_int(enrollment_text)
    planned_enrollment = enrollment_value if enrollment_type == "anticipated" else None
    actual_enrollment = enrollment_value if enrollment_type == "actual" else None
    if "No_Subj_Planned" in row:
        row["No_Subj_Planned"] = str(planned_enrollment) if planned_enrollment is not None else ""
    if ACTUAL_ENROLL_FIELD in row:
        row[ACTUAL_ENROLL_FIELD] = str(actual_enrollment) if actual_enrollment is not None else ""

    dmc = xml_text(root, "oversight_info/has_dmc")
    if "DMC" in row:
        row["DMC"] = dmc

    study_design = root.find("study_design_info")
    allocation = xml_text(root, "study_design_info/allocation")
    intervention_model = xml_text(root, "study_design_info/intervention_model")
    masking = xml_text(root, "study_design_info/masking")

    if "Randomization" in row:
        if allocation:
            row["Randomization"] = yes_no("random" in allocation.lower())

    if "Random_Parallel" in row and intervention_model:
        row["Random_Parallel"] = yes_no("parallel" in intervention_model.lower())
    if "Random_Crossover" in row and intervention_model:
        row["Random_Crossover"] = yes_no("crossover" in intervention_model.lower())
    if "Random_Fact" in row and intervention_model:
        row["Random_Fact"] = yes_no("factorial" in intervention_model.lower())
    if "Random_Cluster" in row and intervention_model:
        row["Random_Cluster"] = yes_no("cluster" in intervention_model.lower())

    if "Single_Arm" in row:
        if intervention_model:
            row["Single_Arm"] = yes_no("single group" in intervention_model.lower())

    if "Stratification" in row:
        strat = xml_text(root, "study_design_info/stratification")
        if strat:
            row["Stratification"] = "Yes" if strat.lower() not in {"no", "none"} else "No"

    blinding, level = infer_blinding(masking)
    if "Blinding" in row:
        row["Blinding"] = blinding
    if "Level_Blinding" in row:
        row["Level_Blinding"] = level

    arms = parse_arm_groups(root)
    arm_types = [a["type"] for a in arms if a["type"]]
    if arm_types:
        if "Placebo_control" in row:
            row["Placebo_control"] = yes_no(any("placebo" in t.lower() for t in arm_types))
        if "Active_Control" in row:
            row["Active_Control"] = yes_no(any("active" in t.lower() for t in arm_types))
        if "Hist_control" in row:
            row["Hist_control"] = yes_no(any("histor" in t.lower() for t in arm_types))

    arm_labels_by_type = {
        a["label"]: a["type"] for a in arms if a["label"] and a["type"]
    }
    control_labels = {
        label
        for label, arm_type in arm_labels_by_type.items()
        if "placebo" in arm_type.lower() or "active" in arm_type.lower()
    }
    control_drugs = []
    for iv in interventions:
        if not iv["labels"]:
            continue
        if any(label in control_labels for label in iv["labels"]):
            if iv["name"]:
                control_drugs.append(iv["name"])
            control_drugs.extend(iv["other_names"])
    if "Control_Drug" in row:
        row["Control_Drug"] = join_values(control_drugs)

    num_arms = safe_int(xml_text(root, "number_of_arms"))
    if num_arms is None and arms:
        num_arms = len(arms)
    if "No_Arm" in row:
        row["No_Arm"] = str(num_arms) if num_arms is not None else ""
    if "Single_Arm" in row and row.get("Single_Arm") == "" and num_arms is not None:
        row["Single_Arm"] = yes_no(num_arms == 1)

    primary_outcomes = parse_outcomes(root, "primary_outcome")
    secondary_outcomes = parse_outcomes(root, "secondary_outcome")
    if "EP_Primary" in row and primary_outcomes:
        row["EP_Primary"] = primary_outcomes[0]["measure"]
    if "EP_Key_Second" in row and secondary_outcomes:
        row["EP_Key_Second"] = secondary_outcomes[0]["measure"]

    locations = root.findall("location")
    location_countries = parse_locations(root)
    country_list = xml_texts(root, "location_countries/country") or location_countries
    unique_countries = []
    seen = set()
    for country in country_list:
        key = normalize_country(country)
        if key and key not in seen:
            seen.add(key)
            unique_countries.append(country)

    if "MRCT" in row:
        if unique_countries:
            row["MRCT"] = yes_no(len(unique_countries) > 2)

    total_centers = len(locations)
    if "No_Center" in row:
        row["No_Center"] = str(total_centers) if total_centers > 0 else ""

    if total_centers > 0:
        counts = {"NA": 0, "AP": 0, "WEU": 0, "EEU": 0, "AF": 0}
        for country in location_countries:
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

    return row


def extract_text_blocks(root: ET.Element, nct_id: str) -> Dict[str, object]:
    return {
        "nct_id": nct_id,
        "brief_title": xml_text(root, "brief_title"),
        "official_title": xml_text(root, "official_title"),
        "brief_summary": xml_text(root, "brief_summary/textblock"),
        "detailed_description": xml_text(root, "detailed_description/textblock"),
        "eligibility_criteria": xml_text(root, "eligibility/criteria/textblock"),
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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build Design-feature CSV from CT.gov XML.")
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=project_root / "data/raw/CSR-Vars.xlsx",
        help="Path to CSR-Vars.xlsx (Design sheet).",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=project_root / "data/raw_data",
        help="Root directory containing CT.gov XML files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "data/ctg_design_extract/design_features.csv",
        help="Output CSV path (auto-suffixed when --nct-id is set and this default is used).",
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
        default=str(project_root / "data/ctg_design_extract/ctg_text_blocks.jsonl"),
        help="Optional JSONL output for text blocks (use 'none' to skip).",
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
    project_root = Path(__file__).resolve().parents[2]
    default_csv = project_root / "data/ctg_design_extract/design_features.csv"
    default_text = project_root / "data/ctg_design_extract/ctg_text_blocks.jsonl"

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Missing xlsx: {args.xlsx}")
    if not args.xml_root.exists():
        raise FileNotFoundError(f"Missing xml root: {args.xml_root}")

    fields = ensure_extra_fields(load_design_fields(args.xlsx))
    region_map = region_sets()
    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)
    if nct_ids:
        suffix = suffix_for_ncts(nct_ids)
        if args.output_csv == default_csv:
            args.output_csv = default_csv.with_name(f"design_features_{suffix}.csv")
        text_out_arg = (args.text_out or "").strip()
        if text_out_arg and text_out_arg.lower() not in {"none", "null"}:
            if Path(text_out_arg) == default_text:
                args.text_out = str(default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl"))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    text_out_arg = (args.text_out or "").strip()
    text_out_path = None if text_out_arg.lower() in {"", "none", "null"} else Path(text_out_arg)
    text_file = None
    if text_out_path:
        text_out_path.parent.mkdir(parents=True, exist_ok=True)
        text_file = text_out_path.open("w")

    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for xml_path in iter_xml_files(args.xml_root, nct_ids):
            try:
                root = ET.parse(xml_path).getroot()
            except Exception as exc:
                print(f"[WARN] Failed to parse {xml_path}: {exc}")
                continue

            row = extract_features(root, fields, region_map)
            writer.writerow(row)

            if text_file is not None:
                nct_id = row.get("NCT_No") or xml_path.stem
                text_blocks = extract_text_blocks(root, nct_id)
                text_file.write(json.dumps(text_blocks, ensure_ascii=False) + "\n")

    if text_file is not None:
        text_file.close()


if __name__ == "__main__":
    main()
