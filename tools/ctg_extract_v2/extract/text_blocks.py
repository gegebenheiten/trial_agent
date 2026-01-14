from __future__ import annotations

from typing import Dict, List
import xml.etree.ElementTree as ET

from .common import join_values, normalize_whitespace, parse_arm_groups, parse_interventions, xml_text, xml_texts


def parse_outcomes(root: ET.Element, tag: str) -> List[Dict[str, str]]:
    outcomes: List[Dict[str, str]] = []
    for outcome in root.findall(tag):
        measure = xml_text(outcome, "measure")
        time_frame = xml_text(outcome, "time_frame")
        description = xml_text(outcome, "description")
        outcomes.append(
            {
                "title": measure,
                "time_frame": time_frame,
                "description": description,
            }
        )
    return outcomes


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
        "keywords": xml_texts(root, "keyword"),
        "conditions": xml_texts(root, "condition"),
        "location_countries": xml_texts(root, "location_countries/country"),
    }
