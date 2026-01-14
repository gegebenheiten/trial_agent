from __future__ import annotations

from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET

from .common import (
    arm_type_matches,
    extract_eudract_id,
    infer_blinding,
    is_drug_like,
    is_placebo_like_name,
    join_values,
    normalize_intervention_model,
    normalize_label,
    normalize_country,
    parse_arm_groups,
    parse_interventions,
    parse_location_centers,
    parse_officials,
    parse_secondary_ids,
    parse_sponsors,
    region_for_country,
    region_sets,
    safe_int,
    set_if_present,
    xml_text,
    xml_texts,
    yes_no,
)


CONTROL_ARM_KEYWORDS = (
    "placebo",
    "active",
    "sham",
    "no intervention",
    "histor",
)


def extract_d_design_rows(root: ET.Element, fields: List[str], nct_id: str) -> Iterable[Dict[str, str]]:
    row = {field: "" for field in fields}

    set_if_present(row, "StudyID", nct_id)
    set_if_present(row, "NCT_No", nct_id)
    set_if_present(row, "NCT_ID", nct_id)

    set_if_present(row, "Prot_No", xml_text(root, "id_info/org_study_id"))

    secondary_ids = parse_secondary_ids(root)
    euct_ids = []
    other_ids = []
    for sid in secondary_ids:
        eudract_id = extract_eudract_id(sid)
        if eudract_id:
            euct_ids.append(eudract_id)
        else:
            other_ids.append(sid)
    set_if_present(row, "EUCT_No", join_values(euct_ids))
    set_if_present(row, "Other_No", join_values(other_ids))

    interventions = parse_interventions(root)
    arms = parse_arm_groups(root)

    intervention_names = [str(iv.get("name") or "").strip() for iv in interventions if iv.get("name")]
    intervention_all = join_values(intervention_names)
    for field in ("Intervention_All", "intervention_all"):
        set_if_present(row, field, intervention_all)

    set_if_present(row, "Brief_Title", xml_text(root, "brief_title"))
    set_if_present(row, "Official_Title", xml_text(root, "official_title"))
    set_if_present(row, "Brief_Summary", xml_text(root, "brief_summary/textblock"))

    eligibility_text = xml_text(root, "eligibility/criteria/textblock")
    for field in ("Eligibility_Criteria", "Eligibility", "Inclusion_Exclusion"):
        set_if_present(row, field, eligibility_text)

    set_if_present(row, "Study_Phase", xml_text(root, "phase"))
    set_if_present(row, "Study_Type", xml_text(root, "study_type"))

    officials = parse_officials(root)
    pi_officials = [
        official["name"]
        for official in officials
        if "principal investigator" in (official.get("role") or "").lower()
    ]
    rp_name = xml_text(root, "responsible_party/investigator_full_name")
    if pi_officials:
        set_if_present(row, "Name_PI", join_values(pi_officials))
    elif rp_name:
        set_if_present(row, "Name_PI", rp_name)

    sponsors = parse_sponsors(root)
    set_if_present(row, "Sponsor", join_values(sponsors))

    dmc = xml_text(root, "oversight_info/has_dmc")
    if dmc.lower() in {"yes", "no"}:
        set_if_present(row, "DMC", dmc.title())
    else:
        set_if_present(row, "DMC", dmc)

    num_arms = safe_int(xml_text(root, "number_of_arms"))
    if num_arms is None and arms:
        num_arms = len(arms)
    if num_arms is not None:
        set_if_present(row, "No_Arm", str(num_arms))
        set_if_present(row, "Single_Arm", yes_no(num_arms == 1))

    study_type = xml_text(root, "study_type").lower()
    is_interventional = "interventional" in study_type
    allocation = xml_text(root, "study_design_info/allocation")
    allocation_lower = allocation.lower()
    if is_interventional and allocation_lower:
        if "random" in allocation_lower:
            set_if_present(row, "Randomization", yes_no("non" not in allocation_lower))

    model = normalize_intervention_model(xml_text(root, "study_design_info/intervention_model"))
    if is_interventional and model:
        set_if_present(row, "Random_Parallel", yes_no("parallel" in model))
        set_if_present(row, "Random_Crossover", yes_no("crossover" in model))
        set_if_present(row, "Random_Fact", yes_no("factorial" in model))

    masking = xml_text(root, "study_design_info/masking")
    blinding, level = infer_blinding(masking)
    if is_interventional:
        set_if_present(row, "Blinding", blinding)
        set_if_present(row, "Level_Blinding", level)

    arm_types = [str(arm.get("type") or "") for arm in arms if arm.get("type")]
    arm_type_lowers = [arm_type.lower() for arm_type in arm_types]
    has_decisive_arm_types = any("other" not in t for t in arm_type_lowers)
    if has_decisive_arm_types and is_interventional:
        set_if_present(row, "Placebo_control", yes_no(any("placebo" in t for t in arm_type_lowers)))
        set_if_present(row, "Active_Control", yes_no(any("active" in t for t in arm_type_lowers)))
        set_if_present(row, "Hist_control", yes_no(any("histor" in t for t in arm_type_lowers)))

    arm_labels_by_type = {
        normalize_label(arm.get("label", "")): str(arm.get("type", ""))
        for arm in arms
        if arm.get("label") and arm.get("type")
    }
    control_labels = {
        label
        for label, arm_type in arm_labels_by_type.items()
        if arm_type_matches(arm_type, CONTROL_ARM_KEYWORDS)
    }
    control_drugs = []
    for iv in interventions:
        labels = iv.get("labels") or []
        normalized_labels = {normalize_label(label) for label in labels if label}
        if not normalized_labels or not (normalized_labels & control_labels):
            continue
        if not is_drug_like(str(iv.get("type") or "")):
            continue
        name = str(iv.get("name") or "").strip()
        if name and not is_placebo_like_name(name):
            control_drugs.append(name)
    set_if_present(row, "Control_Drug", join_values(control_drugs))

    enrollment = root.find("enrollment")
    enrollment_text = (enrollment.text or "").strip() if enrollment is not None else ""
    enrollment_type = (enrollment.attrib.get("type") or "").lower() if enrollment is not None else ""
    enrollment_value = safe_int(enrollment_text)
    if enrollment_value is not None:
        if enrollment_type == "anticipated":
            set_if_present(row, "Sample_Size", str(enrollment_value))
        elif enrollment_type == "actual":
            set_if_present(row, "Sample_Size_Actual", str(enrollment_value))
        else:
            set_if_present(row, "Sample_Size", str(enrollment_value))

    primary_measures = xml_texts(root, "primary_outcome/measure")
    secondary_measures = xml_texts(root, "secondary_outcome/measure")
    if primary_measures:
        set_if_present(row, "No_Primary_EP", str(len(primary_measures)))
        set_if_present(row, "Name_Primary_EP", join_values(primary_measures))
    if secondary_measures:
        set_if_present(row, "No_Second_EP", str(len(secondary_measures)))
        set_if_present(row, "Name_Second_EP", join_values(secondary_measures))

    centers = parse_location_centers(root)
    location_countries = [country for _, _, country in centers if country]
    country_list = xml_texts(root, "location_countries/country") or location_countries
    unique_countries: List[str] = []
    seen = set()
    for country in country_list:
        key = normalize_country(country)
        if key and key not in seen:
            seen.add(key)
            unique_countries.append(country)

    region_map = region_sets()
    if unique_countries:
        region_hits = set()
        for country in unique_countries:
            region = region_for_country(country, region_map)
            if region:
                region_hits.add(region)
        if region_hits:
            set_if_present(row, "MRCT", yes_no(len(region_hits) >= 2))

    total_centers = len(centers)
    if total_centers > 0:
        set_if_present(row, "No_Center", str(total_centers))
        counts = {"NA": 0, "AP": 0, "WEU": 0, "EEU": 0, "AF": 0}
        for _, _, country in centers:
            region = region_for_country(country, region_map)
            if region:
                counts[region] += 1
        set_if_present(row, "No_Center_NA", str(counts["NA"]))
        set_if_present(row, "No_Center_AP", str(counts["AP"]))
        set_if_present(row, "No_Center_WEU", str(counts["WEU"]))
        set_if_present(row, "No_Center_EEU", str(counts["EEU"]))
        set_if_present(row, "No_Center_AF", str(counts["AF"]))

    yield row
