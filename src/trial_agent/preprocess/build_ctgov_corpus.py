"""
Build a processed trials.jsonl corpus from:
- A CSV containing NCT IDs (column: nctid)
- CT.gov XML files stored under data_curation/raw_data
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append('src')
from trial_agent.ingest.clean_text import normalize_whitespace
from trial_agent.ingest.parse_ctgov import normalize_trial, save_jsonl

# Resolve paths relative to the repo to avoid hard-coded absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../Trialbench/trial_agent
REPO_ROOT = PROJECT_ROOT.parent  # .../Trialbench

DEFAULT_CSV = PROJECT_ROOT / "data/raw/Phase_2_filtered_icd_C00_D48.csv"
DEFAULT_XML_ROOT = REPO_ROOT / "data_curation/raw_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/trials_ctgov_phase2_oncology.jsonl"


def load_nct_ids(csv_path: Path) -> List[str]:
    ids: List[str] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            nct = (row.get("nctid") or "").strip()
            if nct:
                ids.append(nct)
    return ids


def xml_path_for_id(nctid: str, xml_root: Path) -> Path:
    prefix = nctid[:7] + "xxxx"
    return xml_root / prefix / f"{nctid}.xml"

def split_criteria(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    lower = text.lower()
    inc_idx = lower.find("inclusion")
    exc_idx = lower.find("exclusion")
    if inc_idx != -1 and exc_idx != -1:
        if inc_idx < exc_idx:
            inclusion = text[inc_idx:exc_idx]
            exclusion = text[exc_idx:]
        else:
            exclusion = text[exc_idx:inc_idx]
            inclusion = text[inc_idx:]
    elif inc_idx != -1:
        inclusion, exclusion = text[inc_idx:], ""
    elif exc_idx != -1:
        inclusion, exclusion = "", text[exc_idx:]
    else:
        inclusion, exclusion = text, ""
    return normalize_whitespace(inclusion), normalize_whitespace(exclusion)


def classify_endpoint(measure: str) -> str:
    if not measure:
        return ""
    m = measure.lower()
    if "overall survival" in m or m.startswith("os"):
        return "OS"
    if "progression-free" in m or "pfs" in m:
        return "PFS"
    if "objective response" in m or "overall response" in m or "orr" in m:
        return "ORR"
    if "disease control" in m or "dcr" in m:
        return "DCR"
    if "duration of response" in m or "dor" in m:
        return "DoR"
    if "adverse" in m or "safety" in m or "ae" in m:
        return "Safety"
    if "quality of life" in m or "qol" in m:
        return "QoL"
    if "biomarker" in m:
        return "Biomarker"
    return ""


def parse_endpoints(root: ET.Element) -> Dict:
    primary = []
    secondary = []
    for ep in root.findall("primary_outcome"):
        primary.append(
            {
                "name": normalize_whitespace(ep.findtext("measure") or ""),
                "time_frame": normalize_whitespace(ep.findtext("time_frame") or ""),
                "description": normalize_whitespace(ep.findtext("description") or ""),
            }
        )
    for ep in root.findall("secondary_outcome"):
        secondary.append(
            {
                "name": normalize_whitespace(ep.findtext("measure") or ""),
                "time_frame": normalize_whitespace(ep.findtext("time_frame") or ""),
                "description": normalize_whitespace(ep.findtext("description") or ""),
            }
        )
    primary_type = classify_endpoint(primary[0]["name"]) if primary else ""
    return {"primary": primary, "secondary": secondary, "parsed": {"primary_type": primary_type}}


def parse_criteria(root: ET.Element) -> Dict:
    textblock = root.findtext("eligibility/criteria/textblock") or ""
    inclusion_text, exclusion_text = split_criteria(textblock)
    min_age_text = root.findtext("eligibility/minimum_age") or ""
    max_age_text = root.findtext("eligibility/maximum_age") or ""
    def _age_value(val: str):
        if not val or val.lower().startswith("n/a"):
            return None
        match = re.search(r"(\\d+)", val)
        return int(match.group(1)) if match else None
    age_min = _age_value(min_age_text)
    age_max = _age_value(max_age_text)
    return {
        "inclusion_text": inclusion_text,
        "exclusion_text": exclusion_text,
        "parsed": {
            "age_min": age_min,
            "age_max": age_max,
            "ecog_max": None,
            "prior_lines_max": None,
            "key_flags": [],
        },
    }


def parse_design(root: ET.Element) -> Dict:
    design = root.find("study_design_info")
    def _get(tag: str) -> str:
        return normalize_whitespace(design.findtext(tag) or "") if design is not None else ""
    arms = []
    for arm in root.findall("arm_group"):
        arms.append(
            {
                "name": normalize_whitespace(arm.findtext("arm_group_label") or ""),
                "description": normalize_whitespace(arm.findtext("description") or ""),
            }
        )
    return {
        "allocation": _get("allocation"),
        "intervention_model": _get("intervention_model"),
        "masking": _get("masking"),
        "primary_purpose": _get("primary_purpose"),
        "arms": arms,
        "dose": "",
    }


def parse_interventions(root: ET.Element) -> List[Dict]:
    interventions = []
    for iv in root.findall("intervention"):
        interventions.append(
            {
                "type": normalize_whitespace(iv.findtext("intervention_type") or ""),
                "name": normalize_whitespace(iv.findtext("intervention_name") or ""),
            }
        )
    return interventions


def parse_trial(nctid: str, xml_path: Path) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    trial = {
        "trial_id": nctid,
        "condition": [normalize_whitespace(c.text or "") for c in root.findall("condition")],
        "phase": normalize_whitespace(root.findtext("phase") or ""),
        "interventions": parse_interventions(root),
        "design": parse_design(root),
        "criteria": parse_criteria(root),
        "endpoints": parse_endpoints(root),
        "outcome_label": {"status": "unknown", "source": "ctgov_xml", "notes": ""},
    }
    return normalize_trial(trial)


def build_corpus(nct_ids: List[str], xml_root: Path) -> List[Dict]:
    corpus: List[Dict] = []
    for nctid in nct_ids:
        xml_path = xml_path_for_id(nctid, xml_root)
        if not xml_path.exists():
            print(f"[WARN] XML not found for {nctid} at {xml_path}")
            continue
        try:
            corpus.append(parse_trial(nctid, xml_path))
        except Exception as exc:
            print(f"[WARN] Failed to parse {nctid}: {exc}")
    return corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed trials.jsonl from CT.gov XML and NCT list.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to CSV containing nctid column.")
    parser.add_argument("--xml-root", type=Path, default=DEFAULT_XML_ROOT, help="Root directory of CT.gov XML tree.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path in processed folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nct_ids = load_nct_ids(args.csv)
    print(f"Loaded {len(nct_ids)} NCT IDs from {args.csv}")
    corpus = build_corpus(nct_ids, args.xml_root)
    print(f"Parsed {len(corpus)} trials; saving to {args.output}")
    save_jsonl(corpus, args.output)


if __name__ == "__main__":
    main()
