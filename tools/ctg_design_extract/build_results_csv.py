#!/usr/bin/env python3
"""
Create Results CSVs from ClinicalTrials.gov XML files.
Outputs are split into outcomes, measurements, and analyses.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WHITESPACE_RE = re.compile(r"\s+")
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")


OUTCOME_FIELDS = [
    "nct_id",
    "outcome_id",
    "outcome_type",
    "outcome_title",
    "outcome_description",
    "outcome_time_frame",
    "outcome_population",
    "is_key_secondary",
]

MEASUREMENT_FIELDS = [
    "nct_id",
    "outcome_id",
    "measure_id",
    "measure_title",
    "measure_description",
    "unit",
    "param_type",
    "dispersion_type",
    "dispersion_value",
    "lower_limit",
    "upper_limit",
    "n_analyzed",
    "group_id",
    "group_title",
    "group_description",
    "category_title",
    "class_title",
    "value",
    "value_text",
    "measure_time_frame",
    "measure_population",
]

ANALYSIS_FIELDS = [
    "nct_id",
    "outcome_id",
    "analysis_id",
    "non_inferiority_type",
    "method",
    "param_type",
    "groups_desc",
    "method_desc",
    "estimate_desc",
    "p_value",
    "p_value_desc",
    "group_ids",
]


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def xml_text(elem: Optional[ET.Element], path: str) -> str:
    if elem is None:
        return ""
    return normalize_whitespace(elem.findtext(path) or "")


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


def parse_group_map(outcome: ET.Element) -> Dict[str, Dict[str, str]]:
    groups = {}
    for group in outcome.findall("group_list/group"):
        gid = (group.attrib.get("group_id") or "").strip()
        title = xml_text(group, "title")
        description = xml_text(group, "description")
        if gid or title or description:
            groups[gid] = {"title": title, "description": description}
    return groups


def parse_analyzed_counts(measure: ET.Element) -> Dict[str, str]:
    counts: Dict[str, List[str]] = {}
    for analyzed in measure.findall("analyzed_list/analyzed"):
        for count in analyzed.findall("count_list/count"):
            gid = (count.attrib.get("group_id") or "").strip()
            value = (count.attrib.get("value") or "").strip()
            if not value:
                continue
            counts.setdefault(gid, []).append(value)
    return {gid: join_values(values) for gid, values in counts.items()}


def iter_measurement_elements(measure: ET.Element) -> Iterable[Tuple[str, str, ET.Element]]:
    classes = measure.findall("class_list/class")
    if not classes:
        for measurement in measure.findall("measurement_list/measurement"):
            yield "", "", measurement
        return

    for class_el in classes:
        class_title = xml_text(class_el, "title")
        categories = class_el.findall("category_list/category")
        if not categories:
            for measurement in class_el.findall("measurement_list/measurement"):
                yield class_title, "", measurement
            continue
        for category_el in categories:
            category_title = xml_text(category_el, "title")
            measurements = category_el.findall("measurement_list/measurement")
            if not measurements:
                measurements = category_el.findall("measurement")
            for measurement in measurements:
                yield class_title, category_title, measurement


def format_value_text(value: str, spread: str, lower: str, upper: str) -> str:
    if not value:
        return ""
    if lower or upper:
        return f"{value} [{lower}, {upper}]".strip()
    if spread:
        return f"{value} (spread={spread})"
    return value


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build Results CSVs from CT.gov XML.")
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=project_root / "data/raw_data",
        help="Root directory containing CT.gov XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/ctg_results_extract",
        help="Output directory for results CSVs.",
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
    return parser.parse_args()


def resolve_output_paths(output_dir: Path, nct_ids: List[str]) -> Tuple[Path, Path, Path]:
    outcomes_path = output_dir / "results_outcomes.csv"
    measurements_path = output_dir / "results_measurements.csv"
    analyses_path = output_dir / "results_analyses.csv"
    if nct_ids:
        suffix = suffix_for_ncts(nct_ids)
        outcomes_path = output_dir / f"results_outcomes_{suffix}.csv"
        measurements_path = output_dir / f"results_measurements_{suffix}.csv"
        analyses_path = output_dir / f"results_analyses_{suffix}.csv"
    return outcomes_path, measurements_path, analyses_path


def main() -> None:
    args = parse_args()
    if not args.xml_root.exists():
        raise FileNotFoundError(f"Missing xml root: {args.xml_root}")

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path, measurements_path, analyses_path = resolve_output_paths(args.output_dir, nct_ids)

    with outcomes_path.open("w", newline="") as f_outcomes, \
        measurements_path.open("w", newline="") as f_measurements, \
        analyses_path.open("w", newline="") as f_analyses:
        outcomes_writer = csv.DictWriter(f_outcomes, fieldnames=OUTCOME_FIELDS)
        measurements_writer = csv.DictWriter(f_measurements, fieldnames=MEASUREMENT_FIELDS)
        analyses_writer = csv.DictWriter(f_analyses, fieldnames=ANALYSIS_FIELDS)
        outcomes_writer.writeheader()
        measurements_writer.writeheader()
        analyses_writer.writeheader()

        for xml_path in iter_xml_files(args.xml_root, nct_ids):
            try:
                root = ET.parse(xml_path).getroot()
            except Exception as exc:
                print(f"[WARN] Failed to parse {xml_path}: {exc}")
                continue

            nct_id = xml_text(root, "id_info/nct_id") or xml_text(root, "nct_id") or xml_path.stem
            results = root.find("clinical_results")
            if results is None:
                continue
            outcome_list = results.find("outcome_list")
            if outcome_list is None:
                continue

            outcomes = outcome_list.findall("outcome")
            for outcome_idx, outcome in enumerate(outcomes, start=1):
                outcome_id = str(outcome_idx)
                outcome_type = xml_text(outcome, "type")
                outcome_title = xml_text(outcome, "title")
                outcome_description = xml_text(outcome, "description")
                outcome_time_frame = xml_text(outcome, "time_frame")
                outcome_population = xml_text(outcome, "population")

                outcomes_writer.writerow(
                    {
                        "nct_id": nct_id,
                        "outcome_id": outcome_id,
                        "outcome_type": outcome_type,
                        "outcome_title": outcome_title,
                        "outcome_description": outcome_description,
                        "outcome_time_frame": outcome_time_frame,
                        "outcome_population": outcome_population,
                        "is_key_secondary": "",
                    }
                )

                group_map = parse_group_map(outcome)

                measures = outcome.findall("measure")
                for measure_idx, measure in enumerate(measures, start=1):
                    measure_id = str(measure_idx)
                    measure_title = xml_text(measure, "title")
                    measure_description = xml_text(measure, "description")
                    unit = xml_text(measure, "units")
                    param_type = xml_text(measure, "param")
                    dispersion_type = xml_text(measure, "dispersion")
                    measure_time_frame = xml_text(measure, "time_frame")
                    measure_population = xml_text(measure, "population")

                    counts_by_group = parse_analyzed_counts(measure)
                    for class_title, category_title, measurement in iter_measurement_elements(measure):
                        group_id = (measurement.attrib.get("group_id") or "").strip()
                        value = (measurement.attrib.get("value") or "").strip()
                        spread = (measurement.attrib.get("spread") or "").strip()
                        lower_limit = (measurement.attrib.get("lower_limit") or "").strip()
                        upper_limit = (measurement.attrib.get("upper_limit") or "").strip()
                        n_analyzed = counts_by_group.get(group_id, counts_by_group.get("", ""))
                        group_info = group_map.get(group_id, {})
                        measurements_writer.writerow(
                            {
                                "nct_id": nct_id,
                                "outcome_id": outcome_id,
                                "measure_id": measure_id,
                                "measure_title": measure_title,
                                "measure_description": measure_description,
                                "unit": unit,
                                "param_type": param_type,
                                "dispersion_type": dispersion_type,
                                "dispersion_value": spread,
                                "lower_limit": lower_limit,
                                "upper_limit": upper_limit,
                                "n_analyzed": n_analyzed,
                                "group_id": group_id,
                                "group_title": group_info.get("title", ""),
                                "group_description": group_info.get("description", ""),
                                "category_title": category_title,
                                "class_title": class_title,
                                "value": value,
                                "value_text": format_value_text(value, spread, lower_limit, upper_limit),
                                "measure_time_frame": measure_time_frame,
                                "measure_population": measure_population,
                            }
                        )

                analyses = outcome.findall("analysis_list/analysis")
                for analysis_idx, analysis in enumerate(analyses, start=1):
                    group_ids = join_values(
                        [normalize_whitespace(g.text or "") for g in analysis.findall("group_id_list/group_id")]
                    )
                    analyses_writer.writerow(
                        {
                            "nct_id": nct_id,
                            "outcome_id": outcome_id,
                            "analysis_id": str(analysis_idx),
                            "non_inferiority_type": xml_text(analysis, "non_inferiority_type"),
                            "method": xml_text(analysis, "method"),
                            "param_type": xml_text(analysis, "param_type"),
                            "groups_desc": xml_text(analysis, "groups_desc"),
                            "method_desc": xml_text(analysis, "method_desc"),
                            "estimate_desc": xml_text(analysis, "estimate_desc"),
                            "p_value": xml_text(analysis, "p_value"),
                            "p_value_desc": xml_text(analysis, "p_value_desc"),
                            "group_ids": group_ids,
                        }
                    )


if __name__ == "__main__":
    main()
