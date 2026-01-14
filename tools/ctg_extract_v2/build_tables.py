#!/usr/bin/env python3
"""
Build CTG tables (v2) from ClinicalTrials.gov XML files.
Outputs D_Design/D_Pop/D_Drug/R_Study/R_Arm using schema_fields.py.
Writes to data/ctg_extract_v2/<NCTID>/<table>.csv by default.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
import xml.etree.ElementTree as ET

from schema import load_sheet_fields, parse_tables
from extract.common import iter_xml_files, load_nct_ids_from_csv, merge_nct_ids, parse_nct_ids, xml_text
from extract.d_design import extract_d_design_rows
from extract.d_pop import extract_d_pop_rows
from extract.d_drug import build_drugbank_index, extract_d_drug_rows_with_index, load_drugbank_minimal
from extract.r_study import extract_r_study_rows
from extract.r_arm import GROUP_ARM_MAPPING_FIELDS, build_group_arm_mapping_rows, extract_r_arm_rows


TABLE_EXTRACTORS = {
    "D_Design": extract_d_design_rows,
    "D_Pop": extract_d_pop_rows,
    "R_Study": extract_r_study_rows,
    "R_Arm": extract_r_arm_rows,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build CTG tables (v2) from CT.gov XML.")
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=project_root / "data/raw/CSR-Vars 2026-01-12.xlsx",
        help="Ignored (field lists are loaded from tools/ctg_extract_v2/schema_fields.py).",
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
        default=project_root / "data/ctg_extract_v2",
        help="Root output directory for CTG tables (<NCTID>/<table>.csv).",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default="",
        help="Comma-separated list of tables to output.",
    )
    parser.add_argument(
        "--drugbank-jsonl",
        type=Path,
        default=project_root / "data/processed/drugbank_minimal.jsonl",
        help="DrugBank minimal JSONL for D_Drug enrichment.",
    )
    parser.add_argument(
        "--nct-id",
        "--nct_id",
        dest="nct_id",
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


def output_path_for_table(output_root: Path, table: str, nct_id: str) -> Path:
    return output_root / nct_id / f"{table}.csv"


def main() -> None:
    args = parse_args()

    if not args.xml_root.exists():
        raise FileNotFoundError(f"Missing xml root: {args.xml_root}")

    tables = parse_tables(args.tables)
    build_d_drug = "D_Drug" in tables
    tables_no_drug = tuple(table for table in tables if table != "D_Drug")
    table_fields = {table: load_sheet_fields(args.xlsx, table) for table in tables_no_drug}
    drug_fields = load_sheet_fields(args.xlsx, "D_Drug") if build_d_drug else []

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)

    drugbank_index = (
        build_drugbank_index(load_drugbank_minimal(args.drugbank_jsonl)) if build_d_drug else {}
    )

    for xml_path in iter_xml_files(args.xml_root, nct_ids):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            print(f"[WARN] Failed to parse {xml_path}: {exc}")
            continue

        nct_id = xml_text(root, "id_info/nct_id") or xml_text(root, "nct_id") or xml_path.stem
        if tables_no_drug:
            output_paths = {
                table: output_path_for_table(args.output_root, table, nct_id) for table in tables_no_drug
            }
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

            with ExitStack() as stack:
                writers = {}
                for table in tables_no_drug:
                    handle = stack.enter_context(output_paths[table].open("w", newline=""))
                    writer = csv.DictWriter(handle, fieldnames=table_fields[table], extrasaction="ignore")
                    writer.writeheader()
                    writers[table] = writer

                for table in tables_no_drug:
                    extractor = TABLE_EXTRACTORS.get(table)
                    if extractor is None:
                        continue
                    for row in extractor(root, table_fields[table], nct_id):
                        writers[table].writerow(row)

            if "R_Arm" in tables_no_drug:
                mapping_rows = build_group_arm_mapping_rows(root, nct_id)
                if mapping_rows:
                    mapping_path = args.output_root / nct_id / "GroupArmMapping.csv"
                    with mapping_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=GROUP_ARM_MAPPING_FIELDS,
                            extrasaction="ignore",
                        )
                        writer.writeheader()
                        for row in mapping_rows:
                            writer.writerow(row)

        if build_d_drug:
            drug_path = output_path_for_table(args.output_root, "D_Drug", nct_id)
            drug_path.parent.mkdir(parents=True, exist_ok=True)
            with drug_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=drug_fields, extrasaction="ignore")
                writer.writeheader()
                for row in extract_d_drug_rows_with_index(root, drug_fields, nct_id, drugbank_index):
                    writer.writerow(row)

if __name__ == "__main__":
    main()
