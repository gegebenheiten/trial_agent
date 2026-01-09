#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple
import xml.etree.ElementTree as ET

from build_ctg_tables import (
    extract_text_blocks,
    iter_xml_files,
    load_nct_ids_from_csv,
    merge_nct_ids,
    parse_nct_ids,
    suffix_for_ncts,
    xml_text,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build CTG text blocks from CT.gov XML.")
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
        help="Root output directory for CTG text blocks (by_nctid/<NCTID>/ctg_text_blocks.jsonl).",
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


def normalize_text_paths(base: Path) -> Tuple[Path, Path]:
    if base.suffix == ".jsonl":
        return base, base.with_suffix(".json")
    if base.suffix == ".json":
        return base.with_suffix(".jsonl"), base
    return base.with_suffix(".jsonl"), base.with_suffix(".json")


def main() -> None:
    args = parse_args()
    if not args.xml_root.exists():
        raise FileNotFoundError(f"Missing xml root: {args.xml_root}")

    csv_ids = []
    if args.nct_csv:
        csv_ids = load_nct_ids_from_csv(args.nct_csv, args.nct_id_col, args.limit)
    nct_ids = merge_nct_ids(csv_ids, parse_nct_ids(args.nct_id), args.limit)
    suffix = suffix_for_ncts(nct_ids) if nct_ids else ""

    default_text = args.output_root / "ctg_text_blocks.jsonl"
    text_out_arg = (args.text_out or "").strip()
    text_out_lower = text_out_arg.lower()
    text_format = (args.text_format or "auto").lower()

    if nct_ids:
        if len(nct_ids) == 1:
            nct_id = nct_ids[0]
            per_nct_text = args.output_root / "by_nctid" / nct_id / "ctg_text_blocks.jsonl"
            if text_out_lower in {"", "auto"}:
                args.text_out = str(per_nct_text)
            elif text_out_lower in {"none", "null"}:
                pass
            elif text_out_arg and Path(text_out_arg) == default_text:
                args.text_out = str(per_nct_text)
        else:
            if text_out_lower in {"", "auto"}:
                args.text_out = str(default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl"))
            elif text_out_lower not in {"none", "null"} and text_out_arg and Path(text_out_arg) == default_text:
                args.text_out = str(default_text.with_name(f"ctg_text_blocks_{suffix}.jsonl"))

    text_out_arg = (args.text_out or "").strip()
    if text_out_arg.lower() in {"", "none", "null"}:
        text_out_path = None
    else:
        text_out_path = Path(text_out_arg)

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

    for xml_path in iter_xml_files(args.xml_root, nct_ids):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            print(f"[WARN] Failed to parse {xml_path}: {exc}")
            continue

        nct_id = xml_text(root, "id_info/nct_id") or xml_text(root, "nct_id") or xml_path.stem
        text_blocks = extract_text_blocks(root, nct_id)

        if text_jsonl_file is not None:
            text_jsonl_file.write(json.dumps(text_blocks, ensure_ascii=False) + "\n")
        if text_json_file is not None:
            if not text_json_first:
                text_json_file.write(",\n")
            entry = json.dumps(text_blocks, ensure_ascii=False, indent=2)
            indented = "\n".join(f"  {line}" if line else line for line in entry.splitlines())
            text_json_file.write(indented)
            text_json_first = False

    if text_jsonl_file is not None:
        text_jsonl_file.close()
    if text_json_file is not None:
        if not text_json_first:
            text_json_file.write("\n")
        text_json_file.write("]\n")
        text_json_file.close()


if __name__ == "__main__":
    main()
