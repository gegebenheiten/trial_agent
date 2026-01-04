import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _table_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_part_\\d+(?:_\\d+)?$", "", name)
    name = re.sub(r"_chunk_\\d+(?:_\\d+)?$", "", name)
    name = re.sub(r"_\\d+$", "", name)
    return name


def _primary_key_guess(table: str, columns: List[str]) -> str:
    candidates = ["study_id", "nct_id", "nctid", "trial_id"]
    for col in candidates:
        if col in columns:
            return col
    if table == "relations" and "head_id" in columns and "tail_id" in columns:
        if "relation_type" in columns:
            return "head_id + tail_id + relation_type"
        return "head_id + tail_id"
    if "drug_moa_id" in columns:
        return "drug_moa_id"
    id_like = [c for c in columns if c.endswith("_id")]
    if id_like:
        return " + ".join(id_like[:2]) if len(id_like) > 1 else id_like[0]
    return "unknown"


def _truncate(value: Optional[str], limit: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TrialPanorama tables.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/trialpanorama/raw"),
        help="Directory containing TrialPanorama parquet files.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file to print one sample record.",
    )
    parser.add_argument(
        "--jsonl-index",
        type=int,
        default=0,
        help="Record index to print from JSONL (0-based).",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Optional path to write the selected JSONL record.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1,
        help="Number of sample rows to print per table (0 disables).",
    )
    parser.add_argument(
        "--sample-table",
        type=str,
        default="",
        help="If set, only print sample for this table name.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=160,
        help="Max length for string fields in sample output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jsonl:
        if not args.jsonl.exists():
            raise FileNotFoundError(f"JSONL not found: {args.jsonl}")
        with args.jsonl.open() as f:
            for idx, line in enumerate(f):
                if idx == args.jsonl_index:
                    record = json.loads(line)
                    output = json.dumps(record, ensure_ascii=False, indent=2)
                    print(output)
                    if args.jsonl_output:
                        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
                        args.jsonl_output.write_text(output)
                    return
        raise IndexError(f"JSONL index out of range: {args.jsonl_index}")

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {args.raw_dir}")

    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "polars is required to read parquet. Install with `pip install polars`."
        ) from exc

    files = sorted(args.raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.raw_dir}")

    grouped: Dict[str, List[Path]] = {}
    for path in files:
        grouped.setdefault(_table_name(path), []).append(path)

    for table in sorted(grouped.keys()):
        table_files = grouped[table]
        lf = pl.scan_parquet(table_files)
        schema = lf.collect_schema()
        columns = list(schema.keys())
        id_columns = [c for c in columns if c.endswith("_id") or c in {"nct_id", "nctid"}]
        print("=" * 80)
        print(f"Table: {table}")
        print(f"Files: {len(table_files)}")
        print(f"Columns ({len(columns)}): {columns}")
        print(f"ID columns: {id_columns}")
        print(f"Primary key (heuristic): {_primary_key_guess(table, columns)}")
        if args.sample_rows > 0 and (not args.sample_table or args.sample_table == table):
            df = lf.limit(args.sample_rows).collect()
            for row in df.iter_rows(named=True):
                sample = {}
                for key, value in row.items():
                    if isinstance(value, str):
                        sample[key] = _truncate(value, args.max_len)
                    else:
                        try:
                            sample[key] = json.loads(json.dumps(value))
                        except TypeError:
                            sample[key] = value
                print(f"Sample row: {sample}")


if __name__ == "__main__":
    main()
