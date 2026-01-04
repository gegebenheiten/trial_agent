"""
Build a relation graph from TrialPanorama relations parquet tables.

Outputs a JSON dict mapping node_id -> list of neighbor node_ids.
Node IDs are formatted as "{type}:{id}" with lowercased types.

Only relations that touch a trial (as defined by the studies table)
are kept to focus on trial-centric retrieval.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from trial_agent.config import settings


def _pick_col(schema: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in schema:
            return col
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TrialPanorama relation graph for trial retrieval."
    )
    parser.add_argument("--raw-dir", type=Path, default=settings.trialpanorama_raw_dir)
    parser.add_argument("--output", type=Path, default=settings.trialpanorama_relations)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "polars is required to parse parquet. Install with `pip install polars`."
        ) from exc

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"TrialPanorama raw directory not found: {args.raw_dir}")

    rel_files = sorted(args.raw_dir.glob("relations*.parquet"))
    if not rel_files:
        raise FileNotFoundError(f"No relations parquet files found in {args.raw_dir}")

    studies_files = sorted(args.raw_dir.glob("studies*.parquet"))
    if not studies_files:
        raise FileNotFoundError(f"No studies parquet files found in {args.raw_dir}")

    studies_lf = pl.scan_parquet(studies_files)
    studies_schema = studies_lf.schema
    candidate_cols = ["study_id", "nct_id", "nctid", "trial_id"]
    study_cols = [col for col in candidate_cols if col in studies_schema]
    if not study_cols:
        raise RuntimeError("Could not find study_id/nct_id column in studies table.")

    studies = pl.concat(
        [studies_lf.select(pl.col(col).cast(pl.Utf8).alias("study_key")) for col in study_cols],
        how="vertical",
    ).unique()
    studies = studies.with_columns(pl.col("study_key").alias("study_key_join"))

    lf = pl.scan_parquet(rel_files)
    schema = lf.schema
    head_id = _pick_col(schema, ["head_id"])
    tail_id = _pick_col(schema, ["tail_id"])
    head_type = _pick_col(schema, ["head_type"])
    tail_type = _pick_col(schema, ["tail_type"])
    if not all([head_id, tail_id, head_type, tail_type]):
        raise RuntimeError("relations table missing required head/tail columns.")

    lf = lf.select([head_id, head_type, tail_id, tail_type])
    lf = lf.with_columns(
        [
            pl.col(head_id).cast(pl.Utf8).alias("head_id"),
            pl.col(tail_id).cast(pl.Utf8).alias("tail_id"),
            pl.col(head_type)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .fill_null("unknown")
            .alias("head_type"),
            pl.col(tail_type)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .fill_null("unknown")
            .alias("tail_type"),
        ]
    )

    def mark_is_study(frame: "pl.LazyFrame", id_col: str, flag_name: str) -> "pl.LazyFrame":
        frame = frame.join(studies, left_on=id_col, right_on="study_key_join", how="left")
        return frame.with_columns(pl.col("study_key").is_not_null().alias(flag_name)).drop(
            "study_key"
        )

    lf = mark_is_study(lf, "head_id", "head_is_study")
    lf = mark_is_study(lf, "tail_id", "tail_is_study")

    # Keep edges touching studies to reduce graph size.
    lf = lf.filter(pl.col("head_is_study") | pl.col("tail_is_study"))

    df = lf.collect(streaming=True)
    adjacency: Dict[str, set] = {}
    for row in df.iter_rows(named=True):
        head_type_val = "study" if row.get("head_is_study") else row.get("head_type", "unknown")
        tail_type_val = "study" if row.get("tail_is_study") else row.get("tail_type", "unknown")
        head_node = f"{head_type_val}:{row['head_id']}"
        tail_node = f"{tail_type_val}:{row['tail_id']}"
        adjacency.setdefault(head_node, set()).add(tail_node)
        adjacency.setdefault(tail_node, set()).add(head_node)

    output = {node: sorted(list(neighbors)) for node, neighbors in adjacency.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True))
    print(f"Wrote {len(output)} nodes to {args.output}")


if __name__ == "__main__":
    main()
