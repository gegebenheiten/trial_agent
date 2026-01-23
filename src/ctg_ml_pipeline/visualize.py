from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


@dataclass
class MissingSummary:
    data: pl.DataFrame


@dataclass
class RangeSummary:
    numeric: pl.DataFrame
    categorical: pl.DataFrame


def _normalize_empty(df: pl.DataFrame) -> pl.DataFrame:
    str_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    if not str_cols:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
            for col in str_cols
        ]
    )


def summarize_missing(df: pl.DataFrame, exclude: Iterable[str] = ()) -> MissingSummary:
    exclude = set(exclude)
    df = _normalize_empty(df)
    total = df.height
    rows = []
    for name in df.columns:
        if name in exclude:
            continue
        missing = df.select(pl.col(name).is_null().sum()).item()
        rows.append({"feature": name, "missing": missing, "total": total, "missing_rate": missing / total if total else 0.0})
    return MissingSummary(data=pl.DataFrame(rows))


def _top_values(df: pl.DataFrame, col: str, k: int = 3) -> str:
    vc = (
        df.select(pl.col(col))
        .drop_nulls()
        .group_by(col)
        .len()
        .sort("len", descending=True)
        .head(k)
    )
    if vc.is_empty():
        return ""
    parts = []
    for row in vc.rows(named=True):
        parts.append(f"{row[col]} ({row['len']})")
    return "; ".join(parts)


def summarize_ranges(df: pl.DataFrame, exclude: Iterable[str] = ()) -> RangeSummary:
    exclude = set(exclude)
    df = _normalize_empty(df)
    numeric_rows = []
    categorical_rows = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            stats = df.select(
                pl.col(name).min().alias("min"),
                pl.col(name).max().alias("max"),
                pl.col(name).mean().alias("mean"),
                pl.col(name).std().alias("std"),
            ).rows()[0]
            numeric_rows.append({
                "feature": name,
                "min": stats[0],
                "max": stats[1],
                "mean": stats[2],
                "std": stats[3],
            })
        else:
            categorical_rows.append({
                "feature": name,
                "unique": df.select(pl.col(name).n_unique()).item(),
                "top_values": _top_values(df, name),
            })
    return RangeSummary(
        numeric=pl.DataFrame(numeric_rows),
        categorical=pl.DataFrame(categorical_rows),
    )


def export_summaries(
    df: pl.DataFrame,
    output_dir: str | Path,
    exclude: Iterable[str] = (),
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    missing = summarize_missing(df, exclude=exclude).data
    ranges = summarize_ranges(df, exclude=exclude)
    missing.write_csv(output_path / "missing_rate.csv")
    ranges.numeric.write_csv(output_path / "numeric_ranges.csv")
    ranges.categorical.write_csv(output_path / "categorical_summary.csv")
