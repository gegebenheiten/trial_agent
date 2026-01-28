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


def _load_labeled_ids(labels_csv: str | Path) -> set[str]:
    df = pl.read_csv(labels_csv)
    if "StudyID" in df.columns:
        ids = df.get_column("StudyID")
    elif "nctid" in df.columns:
        ids = df.get_column("nctid")
    else:
        raise ValueError("labels CSV must contain StudyID or nctid column")
    return {str(v).strip() for v in ids.to_list() if v is not None and str(v).strip()}


def _filter_to_labeled(df: pl.DataFrame, labeled_ids: set[str]) -> pl.DataFrame:
    if "StudyID" not in df.columns or not labeled_ids:
        return df
    return df.filter(pl.col("StudyID").is_in(sorted(labeled_ids)))


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


def plot_missingness_ranked(
    df: pl.DataFrame,
    output_path: str | Path,
    exclude: Iterable[str] = (),
    top_n: int | None = None,
    labeled_ids: set[str] | None = None,
) -> pl.DataFrame:
    """
    Plot features ranked by proportion of missingness (heatmap).

    Returns the underlying DataFrame used for plotting.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    if labeled_ids:
        df = _filter_to_labeled(df, labeled_ids)
    missing_df = summarize_missing(df, exclude=exclude).data
    if missing_df.is_empty():
        raise ValueError("No features available to plot.")

    missing_df = missing_df.sort("missing_rate", descending=True)
    if top_n is not None and top_n > 0:
        missing_df = missing_df.head(top_n)

    features = missing_df.get_column("feature").to_list()
    rates = missing_df.get_column("missing_rate").to_list()

    fig_height = max(4, 0.25 * len(features))
    plt.figure(figsize=(6, fig_height))
    data = [[r] for r in rates[::-1]]
    im = plt.imshow(data, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Proportion missing")
    plt.yticks(range(len(features)), features[::-1])
    plt.xticks([0], ["missing_rate"])
    plt.title("Features ranked by proportion of missingness")
    for i, rate in enumerate(rates[::-1]):
        plt.text(0, i, f"{rate:.1%}", va="center", ha="center", color="black", fontsize=7)
    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return missing_df


def plot_missingness_ranked_by_table(
    tables_dir: str | Path,
    output_dir: str | Path,
    exclude: Iterable[str] = (),
    top_n: int | None = None,
    labeled_ids: set[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Plot missingness ranking per table in a directory of CSVs.

    Returns mapping table_name -> missingness DataFrame.
    """
    tables_path = Path(tables_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pl.DataFrame] = {}
    for csv_path in sorted(tables_path.glob("*.csv")):
        df = pl.read_csv(csv_path, infer_schema_length=0)
        out_path = out_dir / f"{csv_path.stem}_missingness.png"
        results[csv_path.stem] = plot_missingness_ranked(
            df,
            output_path=out_path,
            exclude=exclude,
            top_n=top_n,
            labeled_ids=labeled_ids,
        )
    return results
