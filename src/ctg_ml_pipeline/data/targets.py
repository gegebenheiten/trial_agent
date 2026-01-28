from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from ctg_ml_pipeline.data.dataset import TargetConfig


def _require_pandas() -> None:
    try:
        import pandas as _  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pandas is required to read .xlsx files") from exc


def _read_excel(path: Path, sheet_name: str | None = None) -> pl.DataFrame:
    _require_pandas()
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet_name)
    # Avoid pyarrow dependency by constructing from dict-of-lists
    return pl.DataFrame(df.to_dict(orient="list"), strict=False)


def _normalize_nctid(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.extract(r"(NCT\d+)", 1).str.to_uppercase()


def _label_from_outcome(outcome_col: pl.Expr, cfg: TargetConfig) -> pl.Expr:
    return (
        pl.when(outcome_col.is_in(cfg.success_types))
        .then(pl.lit(1))
        .when(outcome_col.is_in(cfg.fail_types))
        .then(pl.lit(-1))
        .otherwise(pl.lit(None))
        .alias("label")
    )


def build_target_labels(
    excel_path: str | Path,
    missing_map_path: str | Path,
    output_csv: str | Path,
    *,
    sheet_name: str | None = None,
    success_types: Iterable[str] | None = None,
    fail_types: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Build target_labels.csv from the master Excel file and Missing_map.csv.

    Priority: Excel labels first, then Missing_map for any NCT IDs not in Excel.
    """
    excel_path = Path(excel_path)
    missing_map_path = Path(missing_map_path)
    output_csv = Path(output_csv)

    cfg = TargetConfig()
    if success_types is not None:
        cfg.success_types = list(success_types)
    if fail_types is not None:
        cfg.fail_types = list(fail_types)

    # Read Excel
    excel_df = _read_excel(excel_path, sheet_name=sheet_name)
    if "nctid" not in excel_df.columns or "outcome_type" not in excel_df.columns:
        raise ValueError("Excel file must include columns: nctid, outcome_type")

    excel_df = excel_df.with_columns(
        _normalize_nctid(pl.col("nctid")).alias("nctid"),
        pl.col("outcome_type").cast(pl.Utf8).alias("outcome_type"),
    )
    excel_df = excel_df.with_columns(_label_from_outcome(pl.col("outcome_type"), cfg))
    excel_df = excel_df.filter(pl.col("label").is_not_null()).select(
        ["nctid", "outcome_type", "label"]
    )

    # Read Missing_map
    missing_df = pl.read_csv(missing_map_path, infer_schema_length=0)
    if "nctid" not in missing_df.columns or "Outcome" not in missing_df.columns:
        raise ValueError("Missing_map.csv must include columns: nctid, Outcome")
    missing_df = missing_df.rename({"Outcome": "outcome_type"})
    missing_df = missing_df.with_columns(
        _normalize_nctid(pl.col("nctid")).alias("nctid"),
        pl.col("outcome_type").cast(pl.Utf8).alias("outcome_type"),
    )
    missing_df = missing_df.with_columns(_label_from_outcome(pl.col("outcome_type"), cfg))
    missing_df = missing_df.filter(pl.col("label").is_not_null()).select(
        ["nctid", "outcome_type", "label"]
    )

    # Combine, prefer Excel
    combined = pl.concat([excel_df, missing_df], how="vertical")
    combined = combined.unique(subset=["nctid"], keep="first")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output_csv)
    return combined
