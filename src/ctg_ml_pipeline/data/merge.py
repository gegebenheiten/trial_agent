from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from ctg_ml_pipeline.config import ALL_TABLES, DEFAULT_TABLES


@dataclass
class MergeResult:
    data: pl.DataFrame
    included: list[str]
    skipped: list[str]


@dataclass
class BackfillResult:
    table: str
    columns: list[str]
    updated: list[str]
    already_present: list[str]
    missing_notebooklm: list[str]
    missing_base: list[str]
    needs_reextract: list[str]
    still_missing: list[str]


def _read_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0)
    str_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    if str_cols:
        df = df.with_columns(
            [
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
                for col in str_cols
            ]
        )
    return df


def _reduce_multirow(df: pl.DataFrame, key: str = "StudyID") -> pl.DataFrame:
    if df.is_empty():
        return df
    agg_exprs = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name == key:
            continue
        if dtype.is_numeric():
            agg_exprs.append(pl.col(name).mean().alias(name))
        else:
            agg_exprs.append(pl.col(name).drop_nulls().first().alias(name))
    if not agg_exprs:
        return df.unique(subset=[key])
    return df.group_by(key).agg(agg_exprs)


def _load_table(nct_dir: Path, table: str, prefer_notebooklm: bool = True) -> pl.DataFrame | None:
    if prefer_notebooklm:
        csv_path = nct_dir / "notebooklm" / f"{table}_notebooklm.csv"
        if csv_path.exists():
            return _read_csv(csv_path)
    csv_path = nct_dir / f"{table}.csv"
    if csv_path.exists():
        return _read_csv(csv_path)
    return None


def _table_csv_path(nct_dir: Path, table: str, source: str = "auto") -> Path | None:
    if source == "notebooklm":
        csv_path = nct_dir / "notebooklm" / f"{table}_notebooklm.csv"
        return csv_path if csv_path.exists() else None
    if source == "base":
        csv_path = nct_dir / f"{table}.csv"
        return csv_path if csv_path.exists() else None
    if source == "auto":
        csv_path = nct_dir / "notebooklm" / f"{table}_notebooklm.csv"
        if csv_path.exists():
            return csv_path
        csv_path = nct_dir / f"{table}.csv"
        return csv_path if csv_path.exists() else None
    raise ValueError("source must be 'auto', 'notebooklm', or 'base'")


def _merge_single_nct(
    nct_dir: Path,
    tables: Iterable[str],
    mode: str = "study",
    prefer_notebooklm: bool = True,
) -> pl.DataFrame | None:
    tables = tuple(tables)
    required = {"D_Design", "D_Pop", "D_Drug", "R_Study", "R_Arm_Study"}
    if not required.issubset(set(tables)):
        raise ValueError("tables must include D_Design, D_Pop, D_Drug, R_Study, R_Arm_Study")

    d_design = _load_table(nct_dir, "D_Design", prefer_notebooklm)
    d_pop = _load_table(nct_dir, "D_Pop", prefer_notebooklm)
    d_drug = _load_table(nct_dir, "D_Drug", prefer_notebooklm)
    r_study = _load_table(nct_dir, "R_Study", prefer_notebooklm)
    r_arm = _load_table(nct_dir, "R_Arm_Study", prefer_notebooklm)

    if any(df is None for df in (d_design, d_pop, d_drug, r_study, r_arm)):
        return None

    # Supplement missing columns from base CSV if needed (e.g., Group_Type)
    if "R_Arm_Study" in SUPPLEMENT_FROM_BASE and r_arm is not None:
        r_arm = _supplement_columns_from_base(
            r_arm, nct_dir, "R_Arm_Study", SUPPLEMENT_FROM_BASE["R_Arm_Study"]
        )

    d_drug = _reduce_multirow(d_drug)
    r_arm_reduced = _reduce_multirow(r_arm)

    if mode == "study":
        base = d_design
        base = base.join(d_pop, on="StudyID", how="left", suffix="_pop")
        base = base.join(d_drug, on="StudyID", how="left", suffix="_drug")
        base = base.join(r_study, on="StudyID", how="left", suffix="_study")
        base = base.join(r_arm_reduced, on="StudyID", how="left", suffix="_arm")
        return base

    if mode == "arm":
        base = r_arm
        base = base.join(d_design, on="StudyID", how="left", suffix="_design")
        base = base.join(d_pop, on="StudyID", how="left", suffix="_pop")
        base = base.join(d_drug, on="StudyID", how="left", suffix="_drug")
        base = base.join(r_study, on="StudyID", how="left", suffix="_study")
        return base

    raise ValueError("mode must be 'study' or 'arm'")


def merge_group_tables(
    group_dir: str | Path,
    output_csv: str | Path | None = None,
    tables: Iterable[str] = DEFAULT_TABLES,
    mode: str = "study",
    prefer_notebooklm: bool = True,
    strict: bool = False,
) -> MergeResult:
    group_path = Path(group_dir)
    rows = []
    included = []
    skipped = []

    for nct_dir in sorted(group_path.glob("NCT*")):
        merged = _merge_single_nct(nct_dir, tables, mode=mode, prefer_notebooklm=prefer_notebooklm)
        if merged is None:
            skipped.append(nct_dir.name)
            if strict:
                continue
            else:
                continue
        rows.append(merged)
        included.append(nct_dir.name)

    if rows:
        # Use diagonal concat to tolerate schema drift across trials.
        merged_df = pl.concat(rows, how="diagonal")
    else:
        merged_df = pl.DataFrame()

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.write_csv(output_path)

    return MergeResult(data=merged_df, included=included, skipped=skipped)


# Tables that do not have NotebookLM versions (only base CSV)
BASE_ONLY_TABLES = {"R_Study_Endpoint", "R_Arm_Study_Endpoint"}

# Tables that need columns supplemented from base CSV
# Format: {table_name: [columns_to_supplement]}
SUPPLEMENT_FROM_BASE = {
    "R_Arm_Study": ["Group_Type"],
}


def _get_complete_trials(group_path: Path, notebooklm_tables: Iterable[str]) -> set[str]:
    """Get trial IDs that have complete notebooklm extraction for all specified tables."""
    notebooklm_tables = set(notebooklm_tables) - BASE_ONLY_TABLES
    complete = set()
    for nct_dir in group_path.glob("NCT*"):
        nb_dir = nct_dir / "notebooklm"
        if not nb_dir.exists():
            continue
        has_all = all(
            (nb_dir / f"{table}_notebooklm.csv").exists()
            for table in notebooklm_tables
        )
        if has_all:
            complete.add(nct_dir.name)
    return complete


def _supplement_columns_from_base(
    df: pl.DataFrame,
    nct_dir: Path,
    table: str,
    columns: list[str],
) -> pl.DataFrame:
    """Supplement missing columns from base CSV using join on common keys."""
    base_path = nct_dir / f"{table}.csv"
    if not base_path.exists():
        return df

    base_df = _read_csv(base_path)
    # Find columns that exist in base but not in df
    missing_cols = [c for c in columns if c in base_df.columns and c not in df.columns]
    if not missing_cols:
        return df

    # Find common key columns for joining (StudyID, Arm_ID, etc.)
    possible_keys = ["StudyID", "Arm_ID", "Drug_Name"]
    join_keys = [k for k in possible_keys if k in df.columns and k in base_df.columns]

    if not join_keys:
        # No common keys, just add columns with nulls
        for col in missing_cols:
            df = df.with_columns(pl.lit(None).alias(col))
        return df

    # Select only the join keys and missing columns from base
    base_subset = base_df.select(join_keys + missing_cols)
    # Join to supplement missing columns
    df = df.join(base_subset, on=join_keys, how="left")
    return df


def backfill_notebooklm_columns(
    group_dir: str | Path,
    table: str,
    columns: Iterable[str],
    nctids: Iterable[str] | None = None,
    require_base_value: bool = True,
    treat_empty_as_missing: bool = False,
    dry_run: bool = False,
) -> BackfillResult:
    """Backfill missing columns in NotebookLM CSVs using base CSV."""
    group_path = Path(group_dir)
    columns = list(columns)
    updated: list[str] = []
    already_present: list[str] = []
    missing_notebooklm: list[str] = []
    missing_base: list[str] = []
    needs_reextract: list[str] = []
    still_missing: list[str] = []

    nctid_set = {str(n).upper() for n in nctids} if nctids else None
    for nct_dir in sorted(group_path.glob("NCT*")):
        if nctid_set is not None and nct_dir.name.upper() not in nctid_set:
            continue
        nb_path = nct_dir / "notebooklm" / f"{table}_notebooklm.csv"
        base_path = nct_dir / f"{table}.csv"

        if not nb_path.exists():
            missing_notebooklm.append(nct_dir.name)
            continue
        if not base_path.exists():
            missing_base.append(nct_dir.name)
            continue

        df = _read_csv(nb_path)
        has_all_cols = all(col in df.columns for col in columns)
        if has_all_cols and treat_empty_as_missing:
            has_values = True
            for col in columns:
                series = df.get_column(col)
                if series.dtype != pl.Utf8:
                    series = series.cast(pl.Utf8)
                non_empty = series.drop_nulls().str.strip_chars()
                if (non_empty != "").sum() == 0:
                    has_values = False
                    break
            if has_values:
                already_present.append(nct_dir.name)
                continue
        elif has_all_cols:
            already_present.append(nct_dir.name)
            continue

        supplemented = _supplement_columns_from_base(df, nct_dir, table, columns)
        if not all(col in supplemented.columns for col in columns):
            still_missing.append(nct_dir.name)
            continue

        if require_base_value:
            base_has_value = True
            for col in columns:
                series = supplemented.get_column(col)
                if series.dtype != pl.Utf8:
                    series = series.cast(pl.Utf8)
                non_empty = series.drop_nulls().str.strip_chars()
                if (non_empty != "").sum() == 0:
                    base_has_value = False
                    break
            if not base_has_value:
                needs_reextract.append(nct_dir.name)
                continue

        if not dry_run:
            supplemented.write_csv(nb_path)
        updated.append(nct_dir.name)

    return BackfillResult(
        table=table,
        columns=columns,
        updated=updated,
        already_present=already_present,
        missing_notebooklm=missing_notebooklm,
        missing_base=missing_base,
        needs_reextract=needs_reextract,
        still_missing=still_missing,
    )


def merge_group_tables_by_table(
    group_dir: str | Path,
    output_dir: str | Path | None = None,
    tables: Iterable[str] = ALL_TABLES,
    source: str = "auto",
    ensure_study_id: bool = True,
    consistent: bool = False,
) -> dict[str, MergeResult]:
    """Merge all trials' data for each table separately.

    Args:
        group_dir: Root directory containing NCT* subdirectories.
        output_dir: Optional directory to save merged CSVs.
        tables: Table names to merge.
        source: "auto", "notebooklm", or "base". For tables in BASE_ONLY_TABLES,
            "notebooklm" is automatically downgraded to "auto".
        ensure_study_id: If True, add StudyID column if missing.
        consistent: If True, only merge trials that have complete notebooklm
            extraction for all non-BASE_ONLY_TABLES. This ensures all 7 tables
            contain the same set of trials.
    """
    group_path = Path(group_dir)
    tables = list(tables)
    results: dict[str, MergeResult] = {}

    # If consistent mode, get complete trials first
    allowed_trials: set[str] | None = None
    if consistent:
        allowed_trials = _get_complete_trials(group_path, tables)

    for table in tables:
        frames = []
        included: list[str] = []
        skipped: list[str] = []

        # For tables without NotebookLM versions, use auto source instead
        effective_source = source
        if table in BASE_ONLY_TABLES and source == "notebooklm":
            effective_source = "auto"

        for nct_dir in sorted(group_path.glob("NCT*")):
            # Skip trials not in allowed set (consistent mode)
            if allowed_trials is not None and nct_dir.name not in allowed_trials:
                skipped.append(nct_dir.name)
                continue

            csv_path = _table_csv_path(nct_dir, table, source=effective_source)
            if csv_path is None:
                skipped.append(nct_dir.name)
                continue
            df = _read_csv(csv_path)

            # Supplement columns from base CSV if needed
            if table in SUPPLEMENT_FROM_BASE:
                df = _supplement_columns_from_base(
                    df, nct_dir, table, SUPPLEMENT_FROM_BASE[table]
                )

            if ensure_study_id and "StudyID" not in df.columns:
                df = df.with_columns(pl.lit(nct_dir.name).alias("StudyID"))
            frames.append(df)
            included.append(nct_dir.name)

        # Use how="diagonal" to handle different column schemas across trials
        merged = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            merged.write_csv(output_path / f"{table}_all.csv")
        results[table] = MergeResult(data=merged, included=included, skipped=skipped)

    return results
