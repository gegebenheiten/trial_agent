"""
Variable statistics module for CTG ML Pipeline.

Provides comprehensive variable-level statistics including:
- missing_ratio: ratio of missing values (including pseudo-missing)
- var_type: variable type classification (continuous, categorical, string)
- variance: variance for numeric columns, unique count for categorical/string
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from ctg_ml_pipeline.analysis.eda import load_timepoint_mapping

# Pseudo-missing patterns (reuse from eda.py)
PSEUDO_MISSING_PATTERNS = [
    r"^n/?a$", r"^nan$", r"^null$", r"^none$", r"^missing$",
    r"^not\s*(available|applicable|reported|specified)$",
    r"^unknown$", r"^-$", r"^\.$", r"^--$", r"^\s*$"
]
PSEUDO_MISSING_RE = re.compile("|".join(PSEUDO_MISSING_PATTERNS), re.IGNORECASE)


VarType = Literal["continuous", "categorical", "string"]


@dataclass
class VarStat:
    """Statistics for a single variable."""
    name: str
    var_type: VarType
    missing_ratio: float
    variance: float | None  # variance for continuous, None for non-numeric
    n_unique: int           # unique value count
    is_constant: bool       # True if all values are the same (or all missing)
    total_rows: int
    null_count: int
    pseudo_missing_count: int


def _is_pseudo_missing(val: str | None) -> bool:
    """Check if value is pseudo-missing."""
    if val is None:
        return True
    if isinstance(val, str):
        return bool(PSEUDO_MISSING_RE.match(val.strip()))
    return False


def _classify_var_type(
    col: pl.Series,
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
) -> VarType:
    """
    Classify variable type based on dtype and cardinality.
    
    Rules:
    - Numeric dtype (int/float) with cardinality <= threshold -> categorical
    - Numeric dtype with cardinality > threshold -> continuous
    - String dtype with cardinality <= threshold -> categorical
    - String dtype with cardinality > threshold or long text -> string
    """
    dtype = col.dtype
    non_null = col.drop_nulls()
    n_unique = non_null.n_unique() if len(non_null) > 0 else 0
    
    # Numeric types
    if dtype.is_numeric():
        # Check if it looks like a categorical encoded as int
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            if n_unique <= categorical_threshold:
                return "categorical"
        return "continuous"
    
    # String type
    if dtype == pl.Utf8:
        if n_unique == 0:
            return "string"
        
        # Check average text length
        if len(non_null) > 0:
            avg_len = non_null.str.len_chars().mean()
            if avg_len is not None and avg_len > text_length_threshold:
                return "string"
        
        # Low cardinality -> categorical
        if n_unique <= categorical_threshold:
            return "categorical"
        
        return "string"
    
    # Boolean or other types
    if dtype == pl.Boolean:
        return "categorical"
    
    # Default to string for unknown types
    return "string"


def _compute_variance(col: pl.Series, var_type: VarType) -> float | None:
    """
    Compute variance for numeric columns.
    Returns None for non-numeric types.
    """
    if var_type != "continuous":
        return None
    
    if not col.dtype.is_numeric():
        return None
    
    non_null = col.drop_nulls()
    if len(non_null) <= 1:
        return 0.0
    
    # Use numpy for accurate variance calculation
    arr = non_null.to_numpy()
    return float(np.var(arr, ddof=1))  # sample variance


def analyze_var(
    col: pl.Series,
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
) -> VarStat:
    """
    Analyze a single variable and compute statistics.
    
    Args:
        col: Polars Series to analyze
        categorical_threshold: max unique values for categorical classification
        text_length_threshold: avg text length above which string is text (not categorical)
    
    Returns:
        VarStat with all computed statistics
    """
    total_rows = len(col)
    null_count = col.null_count()
    
    # Pseudo-missing count for string columns
    pseudo_missing_count = 0
    if col.dtype == pl.Utf8:
        pseudo_missing_count = sum(
            1 for v in col.to_list() 
            if v is not None and _is_pseudo_missing(v)
        )
    
    total_missing = null_count + pseudo_missing_count
    missing_ratio = total_missing / total_rows if total_rows > 0 else 0.0
    
    # Variable type classification
    var_type = _classify_var_type(col, categorical_threshold, text_length_threshold)
    
    # Unique count (excluding nulls)
    non_null = col.drop_nulls()
    n_unique = non_null.n_unique() if len(non_null) > 0 else 0
    
    # Constant check
    is_constant = n_unique <= 1
    
    # Variance
    variance = _compute_variance(col, var_type)
    
    return VarStat(
        name=col.name,
        var_type=var_type,
        missing_ratio=missing_ratio,
        variance=variance,
        n_unique=n_unique,
        is_constant=is_constant,
        total_rows=total_rows,
        null_count=null_count,
        pseudo_missing_count=pseudo_missing_count,
    )


def analyze_vars(
    df: pl.DataFrame,
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
    exclude_cols: list[str] | None = None,
) -> list[VarStat]:
    """
    Analyze all variables in a DataFrame.
    
    Args:
        df: Input DataFrame
        categorical_threshold: max unique values for categorical
        text_length_threshold: avg text length threshold for string classification
        exclude_cols: columns to skip
    
    Returns:
        List of VarStat for each column
    """
    exclude = set(exclude_cols or [])
    results = []
    
    for col_name in df.columns:
        if col_name in exclude:
            continue
        col = df.get_column(col_name)
        stat = analyze_var(col, categorical_threshold, text_length_threshold)
        results.append(stat)
    
    return results


def var_stats_to_dataframe(stats: list[VarStat]) -> pl.DataFrame:
    """
    Convert list of VarStat to a Polars DataFrame.
    """
    rows = []
    for s in stats:
        rows.append({
            "var_name": s.name,
            "var_type": s.var_type,
            "missing_ratio": round(s.missing_ratio, 4),
            "variance": round(s.variance, 6) if s.variance is not None else None,
            "n_unique": s.n_unique,
            "is_constant": s.is_constant,
            "total_rows": s.total_rows,
            "null_count": s.null_count,
            "pseudo_missing_count": s.pseudo_missing_count,
        })
    return pl.DataFrame(rows)


def summarize_var_stats(
    df: pl.DataFrame,
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
    exclude_cols: list[str] | None = None,
    output_csv: str | Path | None = None,
) -> pl.DataFrame:
    """
    Main function: analyze all variables and return summary DataFrame.
    
    Args:
        df: Input DataFrame
        categorical_threshold: max unique values for categorical
        text_length_threshold: avg text length for string classification
        exclude_cols: columns to skip
        output_csv: if provided, export to CSV
    
    Returns:
        DataFrame with columns:
        - var_name: variable name
        - var_type: continuous/categorical/string
        - missing_ratio: missing rate (0-1)
        - variance: variance for continuous, None for others
        - n_unique: number of unique values
        - is_constant: True if all values are the same
    """
    stats = analyze_vars(df, categorical_threshold, text_length_threshold, exclude_cols)
    result_df = var_stats_to_dataframe(stats)
    
    if output_csv:
        result_df.write_csv(output_csv)
        print(f"Exported var stats to {output_csv}")
    
    return result_df


def summarize_var_stats_multi_tables(
    tables: dict[str, pl.DataFrame],
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
    exclude_cols: list[str] | None = None,
    output_csv: str | Path | None = None,
) -> pl.DataFrame:
    """
    Analyze variables across multiple tables.
    
    Args:
        tables: dict mapping table_name -> DataFrame
        categorical_threshold: max unique values for categorical
        text_length_threshold: avg text length for string classification
        exclude_cols: columns to skip
        output_csv: if provided, export to CSV
    
    Returns:
        DataFrame with table_name column + var stats
    """
    all_rows = []
    
    for table_name, df in tables.items():
        stats = analyze_vars(df, categorical_threshold, text_length_threshold, exclude_cols)
        for s in stats:
            all_rows.append({
                "table_name": table_name,
                "var_name": s.name,
                "var_type": s.var_type,
                "missing_ratio": round(s.missing_ratio, 4),
                "variance": round(s.variance, 6) if s.variance is not None else None,
                "n_unique": s.n_unique,
                "is_constant": s.is_constant,
                "total_rows": s.total_rows,
                "null_count": s.null_count,
                "pseudo_missing_count": s.pseudo_missing_count,
            })
    
    result_df = pl.DataFrame(all_rows)
    
    if output_csv:
        result_df.write_csv(output_csv)
        print(f"Exported multi-table var stats to {output_csv}")
    
    return result_df


def build_feature_allowlists(
    tables_dir: str | Path,
    timepoint_excel: str | Path | None,
    output_dir: str | Path,
    categorical_threshold: int = 20,
    text_length_threshold: int = 100,
    exclude_cols: list[str] | None = None,
    exclude_tables: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build feature allowlists with timepoint + feature type.

    Outputs:
      - feature_allowlist.csv (all timepoints)
      - feature_allowlist_T0.csv (Availability_Timepoint == "T0")
    """
    tables_path = Path(tables_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pl.DataFrame] = {}
    for csv_file in tables_path.glob("*.csv"):
        tables[csv_file.stem] = pl.read_csv(csv_file)

    if not tables:
        raise ValueError(f"No CSV files found in {tables_path}")

    stats_df = summarize_var_stats_multi_tables(
        tables,
        categorical_threshold=categorical_threshold,
        text_length_threshold=text_length_threshold,
        exclude_cols=exclude_cols,
    )

    timepoint_map: dict[str, str] = {}
    if timepoint_excel:
        timepoint_map = load_timepoint_mapping(timepoint_excel)

    skip_tables = set(exclude_tables or [])
    rows = []
    for row in stats_df.iter_rows(named=True):
        table_name = str(row.get("table_name", "")).strip()
        var_name = str(row.get("var_name", "")).strip()
        if not table_name or not var_name:
            continue
        if table_name in skip_tables:
            continue
        if var_name == "StudyID":
            continue

        short_table = table_name.replace("_all", "")
        var_type = row.get("var_type", "")
        feature_type = "text" if var_type == "string" else var_type

        rows.append({
            "table": short_table,
            "Variable": var_name,
            "Availability_Timepoint": timepoint_map.get(var_name, ""),
            "FeatureType": feature_type,
        })

    allowlist_df = pl.DataFrame(rows).sort(["table", "Variable"])
    allowlist_path = out_dir / "feature_allowlist.csv"
    allowlist_df.write_csv(allowlist_path)
    print(f"Wrote allowlist: {allowlist_path}")

    allowlist_t0_df = allowlist_df.filter(pl.col("Availability_Timepoint") == "T0")
    allowlist_t0_path = out_dir / "feature_allowlist_T0.csv"
    allowlist_t0_df.write_csv(allowlist_t0_path)
    print(f"Wrote T0 allowlist: {allowlist_t0_path}")

    return allowlist_df, allowlist_t0_df


def filter_problematic_vars(
    stats_df: pl.DataFrame,
    max_missing_ratio: float = 0.5,
    min_variance: float = 1e-10,
) -> dict[str, pl.DataFrame]:
    """
    Filter and identify problematic variables.
    
    Returns dict with:
    - high_missing: variables with missing_ratio > max_missing_ratio
    - constant: variables where is_constant=True
    - zero_variance: continuous variables with variance < min_variance
    """
    # High missing
    high_missing = stats_df.filter(pl.col("missing_ratio") > max_missing_ratio)
    
    # Constant columns
    constant = stats_df.filter(pl.col("is_constant") == True)
    
    # Zero variance (for continuous only)
    zero_variance = stats_df.filter(
        (pl.col("var_type") == "continuous") & 
        (pl.col("variance").is_not_null()) &
        (pl.col("variance") < min_variance)
    )
    
    return {
        "high_missing": high_missing,
        "constant": constant,
        "zero_variance": zero_variance,
    }


def print_var_stats_summary(stats_df: pl.DataFrame) -> None:
    """
    Print a human-readable summary of variable statistics.
    """
    total = len(stats_df)
    
    # Count by var_type
    type_counts = stats_df.group_by("var_type").len()
    
    # Problematic vars
    problems = filter_problematic_vars(stats_df)
    
    print("=" * 70)
    print("Variable Statistics Summary")
    print("=" * 70)
    print(f"\nTotal variables: {total}")
    print("\nBy type:")
    for row in type_counts.iter_rows(named=True):
        print(f"  {row['var_type']:<12}: {row['len']:>4}")
    
    print("\n" + "-" * 70)
    print("Problematic Variables")
    print("-" * 70)
    print(f"  High missing (>50%): {len(problems['high_missing'])}")
    print(f"  Constant:            {len(problems['constant'])}")
    print(f"  Zero variance:       {len(problems['zero_variance'])}")
    
    # Show top missing
    if len(problems['high_missing']) > 0:
        print("\nTop 10 high-missing variables:")
        top_missing = problems['high_missing'].sort("missing_ratio", descending=True).head(10)
        for row in top_missing.iter_rows(named=True):
            name = row.get("table_name", "") + "." + row["var_name"] if "table_name" in row else row["var_name"]
            print(f"    {name[:40]:<40} {row['missing_ratio']:.1%}")
    
    # Show constant vars
    if len(problems['constant']) > 0:
        print("\nConstant variables (should be excluded):")
        for row in problems['constant'].iter_rows(named=True):
            name = row.get("table_name", "") + "." + row["var_name"] if "table_name" in row else row["var_name"]
            print(f"    {name[:50]}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Variable statistics + feature allowlist builder")
    parser.add_argument("input_path", help="CSV file or directory of CSVs")
    parser.add_argument("output_csv", nargs="?", default="", help="Optional var-stats CSV output")
    parser.add_argument("--timepoint-excel", default="", help="Excel with Availability_Timepoint mapping")
    parser.add_argument("--allowlist-out", default="", help="Output feature_allowlist.csv path")
    parser.add_argument("--allowlist-t0-out", default="", help="Output feature_allowlist_T0.csv path")
    parser.add_argument("--categorical-threshold", type=int, default=20)
    parser.add_argument("--text-length-threshold", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_csv = args.output_csv or None

    if input_path.is_dir():
        # Load all CSVs in directory
        tables = {}
        for csv_file in input_path.glob("*.csv"):
            tables[csv_file.stem] = pl.read_csv(csv_file)

        if not tables:
            print(f"No CSV files found in {input_path}")
            raise SystemExit(1)

        print(f"Loaded {len(tables)} tables from {input_path}")
        stats_df = summarize_var_stats_multi_tables(
            tables,
            categorical_threshold=args.categorical_threshold,
            text_length_threshold=args.text_length_threshold,
            output_csv=output_csv,
            exclude_cols=["StudyID"],
        )

        # Build allowlists when analyzing tables dir
        out_dir = input_path.parent
        allowlist_out = Path(args.allowlist_out) if args.allowlist_out else out_dir / "feature_allowlist.csv"
        allowlist_t0_out = Path(args.allowlist_t0_out) if args.allowlist_t0_out else out_dir / "feature_allowlist_T0.csv"
        timepoint_excel = Path(args.timepoint_excel) if args.timepoint_excel else None

        allow_df, allow_t0_df = build_feature_allowlists(
            tables_dir=input_path,
            timepoint_excel=timepoint_excel,
            output_dir=allowlist_out.parent,
            categorical_threshold=args.categorical_threshold,
            text_length_threshold=args.text_length_threshold,
            exclude_cols=["StudyID"],
            exclude_tables=["D_Drug_all"],
        )

        # If custom filenames requested, move outputs
        if allowlist_out.name != "feature_allowlist.csv":
            allow_df.write_csv(allowlist_out)
        if allowlist_t0_out.name != "feature_allowlist_T0.csv":
            allow_t0_df.write_csv(allowlist_t0_out)
    else:
        # Single CSV
        df = pl.read_csv(input_path)
        print(f"Loaded {len(df)} rows from {input_path}")
        stats_df = summarize_var_stats(
            df,
            categorical_threshold=args.categorical_threshold,
            text_length_threshold=args.text_length_threshold,
            output_csv=output_csv,
            exclude_cols=["StudyID"],
        )

    print_var_stats_summary(stats_df)
