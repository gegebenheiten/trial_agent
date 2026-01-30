from __future__ import annotations

from typing import Iterable

import polars as pl


UNKNOWN_TOKEN = "Unknown"


def fill_unknowns(
    df: pl.DataFrame,
    columns: Iterable[str] | None = None,
    token: str = UNKNOWN_TOKEN,
    empty_to_unknown: bool = True,
) -> pl.DataFrame:
    """Fill null (and optionally empty) string values with a default token."""
    if columns is None:
        columns = [
            col
            for col in df.columns
            if df.get_column(col).dtype == pl.Utf8
        ]
    cols = [c for c in columns if c in df.columns and df.get_column(c).dtype == pl.Utf8]
    if not cols:
        return df

    exprs: list[pl.Expr] = []
    for col in cols:
        if empty_to_unknown:
            expr = pl.when(
                pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")
            ).then(pl.lit(token)).otherwise(pl.col(col))
        else:
            expr = pl.col(col).fill_null(token)
        exprs.append(expr.alias(col))
    return df.with_columns(exprs)


def normalize_subgroup_ana(
    df: pl.DataFrame,
    column: str = "Subgroup_Ana",
    token_unknown: str = UNKNOWN_TOKEN,
) -> pl.DataFrame:
    """Normalize Subgroup_Ana values to Yes/No/Unknown."""
    if column not in df.columns:
        return df
    if df.get_column(column).dtype != pl.Utf8:
        return df

    col = pl.col(column)
    lowered = col.str.to_lowercase().str.strip_chars()
    expr = (
        pl.when(col.is_null() | (lowered == ""))
        .then(pl.lit(token_unknown))
        .when(lowered == "unknown")
        .then(pl.lit("Unknown"))
        .when(lowered.str.starts_with("yes"))
        .then(pl.lit("Yes"))
        .when(lowered.str.starts_with("no"))
        .then(pl.lit("No"))
        .otherwise(pl.lit(token_unknown))
    )
    return df.with_columns(expr.alias(column))
