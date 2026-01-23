from __future__ import annotations

from typing import Iterable

import polars as pl


def _mode_value(df: pl.DataFrame, col: str):
    series = df.select(pl.col(col).drop_nulls().mode()).to_series()
    if series.is_empty():
        return None
    value = series[0]
    return value


def impute_simple(
    df: pl.DataFrame,
    exclude: Iterable[str] = (),
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pl.DataFrame:
    exclude = set(exclude)
    out = df.clone()
    for name, dtype in zip(out.columns, out.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            if numeric_strategy == "median":
                value = out.select(pl.col(name).median()).item()
            elif numeric_strategy == "mean":
                value = out.select(pl.col(name).mean()).item()
            else:
                raise ValueError("numeric_strategy must be 'median' or 'mean'")
            if value is not None:
                out = out.with_columns(pl.col(name).fill_null(value).alias(name))
        else:
            if categorical_strategy != "mode":
                raise ValueError("categorical_strategy must be 'mode'")
            value = _mode_value(out, name)
            if value is not None:
                out = out.with_columns(pl.col(name).fill_null(value).alias(name))
    return out
