from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterable

import polars as pl


@dataclass
class SelectionResult:
    selected_features: list[str]
    scores: dict[str, float]


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "scikit-learn is required for feature selection. Install it first: pip install scikit-learn"
        )


def _prepare_features(df: pl.DataFrame, target: str) -> tuple[pl.DataFrame, pl.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    y = df.get_column(target)
    X = df.drop(target)
    # One-hot encode categoricals so we can score everything numerically.
    cat_cols = [name for name, dtype in zip(X.columns, X.dtypes) if dtype == pl.Utf8]
    if cat_cols:
        X = X.to_dummies(columns=cat_cols)
    return X, y


def filter_stage(
    df: pl.DataFrame,
    target: str,
    method: str = "mutual_info",
    top_ratio: float = 0.2,
) -> SelectionResult:
    _require_sklearn()
    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

    X_df, y = _prepare_features(df, target)
    X = X_df.to_numpy()
    y_arr = np.asarray(y)

    if method == "fisher":
        scores, _ = f_classif(X, y_arr)
    elif method == "anova":
        scores, _ = f_classif(X, y_arr)
    elif method == "mutual_info":
        scores = mutual_info_classif(X, y_arr, discrete_features="auto")
    elif method == "chi2":
        scores, _ = chi2(X, y_arr)
    else:
        raise ValueError("method must be one of: fisher, anova, mutual_info, chi2")

    features = X_df.columns
    score_map = {name: float(score) for name, score in zip(features, scores)}
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)

    top_k = max(1, int(len(ranked) * top_ratio))
    selected = [name for name, _ in ranked[:top_k]]
    return SelectionResult(selected_features=selected, scores=score_map)


def embedded_stage(
    df: pl.DataFrame,
    target: str,
    method: str = "l1",
    top_ratio: float = 0.2,
) -> SelectionResult:
    _require_sklearn()
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    X_df, y = _prepare_features(df, target)
    X = X_df.to_numpy()
    y_arr = np.asarray(y)

    if method == "l1":
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
        model.fit(X, y_arr)
        scores = np.abs(model.coef_).sum(axis=0)
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X, y_arr)
        scores = model.feature_importances_
    elif method == "gbdt":
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X, y_arr)
        scores = model.feature_importances_
    else:
        raise ValueError("method must be one of: l1, rf, gbdt")

    features = X_df.columns
    score_map = {name: float(score) for name, score in zip(features, scores)}
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    top_k = max(1, int(len(ranked) * top_ratio))
    selected = [name for name, _ in ranked[:top_k]]
    return SelectionResult(selected_features=selected, scores=score_map)
