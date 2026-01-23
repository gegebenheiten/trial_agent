from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterable

import polars as pl


@dataclass
class ModelResult:
    model_name: str
    metrics: dict[str, float]
    model: object


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "scikit-learn is required for modeling. Install it first: pip install scikit-learn"
        )


def _prepare(df: pl.DataFrame, target: str) -> tuple[pl.DataFrame, pl.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    y = df.get_column(target)
    X = df.drop(target)
    cat_cols = [name for name, dtype in zip(X.columns, X.dtypes) if dtype == pl.Utf8]
    if cat_cols:
        X = X.to_dummies(columns=cat_cols)
    return X, y


def train_classifier(
    df: pl.DataFrame,
    target: str,
    model_name: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    _require_sklearn()
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    X_df, y = _prepare(df, target)
    X = X_df.to_numpy()
    y_arr = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError("model_name must be one of: logistic, rf, gbdt")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
    return ModelResult(model_name=model_name, metrics=metrics, model=model)
