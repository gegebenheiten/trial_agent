from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class EvalResult:
    metrics: dict[str, float]


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "scikit-learn is required for evaluation. Install it first: pip install scikit-learn"
        )


def evaluate_classification(y_true, y_pred, y_proba=None) -> EvalResult:
    _require_sklearn()
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
    return EvalResult(metrics=metrics)
