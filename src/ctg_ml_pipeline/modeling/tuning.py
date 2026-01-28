from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Literal

import numpy as np

from ctg_ml_pipeline.modeling.modeling import _build_cv, _require_sklearn


def _require_optuna() -> None:
    if importlib.util.find_spec("optuna") is None:
        raise RuntimeError("optuna is required for tuning. Install: pip install optuna")


@dataclass
class TuneResult:
    model_name: str
    best_score: float
    best_params: dict[str, object]
    cv_scoring: str
    n_trials: int


def _build_model_from_params(model_name: str, params: dict[str, object], random_state: int) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC

    if model_name == "logistic":
        return LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            C=float(params["C"]),
            random_state=random_state,
        )
    if model_name == "lasso":
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            C=float(params["C"]),
            random_state=random_state,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
        )
    if model_name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=random_state,
        )
    if model_name == "svm":
        return SVC(
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            kernel="rbf",
            probability=True,
            random_state=random_state,
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=random_state,
            eval_metric="logloss",
        )
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _suggest_params(trial, model_name: str) -> dict[str, object]:
    if model_name in ("logistic", "lasso"):
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", "auto"]),
        }
    if model_name == "gbdt":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    dataset,
    model_name: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "loo"] = "kfold",
    cv_scoring: str | None = None,
    random_state: int = 42,
    timeout: int | None = None,
) -> TuneResult:
    _require_sklearn()
    _require_optuna()
    import optuna
    from sklearn.model_selection import cross_val_score

    X = dataset.X
    y = dataset.y

    cv = _build_cv(np.asarray(y), cv_folds, cv_strategy, random_state)
    if cv is None:
        raise RuntimeError("CV splitter is None (insufficient samples or single class).")

    if cv_scoring is None:
        cv_scoring = "accuracy" if cv_strategy == "loo" else "roc_auc"

    def objective(trial):
        params = _suggest_params(trial, model_name)
        model = _build_model_from_params(model_name, params, random_state)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=cv_scoring,
            error_score=np.nan,
        )
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return TuneResult(
        model_name=model_name,
        best_score=float(study.best_value),
        best_params=study.best_params,
        cv_scoring=cv_scoring,
        n_trials=n_trials,
    )


def tune_models(
    dataset,
    models: list[str],
    n_trials: int = 50,
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "loo"] = "kfold",
    cv_scoring: str | None = None,
    random_state: int = 42,
    timeout: int | None = None,
) -> dict[str, TuneResult]:
    results: dict[str, TuneResult] = {}
    for model_name in models:
        results[model_name] = tune_model(
            dataset,
            model_name=model_name,
            n_trials=n_trials,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            cv_scoring=cv_scoring,
            random_state=random_state,
            timeout=timeout,
        )
    return results
