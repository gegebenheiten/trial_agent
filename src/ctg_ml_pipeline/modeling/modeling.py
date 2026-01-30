"""
Modeling module for clinical trial outcome prediction.

This module provides functions for training and evaluating ML models
using the TrialDataset class.

Features:
- Confusion matrix with Precision/Recall/Specificity
- Balanced Accuracy / MCC (robust for imbalanced data)
- PR-AUC (sensitive when positive class is rare)
- Threshold-metric curves (P/R/F1 at different thresholds)
- Bootstrap confidence intervals for key metrics

Usage:
    from ctg_ml_pipeline.data.dataset import load_trial_dataset
    from ctg_ml_pipeline.modeling.modeling import train_and_evaluate, compare_models
    
    dataset = load_trial_dataset(group_dir, target_csv)
    result = train_and_evaluate(dataset, model_name="rf")
    result.print_detailed_metrics()
"""

from __future__ import annotations

import json

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import re

import numpy as np
import polars as pl

from ctg_ml_pipeline.data.dataset import FeatureType

if TYPE_CHECKING:
    from ctg_ml_pipeline.data.dataset import TrialDataset


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "na"


def _build_auto_output_dir(
    *,
    base: str,
    models: list[str],
    cv_strategy: str,
    cv_folds: int,
    no_cv: bool,
    time_split: bool,
    max_missing_rate: float,
    select_stage: str,
    select_method: str,
    select_top_ratio: float,
    text_as_bool: bool,
    phase_filter: list[str],
    tuned: bool,
) -> str:
    models_tag = "all" if models else "none"
    if models and len(models) > 1:
        models_tag = "all" if set(models) == set(available_models()) else "-".join(models)
    phase_tag = "all" if not phase_filter else "phase" + "-".join(phase_filter)
    cv_tag = "nocv" if no_cv else f"{cv_strategy}{cv_folds}"
    split_tag = "time" if time_split else "random"
    mode_tag = split_tag if no_cv else cv_tag
    miss_tag = f"miss{max_missing_rate:g}"
    sel_tag = f"{select_stage}-{select_method}-{select_top_ratio:g}"
    text_tag = "textbool" if text_as_bool else "notext"
    tune_tag = "tune" if tuned else "notune"
    parts = [
        "exp",
        phase_tag,
        mode_tag,
        miss_tag,
        sel_tag,
        text_tag,
        tune_tag,
        models_tag,
    ]
    name = "_".join(_sanitize_token(p) for p in parts if p)
    return str(Path(base) / name)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfusionMatrixMetrics:
    """Metrics derived from confusion matrix."""
    
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """TP / (TP + FN) - also called Sensitivity or TPR"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """TN / (TN + FP) - also called TNR"""
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def npv(self) -> float:
        """Negative Predictive Value: TN / (TN + FN)"""
        return self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0.0
    
    def as_matrix(self) -> np.ndarray:
        """Return as 2x2 matrix."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""
    
    mean: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


@dataclass
class ThresholdMetrics:
    """Metrics at different thresholds."""
    
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    precisions: np.ndarray = field(default_factory=lambda: np.array([]))
    recalls: np.ndarray = field(default_factory=lambda: np.array([]))
    f1_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def best_f1_threshold(self) -> float:
        """Threshold that maximizes F1."""
        if len(self.f1_scores) == 0:
            return 0.5
        idx = np.argmax(self.f1_scores)
        return float(self.thresholds[idx])
    
    @property
    def best_f1(self) -> float:
        """Maximum F1 score."""
        if len(self.f1_scores) == 0:
            return 0.0
        return float(np.max(self.f1_scores))


@dataclass
class ModelResult:
    """Result from training a single model."""
    
    model_name: str
    model: object
    
    # Basic metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    
    # Confusion matrix based metrics
    confusion_matrix: ConfusionMatrixMetrics = field(default_factory=ConfusionMatrixMetrics)
    
    # Robust metrics for imbalanced data
    balanced_accuracy: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    
    # AUC metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0  # Precision-Recall AUC
    
    # Cross-validation
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    cv_scoring: str = "roc_auc"
    
    # Bootstrap confidence intervals
    bootstrap_roc_auc: BootstrapCI = field(default_factory=BootstrapCI)
    bootstrap_f1: BootstrapCI = field(default_factory=BootstrapCI)
    bootstrap_recall: BootstrapCI = field(default_factory=BootstrapCI)
    
    # Threshold analysis
    threshold_metrics: ThresholdMetrics = field(default_factory=ThresholdMetrics)
    
    # Feature importance
    feature_importances: dict[str, float] = field(default_factory=dict)
    
    # Raw predictions (for further analysis)
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    y_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    test_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def print_detailed_metrics(self) -> None:
        """Print detailed evaluation metrics."""
        print("\n" + "=" * 70)
        print(f"Detailed Metrics: {self.model_name}")
        print("=" * 70)
        
        # Confusion Matrix
        cm = self.confusion_matrix
        print("\n[1] Confusion Matrix")
        print("-" * 40)
        print(f"                 Predicted")
        print(f"              Neg      Pos")
        print(f"Actual Neg   {cm.tn:4d}     {cm.fp:4d}")
        print(f"Actual Pos   {cm.fn:4d}     {cm.tp:4d}")
        
        # Precision/Recall/Specificity
        print("\n[2] Precision / Recall / Specificity")
        print("-" * 40)
        print(f"  Precision:    {cm.precision:.3f}  (TP / Predicted Pos)")
        print(f"  Recall:       {cm.recall:.3f}  (TP / Actual Pos) = Sensitivity")
        print(f"  Specificity:  {cm.specificity:.3f}  (TN / Actual Neg)")
        print(f"  NPV:          {cm.npv:.3f}  (TN / Predicted Neg)")
        print(f"  F1 Score:     {cm.f1:.3f}")
        
        # Balanced Accuracy / MCC
        print("\n[3] Balanced Accuracy / MCC (Imbalance-robust)")
        print("-" * 40)
        print(f"  Balanced Acc: {self.balanced_accuracy:.3f}  (mean of Recall & Specificity)")
        print(f"  MCC:          {self.mcc:.3f}  (Matthews Correlation Coef, [-1,1])")
        
        # AUC metrics
        print("\n[4] AUC Metrics")
        print("-" * 40)
        print(f"  ROC-AUC:      {self.roc_auc:.3f}")
        print(f"  PR-AUC:       {self.pr_auc:.3f}  (more sensitive for rare positive)")
        print(f"  CV {self.cv_scoring}:   {self.cv_auc_mean:.3f} ± {self.cv_auc_std:.3f}")
        
        # Bootstrap CI
        print("\n[5] Bootstrap 95% Confidence Intervals")
        print("-" * 40)
        print(f"  ROC-AUC:  {self.bootstrap_roc_auc}")
        print(f"  F1:       {self.bootstrap_f1}")
        print(f"  Recall:   {self.bootstrap_recall}")
        
        # Threshold analysis
        tm = self.threshold_metrics
        print("\n[6] Threshold Analysis")
        print("-" * 40)
        print(f"  Default threshold (0.5):")
        print(f"    Precision: {cm.precision:.3f}, Recall: {cm.recall:.3f}, F1: {cm.f1:.3f}")
        print(f"  Best F1 threshold: {tm.best_f1_threshold:.2f}")
        print(f"    Best F1: {tm.best_f1:.3f}")
        
        print("\n" + "=" * 70)


@dataclass
class ComparisonResult:
    """Result from comparing multiple models."""
    
    results: list[ModelResult]
    best_model: str
    best_auc: float
    
    def summary(self) -> None:
        """Print comparison summary."""
        print("\n" + "=" * 90)
        print("Model Comparison Results")
        print("=" * 90)
        
        header = (
            f"{'Model':<12} {'ROC-AUC':<10} {'PR-AUC':<10} "
            f"{'Bal.Acc':<10} {'MCC':<8} {'Prec':<8} {'Recall':<8} {'Spec':<8} {'F1':<8}"
        )
        print(header)
        print("-" * 90)
        
        for r in sorted(self.results, key=lambda x: x.roc_auc, reverse=True):
            cm = r.confusion_matrix
            print(
                f"{r.model_name:<12} "
                f"{r.roc_auc:<10.3f} {r.pr_auc:<10.3f} "
                f"{r.balanced_accuracy:<10.3f} {r.mcc:<8.3f} "
                f"{cm.precision:<8.3f} {cm.recall:<8.3f} {cm.specificity:<8.3f} {cm.f1:<8.3f}"
            )
        print("-" * 90)
        
        # Bootstrap CI summary
        print("\nBootstrap 95% CI:")
        print("-" * 90)
        for r in sorted(self.results, key=lambda x: x.roc_auc, reverse=True):
            print(
                f"{r.model_name:<12} "
                f"AUC: {r.bootstrap_roc_auc}  "
                f"F1: {r.bootstrap_f1}  "
                f"Recall: {r.bootstrap_recall}"
            )
        print("-" * 90)
        
        print(f"\nBest model: {self.best_model} (ROC-AUC: {self.best_auc:.3f})")


# =============================================================================
# Bootstrap Functions
# =============================================================================

def _bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_fn: Function(y_true, y_prob) -> float
        n_iterations: Number of bootstrap iterations
        ci_level: Confidence level (default 0.95)
        random_state: Random seed
        
    Returns:
        BootstrapCI with mean, std, and confidence interval
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            score = metric_fn(y_true_boot, y_prob_boot)
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return BootstrapCI()
    
    scores = np.array(scores)
    alpha = 1 - ci_level
    
    return BootstrapCI(
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        ci_lower=float(np.percentile(scores, 100 * alpha / 2)),
        ci_upper=float(np.percentile(scores, 100 * (1 - alpha / 2))),
    )


# =============================================================================
# Threshold Analysis
# =============================================================================

def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 50,
) -> ThresholdMetrics:
    """
    Compute precision, recall, F1 at different thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        ThresholdMetrics with arrays of metrics at each threshold
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    return ThresholdMetrics(
        thresholds=thresholds,
        precisions=np.array(precisions),
        recalls=np.array(recalls),
        f1_scores=np.array(f1_scores),
    )


# =============================================================================
# Model Training
# =============================================================================

def _require_sklearn() -> None:
    """Check if sklearn is installed."""
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "scikit-learn is required for modeling. Install: pip install scikit-learn"
        )


def _get_model(
    model_name: str,
    random_state: int = 42,
) -> object:
    """Get a model instance by name."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
        "lasso": LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "gbdt": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
    }
    
    # Try to add XGBoost and LightGBM if available
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
        )
    except ImportError as e:
        print(f"XGBoost not available: {e}")
    except Exception as e:
        print(f"XGBoost error: {e}")
    
    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError as e:
        print(f"LightGBM not available: {e}")
    except Exception as e:
        print(f"LightGBM error: {e}")
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name]


def available_models() -> list[str]:
    """Return model names available in the current environment."""
    names = ["logistic", "lasso", "rf", "gbdt", "svm"]
    try:
        import xgboost  # noqa: F401
        names.append("xgb")
    except Exception:
        pass
    try:
        import lightgbm  # noqa: F401
        names.append("lgbm")
    except Exception:
        pass
    return names


# Models that can handle missing values natively
NATIVE_MISSING_MODELS = {"xgb", "lgbm"}


def _get_feature_importances(
    model: object,
    feature_names: list[str],
) -> dict[str, float]:
    """Extract feature importances from model."""
    importances = {}
    
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(feature_names, model.feature_importances_):
            importances[name] = float(imp)
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten()
        for name, c in zip(feature_names, coef):
            importances[name] = float(abs(c))
    
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def _build_cv(
    y: np.ndarray,
    cv_folds: int,
    cv_strategy: Literal["kfold", "loo"],
    random_state: int,
):
    """Create a CV splitter based on strategy and data size."""
    from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

    n_samples = len(y)
    if n_samples < 2:
        return None

    if cv_strategy == "loo":
        return LeaveOneOut()

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return None

    min_class = int(counts.min())
    n_splits = min(cv_folds, n_samples, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


@dataclass
class SimpleTrainResult:
    metrics: dict[str, float]


def train_classifier(
    df: pl.DataFrame,
    target: str,
    model_name: Literal["logistic", "lasso", "rf", "gbdt"] = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SimpleTrainResult:
    """Train a baseline classifier directly from a merged DataFrame."""
    _require_sklearn()
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    y = df.get_column(target)
    X = df.drop(target)
    cat_cols = [name for name, dtype in zip(X.columns, X.dtypes) if dtype == pl.Utf8]
    if cat_cols:
        X = X.to_dummies(columns=cat_cols)

    X_np = X.to_numpy()
    y_np = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, stratify=y_np if len(set(y_np)) > 1 else None
    )

    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "lasso":
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError("model_name must be one of: logistic, lasso, rf, gbdt")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    from ctg_ml_pipeline.modeling.evaluate import evaluate_classification

    result = evaluate_classification(y_test, y_pred, y_proba=y_proba)
    return SimpleTrainResult(metrics=result.metrics)


def train_and_evaluate(
    dataset: "TrialDataset",
    model_name: Literal["logistic", "lasso", "rf", "gbdt", "svm"] = "rf",
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "loo"] = "kfold",
    cv_scoring: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    cv_only: bool = False,
) -> ModelResult:
    """
    Train and evaluate a model with comprehensive metrics.
    
    Args:
        dataset: TrialDataset instance
        model_name: Model type ("logistic", "rf", "gbdt", "svm")
        cv_folds: Number of cross-validation folds
        cv_strategy: "kfold" or "loo" (leave-one-out)
        cv_scoring: sklearn scoring string (default: roc_auc for kfold, accuracy for loo)
        n_bootstrap: Number of bootstrap iterations for CI
        random_state: Random seed
        
    Returns:
        ModelResult with comprehensive metrics
    """
    _require_sklearn()
    
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, matthews_corrcoef,
        roc_auc_score, average_precision_score, confusion_matrix,
        f1_score, recall_score,
    )
    from sklearn.model_selection import cross_val_score
    
    # Create model
    model = _get_model(model_name, random_state)

    if cv_only:
        X = dataset.X
        y = dataset.y

        if cv_scoring is None:
            cv_scoring = "accuracy" if cv_strategy == "loo" else "roc_auc"
        cv_splitter = _build_cv(np.asarray(y), cv_folds, cv_strategy, random_state)
        if cv_splitter is None:
            raise RuntimeError("CV splitter is None (insufficient samples or classes).")

        # CV predictions
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(model, X, y, cv=cv_splitter, method="predict")
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = cross_val_predict(model, X, y, cv=cv_splitter, method="predict_proba")[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = cross_val_predict(model, X, y, cv=cv_splitter, method="decision_function")
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)

        cv_scores = cross_val_score(
            model, X, y,
            cv=cv_splitter,
            scoring=cv_scoring,
            error_score=np.nan,
        )

        # Fit on full dataset for importances / SHAP
        model.fit(X, y)

        train_acc = accuracy_score(y, y_pred)
        test_acc = train_acc
        y_train = y
        y_test = y
        y_train_pred = y_pred
        y_test_pred = y_pred
        test_idx = np.arange(len(y))
    else:
        # Get train/test split with indices
        X_train, X_test, y_train, y_test, _, test_idx = dataset.get_train_test_split(return_indices=True)

        # Cross-validation on training set
        if cv_scoring is None:
            cv_scoring = "accuracy" if cv_strategy == "loo" else "roc_auc"
        cv_splitter = _build_cv(np.asarray(y_train), cv_folds, cv_strategy, random_state)
        if cv_splitter is None:
            cv_scores = np.array([])
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_splitter,
                scoring=cv_scoring,
                error_score=np.nan,
            )

        # Train on full training set
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get probabilities
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            # Normalize to [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)

        # Basic metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_metrics = ConfusionMatrixMetrics(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    
    # Balanced accuracy & MCC
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    
    # AUC metrics
    roc_auc = 0.0
    pr_auc = 0.0
    if y_prob is not None and len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    
    # Bootstrap CI
    bootstrap_roc_auc = BootstrapCI()
    bootstrap_f1 = BootstrapCI()
    bootstrap_recall = BootstrapCI()
    
    if y_prob is not None and len(y_test) >= 10:
        bootstrap_roc_auc = _bootstrap_metric(
            y_test, y_prob,
            lambda y, p: roc_auc_score(y, p),
            n_iterations=n_bootstrap,
            random_state=random_state,
        )
        
        bootstrap_f1 = _bootstrap_metric(
            y_test, y_prob,
            lambda y, p: f1_score(y, (p >= 0.5).astype(int)),
            n_iterations=n_bootstrap,
            random_state=random_state,
        )
        
        bootstrap_recall = _bootstrap_metric(
            y_test, y_prob,
            lambda y, p: recall_score(y, (p >= 0.5).astype(int)),
            n_iterations=n_bootstrap,
            random_state=random_state,
        )
    
    # Threshold analysis
    threshold_metrics = ThresholdMetrics()
    if y_prob is not None:
        threshold_metrics = _compute_threshold_metrics(y_test, y_prob)
    
    # Feature importances
    importances = _get_feature_importances(model, dataset.feature_names)
    
    return ModelResult(
        model_name=model_name,
        model=model,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        confusion_matrix=cm_metrics,
        balanced_accuracy=bal_acc,
        mcc=mcc,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        cv_auc_mean=float(np.nanmean(cv_scores)) if cv_scores.size else 0.0,
        cv_auc_std=float(np.nanstd(cv_scores)) if cv_scores.size else 0.0,
        cv_scoring=cv_scoring,
        bootstrap_roc_auc=bootstrap_roc_auc,
        bootstrap_f1=bootstrap_f1,
        bootstrap_recall=bootstrap_recall,
        threshold_metrics=threshold_metrics,
        feature_importances=importances,
        y_test=y_test,
        y_prob=y_prob if y_prob is not None else np.array([]),
        y_pred=y_test_pred,
        test_indices=test_idx,
    )


def compare_models(
    dataset: "TrialDataset",
    models: list[str] | None = None,
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "loo"] = "kfold",
    cv_scoring: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    cv_only: bool = False,
) -> ComparisonResult:
    """
    Compare multiple models on the dataset.
    
    Args:
        dataset: TrialDataset instance
        models: List of model names to compare (default: all)
        cv_folds: Number of cross-validation folds
        cv_strategy: "kfold" or "loo" (leave-one-out)
        cv_scoring: sklearn scoring string (default: roc_auc for kfold, accuracy for loo)
        n_bootstrap: Number of bootstrap iterations for CI
        random_state: Random seed
        
    Returns:
        ComparisonResult with all model results
    """
    if models is None:
        models = ["logistic", "lasso", "rf", "gbdt"]
    
    # Print features before training
    print("\n" + "-" * 70)
    print(f"Features used for modeling ({len(dataset.feature_names)} total):")
    print("-" * 70)
    for i, name in enumerate(dataset.feature_names):
        stats = dataset.feature_stats.get(name, {})
        display_type = stats.get("display_type")
        if display_type:
            type_str = display_type
        else:
            feat_type = dataset.feature_types.get(name, "unknown")
            type_str = feat_type.value if hasattr(feat_type, "value") else str(feat_type)
        print(f"  {i+1:2}. {name:<35} ({type_str})")
    print("-" * 70 + "\n")
    results = []
    for model_name in models:
        print(f"Training {model_name}...")
        result = train_and_evaluate(
            dataset,
            model_name,
            cv_folds,
            cv_strategy,
            cv_scoring,
            n_bootstrap,
            random_state,
            cv_only=cv_only,
        )
        results.append(result)
    
    # Find best model by ROC-AUC
    best = max(results, key=lambda x: x.roc_auc)
    
    return ComparisonResult(
        results=results,
        best_model=best.model_name,
        best_auc=best.roc_auc,
    )


# =============================================================================
# Feature Importance & SHAP Analysis
# =============================================================================

def _print_feature_importance(
    result: ModelResult,
    dataset: "TrialDataset",
    top_n: int = 15,
) -> None:
    """Print feature importance for the model."""
    if not result.feature_importances:
        print("\n[Feature Importance] Not available for this model.")
        return
    
    print("\n" + "=" * 70)
    print(f"Feature Importance ({result.model_name})")
    print("=" * 70)
    
    importances = result.feature_importances
    total_imp = sum(importances.values())
    
    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':>12} {'Cumulative':>12}")
    print("-" * 70)
    
    cumulative = 0.0
    for i, (feat, imp) in enumerate(list(importances.items())[:top_n]):
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        cumulative += pct
        print(f"{i+1:<6} {feat:<35} {pct:>11.2f}% {cumulative:>11.2f}%")
    
    if len(importances) > top_n:
        remaining = len(importances) - top_n
        remaining_imp = sum(list(importances.values())[top_n:]) / total_imp * 100
        print(f"{'...':<6} {'(other ' + str(remaining) + ' features)':<35} {remaining_imp:>11.2f}%")
    
    print("-" * 70)


def _print_shap_analysis(
    result: ModelResult,
    dataset: "TrialDataset",
    top_n: int = 10,
) -> None:
    """Print SHAP analysis for the model."""
    try:
        import shap
    except ImportError:
        print("\n[SHAP Analysis] shap not installed. Run: pip install shap")
        return
    
    # Get test data
    X_train, X_test, y_train, y_test, _, test_idx = dataset.get_train_test_split(return_indices=True)
    
    print("\n" + "=" * 70)
    print(f"SHAP Analysis ({result.model_name})")
    print("=" * 70)
    
    try:
        # Create SHAP explainer based on model type
        model = result.model
        
        if result.model_name in ["xgb", "lgbm"]:
            # XGBoost/LightGBM use TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif result.model_name in ["rf", "gbdt"]:
            # sklearn tree models - use Explainer with check_additivity=False
            explainer = shap.Explainer(model, X_train, feature_names=dataset.feature_names)
            shap_values_obj = explainer(X_test, check_additivity=False)
            shap_values = shap_values_obj.values
            # For multi-output, get positive class
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        else:
            # Other models use KernelExplainer (slower)
            print("  Using KernelExplainer (may be slow)...")
            # Use a subset of training data as background
            background = shap.sample(X_train, min(50, len(X_train)))
            
            def predict_fn(x):
                return model.predict_proba(x)[:, 1]
            
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test, nsamples=100)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_shap = list(zip(dataset.feature_names, mean_abs_shap))
        feature_shap.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_n} features by mean |SHAP value|:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>15}")
        print("-" * 70)
        
        for i, (feat, shap_val) in enumerate(feature_shap[:top_n]):
            print(f"{i+1:<6} {feat:<35} {shap_val:>15.4f}")
        
        print("-" * 70)
        
        # Show SHAP summary for a few test samples
        print(f"\nSHAP values for test samples (showing top features):")
        print("-" * 70)
        
        top_features = [f for f, _ in feature_shap[:5]]
        top_indices = [dataset.feature_names.index(f) for f in top_features]
        
        print(f"{'StudyID':<15}", end="")
        for f in top_features:
            print(f"{f[:12]:<14}", end="")
        print(f"{'y_true':<8} {'y_pred':<8}")
        print("-" * 70)
        
        for i in range(min(5, len(X_test))):
            study_id = dataset.study_ids[int(test_idx[i])][:13]
            print(f"{study_id:<15}", end="")
            for idx in top_indices:
                val = shap_values[i, idx]
                sign = "+" if val >= 0 else ""
                print(f"{sign}{val:<13.3f}", end="")
            print(f"{int(y_test[i]):<8} {int(result.y_pred[i]):<8}")
        
        print("-" * 70)
        print("Note: Positive SHAP = pushes toward positive class (success)")
        
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_experiment(
    group_dir: str | Path,
    target_csv: str | Path,
    output_dir: str | Path | None = None,
    max_missing_rate: float = 0.5,
    time_split: bool = True,
    models: list[str] | None = None,
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "loo"] = "kfold",
    cv_scoring: str | None = None,
    cv_only: bool = False,
    select_stage: Literal["filter", "embedded", "none"] = "none",
    select_method: str = "mutual_info",
    select_top_ratio: float = 0.2,
    text_as_bool: bool = False,
    phase_filter: list[str] | str | None = None,
    impute_strategy: Literal["median", "mean", "zero", "none"] = "median",
) -> ComparisonResult:
    """
    Run a complete ML experiment.
    
    Args:
        group_dir: Directory containing trial data
        target_csv: Path to target CSV file
        output_dir: Optional output directory for results
        max_missing_rate: Maximum missing rate for features
        time_split: Whether to use time-based split
        models: List of models to compare
        cv_folds: Number of cross-validation folds
        cv_strategy: "kfold" or "loo" (leave-one-out)
        cv_scoring: sklearn scoring string (default: roc_auc for kfold, accuracy for loo)
        
    Returns:
        ComparisonResult with all results
    """
    from ctg_ml_pipeline.data.dataset import load_trial_dataset
    
    print("=" * 70)
    print("Clinical Trial Outcome Prediction Experiment")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = load_trial_dataset(
        group_dir=group_dir,
        target_csv=target_csv,
        max_missing_rate=max_missing_rate,
        time_split=time_split,
        text_as_bool=text_as_bool,
        phase_filter=phase_filter,
        impute_strategy=impute_strategy,
    )

    # Print per-feature missing rates after missing-rate filter (before selection)
    if dataset.feature_stats:
        print("\n" + "-" * 70)
        print("Missing Rates After Filter")
        print("-" * 70)
        feature_info = dataset.get_feature_info()
        for row in feature_info.sort("missing_rate", descending=True).iter_rows(named=True):
            feat_display = row["feature"][:33] + ".." if len(row["feature"]) > 35 else row["feature"]
            table_display = row["table"].replace("_all", "")[:16]
            print(f"{feat_display:<35} {row['type']:<12} {table_display:<18} {row['missing_rate']:>7.1%}")

    if select_stage != "none":
        from ctg_ml_pipeline.preprocess.selection import (
            filter_stage_matrix,
            embedded_stage_matrix,
        )
        if cv_only:
            X_sel, y_sel = dataset.X, dataset.y
        else:
            X_sel, _, y_sel, _, _, _ = dataset.get_train_test_split(return_indices=True)
        if select_stage == "filter":
            result = filter_stage_matrix(
                X_sel,
                y_sel,
                dataset.feature_names,
                method=select_method,
                top_ratio=select_top_ratio,
            )
        else:
            result = embedded_stage_matrix(
                X_sel,
                y_sel,
                dataset.feature_names,
                method=select_method,
                top_ratio=select_top_ratio,
            )
        dataset.select_features(result.selected_features)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            payload = {
                "stage": select_stage,
                "method": select_method,
                "top_ratio": select_top_ratio,
                "selected_features": result.selected_features,
                "scores": result.scores,
            }
            (output_path / "feature_selection.json").write_text(json.dumps(payload, indent=2))
        dataset.feature_filter_summary["selection_stage"] = select_stage
        dataset.feature_filter_summary["selection_method"] = select_method
        dataset.feature_filter_summary["selection_top_ratio"] = select_top_ratio
    else:
        dataset.feature_filter_summary["selection_stage"] = "none"
        dataset.feature_filter_summary["selection_method"] = ""
        dataset.feature_filter_summary["selection_top_ratio"] = 0.0

    dataset.feature_filter_summary["post_selection"] = len(dataset.feature_names)

    if dataset.feature_filter_summary:
        print("\n" + "-" * 70)
        print("Feature Filtering Summary")
        print("-" * 70)
        if dataset.feature_filter_summary.get("allowlist_used"):
            print(
                f"Allowlist: {dataset.feature_filter_summary.get('allowlist_present', 0)} present "
                f"/ {dataset.feature_filter_summary.get('allowlist_total', 0)} total"
            )
        else:
            print("Allowlist: not used")
        print(f"After missing-rate: {dataset.feature_filter_summary.get('post_missing', 0)}")
        print(f"After selection:    {dataset.feature_filter_summary.get('post_selection', 0)}")

    if cv_only:
        n_num = sum(1 for t in dataset.feature_types.values() if t == FeatureType.NUMERIC)
        n_cat = sum(1 for t in dataset.feature_types.values() if t == FeatureType.CATEGORICAL)
        print("\nCV-only mode (no holdout split)")
        print(f"Samples: {len(dataset)}")
        print(f"Features: {len(dataset.feature_names)} (numeric={n_num}, categorical={n_cat})")
    else:
        dataset.summary()
    
    # Train models
    print("\n[2/3] Training models...")
    comparison = compare_models(
        dataset,
        models=models,
        cv_folds=cv_folds,
        cv_strategy=cv_strategy,
        cv_scoring=cv_scoring,
        cv_only=cv_only,
    )
    
    # Show results
    print("\n[3/3] Results")
    comparison.summary()
    
    # Print detailed metrics for best model
    best_result = next(r for r in comparison.results if r.model_name == comparison.best_model)
    best_result.print_detailed_metrics()
    
    # Print feature importance
    _print_feature_importance(best_result, dataset)
    
    # SHAP analysis
    _print_shap_analysis(best_result, dataset)
    
    # Export if output_dir specified
    if output_dir:
        _export_results(output_dir, dataset, comparison)
    
    return comparison


def _export_results(
    output_dir: str | Path,
    dataset: "TrialDataset",
    comparison: ComparisonResult,
) -> None:
    """Export results to files."""
    import json
    import polars as pl
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export feature info
    dataset.get_feature_info().write_csv(output_path / "features.csv")
    
    # Export comprehensive model metrics
    metrics_data = []
    for r in comparison.results:
        cm = r.confusion_matrix
        metrics_data.append({
            "model": r.model_name,
            # Basic
            "train_accuracy": r.train_accuracy,
            "test_accuracy": r.test_accuracy,
            # Confusion matrix derived
            "precision": cm.precision,
            "recall": cm.recall,
            "specificity": cm.specificity,
            "f1": cm.f1,
            "npv": cm.npv,
            # Robust metrics
            "balanced_accuracy": r.balanced_accuracy,
            "mcc": r.mcc,
            # AUC
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "cv_auc_mean": r.cv_auc_mean,
            "cv_auc_std": r.cv_auc_std,
            "cv_scoring": r.cv_scoring,
            # Bootstrap CI
            "roc_auc_ci_lower": r.bootstrap_roc_auc.ci_lower,
            "roc_auc_ci_upper": r.bootstrap_roc_auc.ci_upper,
            "f1_ci_lower": r.bootstrap_f1.ci_lower,
            "f1_ci_upper": r.bootstrap_f1.ci_upper,
            "recall_ci_lower": r.bootstrap_recall.ci_lower,
            "recall_ci_upper": r.bootstrap_recall.ci_upper,
            # Threshold
            "best_f1_threshold": r.threshold_metrics.best_f1_threshold,
            "best_f1": r.threshold_metrics.best_f1,
        })
    pl.DataFrame(metrics_data).write_csv(output_path / "model_metrics.csv")
    
    # Export confusion matrices
    cm_data = []
    for r in comparison.results:
        cm = r.confusion_matrix
        cm_data.append({
            "model": r.model_name,
            "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "tp": cm.tp,
        })
    pl.DataFrame(cm_data).write_csv(output_path / "confusion_matrices.csv")
    
    # Export feature importances for best model
    best_result = next(r for r in comparison.results if r.model_name == comparison.best_model)
    if best_result.feature_importances:
        imp_data = [
            {"feature": k, "importance": v}
            for k, v in best_result.feature_importances.items()
        ]
        pl.DataFrame(imp_data).write_csv(output_path / "feature_importances.csv")
        try:
            import matplotlib.pyplot as plt

            imp_df = pl.DataFrame(imp_data).sort("importance", descending=True)
            top_n = min(30, imp_df.height)
            imp_top = imp_df.head(top_n)
            features = imp_top.get_column("feature").to_list()
            values = imp_top.get_column("importance").to_list()

            fig_height = max(4, 0.25 * len(features))
            plt.figure(figsize=(10, fig_height))
            plt.barh(features[::-1], values[::-1])
            plt.xlabel("Importance")
            plt.title(f"Feature importance ({best_result.model_name})")
            plt.tight_layout()
            plt.savefig(output_path / "feature_importances.png", dpi=150)
            plt.close()
        except Exception as exc:
            print(f"Feature importance plot failed: {exc}")

    # Export best model params
    best_params = {}
    try:
        best_params = best_result.model.get_params()
    except Exception:
        best_params = {}
    (output_path / "best_model_params.json").write_text(
        json.dumps(
            {"model": best_result.model_name, "params": best_params},
            indent=2,
        )
    )

    # Save best model checkpoint
    model_path = output_path / "best_model.pkl"
    try:
        import joblib
        joblib.dump(best_result.model, model_path)
    except Exception:
        import pickle
        with model_path.open("wb") as f:
            pickle.dump(best_result.model, f)
    
    # Export threshold analysis for best model
    tm = best_result.threshold_metrics
    if len(tm.thresholds) > 0:
        thresh_data = [
            {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }
            for t, p, r, f in zip(tm.thresholds, tm.precisions, tm.recalls, tm.f1_scores)
        ]
        pl.DataFrame(thresh_data).write_csv(output_path / "threshold_analysis.csv")
    
    # Export test set predictions for best model
    if len(best_result.test_indices) > 0:
        test_predictions = []
        for i, idx in enumerate(best_result.test_indices):
            study_id = dataset.study_ids[int(idx)]
            y_true = int(best_result.y_test[i])
            y_pred = int(best_result.y_pred[i])
            y_prob = float(best_result.y_prob[i]) if len(best_result.y_prob) > 0 else None
            
            test_predictions.append({
                "StudyID": study_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "correct": y_true == y_pred,
            })
        pl.DataFrame(test_predictions).write_csv(output_path / "test_predictions.csv")
    
    # Export summary
    summary = {
        "n_samples": len(dataset),
        "n_features": len(dataset.feature_names),
        "best_model": comparison.best_model,
        "best_roc_auc": comparison.best_auc,
        "best_pr_auc": best_result.pr_auc,
        "best_mcc": best_result.mcc,
        "split_info": dataset.get_split_info(),
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults exported to: {output_path}")


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clinical Trial Outcome Prediction")
    parser.add_argument(
        "--group-dir",
        default="data/ctg_extract_v2/NSCLC_Trialpanorama_pd1",
        help="Directory containing trial data",
    )
    parser.add_argument(
        "--target-csv",
        default="data/raw/NSCLC_Trialpanorama_pd1_brief_summary.csv",
        help="Path to target CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="output/experiment_results_all",
        help="Output directory for results, or 'auto' to build from parameters",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=[],
        help="Filter by Study_Phase (e.g. --phase 2 or --phase 1 2 or --phase 1,2)",
    )
    parser.add_argument(
        "--max-missing-rate",
        type=float,
        default=0.5,
        help="Maximum missing rate for features (default: 0.5)",
    )
    parser.add_argument(
        "--no-time-split",
        action="store_true",
        help="Use random split instead of time-based split",
    )
    parser.add_argument(
        "--impute-strategy",
        choices=["median", "mean", "zero", "none"],
        default="median",
        help="Imputation strategy (default: median). Use 'none' to keep NaN",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "rf", "gbdt"],
        help="Models to compare or 'all' (default: logistic rf gbdt)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--cv-strategy",
        choices=["kfold", "loo"],
        default="kfold",
        help="Cross-validation strategy (kfold or loo)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Disable cross-validation",
    )
    parser.add_argument(
        "--use-test-split",
        action="store_true",
        help="Use holdout train/test split alongside CV",
    )
    parser.add_argument(
        "--cv-scoring",
        default="",
        help="sklearn scoring string (default: roc_auc for kfold, accuracy for loo)",
    )
    parser.add_argument(
        "--select-stage",
        choices=["none", "filter", "embedded"],
        default="filter",
        help="Feature selection stage (default: filter)",
    )
    parser.add_argument(
        "--select-method",
        default="",
        help="Selection method (filter: mutual_info/anova/fisher/chi2; embedded: l1/rf/gbdt)",
    )
    parser.add_argument(
        "--select-top-ratio",
        type=float,
        default=0.2,
        help="Top ratio of features to keep after selection (default: 0.2)",
    )
    parser.add_argument(
        "--text-as-bool",
        action="store_true",
        help="Convert text features to boolean presence flags",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before training",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Only run tuning and exit",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=0,
        help="Optuna timeout seconds (0 = no limit)",
    )
    parser.add_argument(
        "--tune-output",
        default="",
        help="Path to write tuning results JSON (default: <output-dir>/tuning_results.json)",
    )
    
    args = parser.parse_args()

    models = args.models
    if any(m.lower() == "all" for m in models):
        models = available_models()

    phase_filter: list[str] = []
    for raw in args.phase:
        if not raw:
            continue
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                phase_filter.append(part)

    select_method = args.select_method
    if not select_method:
        select_method = "mutual_info" if args.select_stage == "filter" else "l1"

    output_dir = args.output_dir
    if str(output_dir).lower() == "auto":
        output_dir = _build_auto_output_dir(
            base="output",
            models=models,
            cv_strategy=args.cv_strategy,
            cv_folds=args.cv_folds,
            no_cv=args.no_cv,
            time_split=not args.no_time_split,
            max_missing_rate=args.max_missing_rate,
            select_stage=args.select_stage,
            select_method=select_method,
            select_top_ratio=args.select_top_ratio,
            text_as_bool=args.text_as_bool,
            phase_filter=phase_filter,
            tuned=args.tune,
        )

    if args.no_cv:
        print("CV disabled (--no-cv)")

    if args.tune:
        from ctg_ml_pipeline.data.dataset import load_trial_dataset
        from ctg_ml_pipeline.modeling.tuning import tune_models

        dataset = load_trial_dataset(
            group_dir=args.group_dir,
            target_csv=args.target_csv,
            max_missing_rate=args.max_missing_rate,
            time_split=not args.no_time_split,
            text_as_bool=args.text_as_bool,
            phase_filter=phase_filter,
            impute_strategy=args.impute_strategy,
        )

        timeout = args.tune_timeout if args.tune_timeout > 0 else None
        tune_results = tune_models(
            dataset,
            models=models,
            n_trials=args.tune_trials,
            cv_folds=args.cv_folds,
            cv_strategy=args.cv_strategy,
            cv_scoring=args.cv_scoring or None,
            timeout=timeout,
        )

        output_path = Path(output_dir) if output_dir else Path("output/experiment_results_all")
        output_path.mkdir(parents=True, exist_ok=True)
        tuning_path = Path(args.tune_output) if args.tune_output else (output_path / "tuning_results.json")
        payload = {
            name: {
                "best_score": result.best_score,
                "best_params": result.best_params,
                "cv_scoring": result.cv_scoring,
                "n_trials": result.n_trials,
            }
            for name, result in tune_results.items()
        }
        tuning_path.write_text(json.dumps(payload, indent=2))
        print(f"Tuning results saved to: {tuning_path}")

        if args.tune_only:
            raise SystemExit(0)

    cv_folds = 0 if args.no_cv else args.cv_folds
    cv_only = (not args.no_cv) and (not args.use_test_split)

    run_experiment(
        group_dir=args.group_dir,
        target_csv=args.target_csv,
        output_dir=output_dir,
        max_missing_rate=args.max_missing_rate,
        time_split=not args.no_time_split,
        models=models,
        cv_folds=cv_folds,
        cv_strategy=args.cv_strategy,
        cv_scoring=args.cv_scoring or None,
        select_stage=args.select_stage,
        select_method=select_method,
        select_top_ratio=args.select_top_ratio,
        text_as_bool=args.text_as_bool,
        cv_only=cv_only,
        phase_filter=phase_filter,
        impute_strategy=args.impute_strategy,
    )
