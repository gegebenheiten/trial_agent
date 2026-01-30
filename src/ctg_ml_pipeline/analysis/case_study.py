"""
Case study utilities for inspecting model behavior on specific trials.

Outputs per-study suggestions for single-feature changes that increase
predicted probability (based on the trained model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ctg_ml_pipeline.data.dataset import FeatureType, load_trial_dataset
from ctg_ml_pipeline.modeling.modeling import _build_cv, _get_model


def _load_run_config(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return {}
    run_cfg = payload.get("run_config")
    return run_cfg if isinstance(run_cfg, dict) else {}


def _load_tuned_params(output_dir: Path, run_config: dict) -> dict[str, dict[str, object]]:
    path = run_config.get("tuning_results_path") if isinstance(run_config, dict) else ""
    if not path:
        path = str(output_dir / "tuning_results.json")
    tuning_path = Path(path)
    if not tuning_path.exists():
        return {}
    try:
        payload = json.loads(tuning_path.read_text())
    except Exception:
        return {}
    tuned: dict[str, dict[str, object]] = {}
    for name, info in payload.items():
        if not isinstance(info, dict):
            continue
        params = info.get("best_params")
        if isinstance(params, dict):
            tuned[str(name)] = params
    return tuned


def _load_best_model(output_dir: Path):
    model_path = output_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        import pickle
        with model_path.open("rb") as f:
            return pickle.load(f)


def _load_feature_list(output_dir: Path) -> list[str]:
    feat_path = output_dir / "features.csv"
    if not feat_path.exists():
        return []
    df = pl.read_csv(feat_path)
    if "feature" not in df.columns:
        return []
    return [str(v) for v in df.get_column("feature").to_list() if v]


def _load_best_model_name(output_dir: Path) -> str | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return None
    name = payload.get("best_model")
    return str(name) if name else None


def _predict_proba_single(model, x: np.ndarray) -> float:
    x2 = x.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x2)[:, 1][0])
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x2)
        score = float(scores[0])
        # Min-max normalize to [0,1] for a single point (fallback)
        return 1.0 / (1.0 + np.exp(-score))
    return float(model.predict(x2)[0])


def _scale_value(dataset, col_idx: int, raw_val: float) -> float:
    if getattr(dataset, "_scaler", None) is None:
        return raw_val
    mean = float(dataset._scaler.mean_[col_idx])
    scale = float(dataset._scaler.scale_[col_idx])
    return (raw_val - mean) / (scale if scale != 0 else 1.0)


def _decode_label(dataset, feature: str, raw_val: float) -> str | None:
    le = dataset._label_encoders.get(feature)
    if le is None:
        return None
    idx = int(round(raw_val))
    if 0 <= idx < len(le.classes_):
        return str(le.classes_[idx])
    return None


def _suggest_single_feature_changes(
    dataset_scaled,
    dataset_raw,
    model,
    idx: int,
    top_k: int = 10,
) -> list[dict]:
    X_scaled = dataset_scaled.X
    X_raw = dataset_raw.X

    base_row_scaled = X_scaled[idx].copy()
    base_row_raw = X_raw[idx].copy()
    base_prob = _predict_proba_single(model, base_row_scaled)

    suggestions: list[dict] = []
    for j, feature in enumerate(dataset_scaled.feature_names):
        ftype = dataset_scaled.feature_types.get(feature, FeatureType.NUMERIC)
        stat = dataset_scaled.feature_stats.get(feature, {})
        raw_col = X_raw[:, j]
        current_raw = float(base_row_raw[j])

        candidates: list[float] = []
        if stat.get("type") == "text_boolean":
            candidates = [0.0, 1.0]
        elif ftype == FeatureType.CATEGORICAL:
            uniq = np.unique(raw_col[~np.isnan(raw_col)])
            if len(uniq) > 0:
                candidates = [float(v) for v in uniq.tolist()]
        else:
            if np.all(np.isnan(raw_col)):
                continue
            qs = np.nanquantile(raw_col, [0.1, 0.5, 0.9])
            candidates = [float(v) for v in qs.tolist()]

        if not candidates:
            continue

        best_prob = base_prob
        best_val = current_raw
        for cand in candidates:
            if np.isclose(cand, current_raw, atol=1e-8):
                continue
            mod_row = base_row_scaled.copy()
            mod_row[j] = _scale_value(dataset_scaled, j, cand)
            prob = _predict_proba_single(model, mod_row)
            if prob > best_prob:
                best_prob = prob
                best_val = cand

        if best_prob > base_prob + 1e-6:
            suggestions.append(
                {
                    "feature": feature,
                    "feature_type": stat.get("display_type") or ftype.value,
                    "current_value": current_raw,
                    "suggested_value": best_val,
                    "current_label": _decode_label(dataset_scaled, feature, current_raw),
                    "suggested_label": _decode_label(dataset_scaled, feature, best_val),
                    "base_prob": base_prob,
                    "new_prob": best_prob,
                    "delta": best_prob - base_prob,
                }
            )

    suggestions.sort(key=lambda r: r["delta"], reverse=True)
    return suggestions[:top_k]


def run_case_study(
    *,
    output_dir: Path,
    group_dir: str | Path,
    target_csv: str | Path,
    study_ids: Iterable[str],
    max_missing_rate: float = 1.0,
    text_as_bool: bool = False,
    phase_filter: list[str] | None = None,
    top_k: int = 10,
    use_oof: bool | None = None,
    model_name: str | None = None,
    cv_strategy: str | None = None,
    cv_folds: int | None = None,
    random_state: int = 42,
) -> None:
    run_config = _load_run_config(output_dir)
    if use_oof is None:
        use_oof = bool(run_config.get("cv_only", False))
    if cv_strategy is None or cv_strategy == "auto":
        cv_strategy = str(run_config.get("cv_strategy") or "kfold")
    if cv_folds is None or cv_folds <= 0:
        cfg_folds = run_config.get("cv_folds")
        cv_folds = int(cfg_folds) if isinstance(cfg_folds, int) and cfg_folds > 0 else 5

    model = None if use_oof else _load_best_model(output_dir)
    feature_list = _load_feature_list(output_dir)

    dataset_scaled = load_trial_dataset(
        group_dir=group_dir,
        target_csv=target_csv,
        max_missing_rate=max_missing_rate,
        time_split=True,
        text_as_bool=text_as_bool,
        phase_filter=phase_filter or [],
    )
    dataset_raw = load_trial_dataset(
        group_dir=group_dir,
        target_csv=target_csv,
        max_missing_rate=max_missing_rate,
        time_split=True,
        text_as_bool=text_as_bool,
        phase_filter=phase_filter or [],
        scale_features=False,
    )

    if feature_list:
        dataset_scaled.select_features(feature_list)
        dataset_raw.select_features(feature_list)

    id_to_idx = {sid: i for i, sid in enumerate(dataset_scaled.study_ids)}
    folds = None
    oof_model_name = None
    tuned_params = _load_tuned_params(output_dir, run_config) if use_oof else {}
    if use_oof:
        oof_model_name = model_name or _load_best_model_name(output_dir)
        if not oof_model_name:
            raise ValueError("Missing best_model in summary.json; pass --model-name explicitly.")
        cv_splitter = _build_cv(dataset_scaled.y, cv_folds, cv_strategy, random_state)
        if cv_splitter is None:
            raise ValueError("CV splitter is None; not enough samples/classes.")
        folds = list(cv_splitter.split(dataset_scaled.X, dataset_scaled.y))
    out_rows = []
    for sid in study_ids:
        if sid not in id_to_idx:
            raise ValueError(f"StudyID not found after filters: {sid}")
        idx = id_to_idx[sid]
        if use_oof:
            if folds is None:
                raise ValueError("OOF folds not available.")
            fold_model = None
            for train_idx, test_idx in folds:
                if idx in test_idx:
                    fold_model = _get_model(
                        oof_model_name,
                        random_state,
                        params=tuned_params.get(oof_model_name),
                    )
                    fold_model.fit(dataset_scaled.X[train_idx], dataset_scaled.y[train_idx])
                    break
            if fold_model is None:
                raise ValueError(f"StudyID {sid} not found in any CV fold.")
            base_prob = _predict_proba_single(fold_model, dataset_scaled.X[idx])
            model_for_suggestions = fold_model
        else:
            base_prob = _predict_proba_single(model, dataset_scaled.X[idx])
            model_for_suggestions = model
        base_pred = int(base_prob >= 0.5)
        y_true = int(dataset_scaled.y[idx])

        suggestions = _suggest_single_feature_changes(
            dataset_scaled,
            dataset_raw,
            model_for_suggestions,
            idx,
            top_k=top_k,
        )
        if not suggestions:
            out_rows.append(
                {
                    "StudyID": sid,
                    "y_true": y_true,
                    "base_prob": base_prob,
                    "base_pred": base_pred,
                    "note": "no single-feature change improved probability",
                }
            )
            continue

        for s in suggestions:
            out_rows.append(
                {
                    "StudyID": sid,
                    "y_true": y_true,
                    "base_prob": s["base_prob"],
                    "base_pred": base_pred,
                    "feature": s["feature"],
                    "feature_type": s["feature_type"],
                    "current_value": s["current_value"],
                    "suggested_value": s["suggested_value"],
                    "current_label": s["current_label"],
                    "suggested_label": s["suggested_label"],
                    "new_prob": s["new_prob"],
                    "delta": s["delta"],
                    "flips_pred": int(s["new_prob"] >= 0.5 and base_pred == 0)
                    if base_pred == 0
                    else int(s["new_prob"] < 0.5 and base_pred == 1),
                }
            )

    out_df = pl.DataFrame(out_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "case_study_suggestions.csv"
    out_df.write_csv(out_path)
    print(f"Case study suggestions saved to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Case study for selected trials")
    parser.add_argument("--output-dir", required=True, help="Output dir with best_model.pkl + features.csv")
    parser.add_argument("--group-dir", required=True, help="Dataset group dir")
    parser.add_argument("--target-csv", required=True, help="Target labels CSV")
    parser.add_argument("--study-id", nargs="+", required=True, help="One or more StudyID values")
    parser.add_argument("--max-missing-rate", type=float, default=1.0, help="Max missing rate for loading")
    parser.add_argument("--text-as-bool", action="store_true", help="Use text-as-boolean features")
    parser.add_argument("--phase", nargs="+", default=[], help="Phase filter (e.g., 1 or 1,2)")
    parser.add_argument("--top-k", type=int, default=10, help="Top single-feature changes to report")
    parser.add_argument("--use-oof", action="store_true", default=None, help="Use CV fold model for each sample (OOF)")
    parser.add_argument("--no-use-oof", action="store_true", help="Force using best_model.pkl instead of OOF")
    parser.add_argument("--model-name", default="", help="Model name for OOF (default: best_model from summary.json)")
    parser.add_argument("--cv-strategy", choices=["auto", "kfold", "loo"], default="auto", help="CV strategy for OOF")
    parser.add_argument("--cv-folds", type=int, default=0, help="CV folds for OOF (ignored for loo)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for OOF folds")
    args = parser.parse_args()

    phase_filter: list[str] = []
    for raw in args.phase:
        if not raw:
            continue
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                phase_filter.append(part)

    use_oof = args.use_oof
    if args.no_use_oof:
        use_oof = False

    run_case_study(
        output_dir=Path(args.output_dir),
        group_dir=args.group_dir,
        target_csv=args.target_csv,
        study_ids=args.study_id,
        max_missing_rate=args.max_missing_rate,
        text_as_bool=args.text_as_bool,
        phase_filter=phase_filter,
        top_k=args.top_k,
        use_oof=use_oof,
        model_name=args.model_name or None,
        cv_strategy=None if args.cv_strategy == "auto" else args.cv_strategy,
        cv_folds=args.cv_folds if args.cv_folds > 0 else None,
        random_state=args.random_state,
    )
