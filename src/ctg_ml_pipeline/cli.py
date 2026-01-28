from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ctg_ml_pipeline.data.status import scan_notebooklm_group, summarize_status
from ctg_ml_pipeline.data.merge import (
    merge_group_tables,
    merge_group_tables_by_table,
    backfill_notebooklm_columns,
)
from ctg_ml_pipeline.config import ALL_TABLES
from ctg_ml_pipeline.analysis.visualize import (
    export_summaries,
    plot_missingness_ranked,
    plot_missingness_ranked_by_table,
)
from ctg_ml_pipeline.preprocess.impute import impute_simple
from ctg_ml_pipeline.preprocess.selection import filter_stage, embedded_stage
from ctg_ml_pipeline.modeling.modeling import train_classifier
from ctg_ml_pipeline.modeling.tuning import tune_model, tune_models
from ctg_ml_pipeline.data.dataset import load_trial_dataset
from ctg_ml_pipeline.analysis.eda import run_eda
from ctg_ml_pipeline.data.targets import build_target_labels


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0)


def cmd_status(args: argparse.Namespace) -> None:
    statuses = scan_notebooklm_group(args.group_dir)
    summary = summarize_status(statuses)
    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))


def cmd_merge(args: argparse.Namespace) -> None:
    result = merge_group_tables(
        group_dir=args.group_dir,
        output_csv=args.output_csv,
        mode=args.mode,
        prefer_notebooklm=not args.no_notebooklm,
    )
    print(f"included={len(result.included)} skipped={len(result.skipped)}")
    if args.manifest_json:
        manifest = {"included": result.included, "skipped": result.skipped}
        Path(args.manifest_json).write_text(json.dumps(manifest, indent=2))


def _parse_tables_arg(value: str) -> list[str]:
    if not value:
        return list(ALL_TABLES)
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_merge_tables(args: argparse.Namespace) -> None:
    tables = _parse_tables_arg(args.tables)
    results = merge_group_tables_by_table(
        group_dir=args.group_dir,
        output_dir=args.output_dir,
        tables=tables,
        source=args.source,
        ensure_study_id=not args.no_study_id,
        consistent=args.consistent,
    )
    manifest = {
        table: {
            "included": result.included,
            "skipped": result.skipped,
            "rows": result.data.height,
        }
        for table, result in results.items()
    }
    if args.manifest_json:
        Path(args.manifest_json).write_text(json.dumps(manifest, indent=2))
    for table, result in results.items():
        print(f"{table}: rows={result.data.height} included={len(result.included)} skipped={len(result.skipped)}")


def cmd_backfill_group_type(args: argparse.Namespace) -> None:
    nctids = []
    if args.nctids:
        nctids.extend([item.strip().upper() for item in args.nctids.split(",") if item.strip()])
    if args.nctids_json:
        payload = json.loads(Path(args.nctids_json).read_text())
        if isinstance(payload, list):
            nctids.extend([str(item).upper() for item in payload])
        elif isinstance(payload, dict):
            # Common keys: updated / nctids
            for key in ("updated", "nctids"):
                if key in payload and isinstance(payload[key], list):
                    nctids.extend([str(item).upper() for item in payload[key]])
                    break

    result = backfill_notebooklm_columns(
        group_dir=args.group_dir,
        table="R_Arm_Study",
        columns=["Group_Type"],
        nctids=nctids or None,
        require_base_value=not args.allow_empty_base,
        treat_empty_as_missing=not args.keep_empty,
        dry_run=args.dry_run,
    )
    print(f"updated={len(result.updated)} already_present={len(result.already_present)} "
          f"missing_notebooklm={len(result.missing_notebooklm)} missing_base={len(result.missing_base)} "
          f"needs_reextract={len(result.needs_reextract)} still_missing={len(result.still_missing)}")
    if args.output_json:
        payload = {
            "table": result.table,
            "columns": result.columns,
            "updated": result.updated,
            "already_present": result.already_present,
            "missing_notebooklm": result.missing_notebooklm,
            "missing_base": result.missing_base,
            "needs_reextract": result.needs_reextract,
            "still_missing": result.still_missing,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))


def cmd_viz(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    export_summaries(df, args.output_dir, exclude=args.exclude)


def cmd_missingness(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    labeled_ids = set()
    if args.labels_csv:
        from ctg_ml_pipeline.analysis.visualize import _load_labeled_ids
        labeled_ids = _load_labeled_ids(args.labels_csv)
    plot_missingness_ranked(
        df,
        output_path=args.output_png,
        exclude=args.exclude,
        top_n=None,
        labeled_ids=labeled_ids or None,
    )
    print(f"Wrote: {args.output_png}")


def cmd_missingness_tables(args: argparse.Namespace) -> None:
    labeled_ids = set()
    if args.labels_csv:
        from ctg_ml_pipeline.analysis.visualize import _load_labeled_ids
        labeled_ids = _load_labeled_ids(args.labels_csv)
    plot_missingness_ranked_by_table(
        tables_dir=args.tables_dir,
        output_dir=args.output_dir,
        exclude=args.exclude,
        top_n=None,
        labeled_ids=labeled_ids or None,
    )
    print(f"Wrote missingness plots to: {args.output_dir}")


def cmd_impute(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    imputed = impute_simple(
        df,
        exclude=args.exclude,
        numeric_strategy=args.numeric_strategy,
        categorical_strategy=args.categorical_strategy,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imputed.write_csv(output_path)


def cmd_select(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    if args.stage == "filter":
        result = filter_stage(df, target=args.target, method=args.method, top_ratio=args.top_ratio)
    else:
        result = embedded_stage(df, target=args.target, method=args.method, top_ratio=args.top_ratio)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"selected": result.selected_features, "scores": result.scores}
    output_path.write_text(json.dumps(payload, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    result = train_classifier(df, target=args.target, model_name=args.model)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.metrics, indent=2))
    print(json.dumps(result.metrics, indent=2))


def cmd_eda(args: argparse.Namespace) -> None:
    """Run comprehensive EDA analysis on merged tables."""
    timepoint_excel = args.timepoint_excel if args.timepoint_excel else None
    report = run_eda(
        tables_dir=args.tables_dir,
        timepoint_excel=timepoint_excel,
        output_dir=args.output_dir,
    )
    # Print summary
    print("\n=== EDA Summary ===")
    print(f"Tables analyzed: {len(report.table_stats)}")
    for name, ts in report.table_stats.items():
        print(f"  {name}: {ts.row_count} rows, {ts.col_count} cols, "
              f"{len(ts.constant_columns)} constant, {len(ts.high_missing_columns)} high-missing")
    print(f"\nStudyID total: {report.cross_table_stats.study_id_total}")
    print(f"T0 features: {len(report.leakage_report.t0_features)}")
    print(f"T1 features: {len(report.leakage_report.t1_features)}")
    print(f"T2 features (exclude): {len(report.leakage_report.t2_features)}")
    print(f"Unknown features: {len(report.leakage_report.unknown_features)}")


def cmd_build_labels(args: argparse.Namespace) -> None:
    build_target_labels(
        excel_path=args.excel_path,
        missing_map_path=args.missing_map,
        output_csv=args.output_csv,
        sheet_name=args.sheet_name or None,
    )
    print(f"Wrote: {args.output_csv}")


def cmd_tune(args: argparse.Namespace) -> None:
    dataset = load_trial_dataset(
        group_dir=args.group_dir,
        target_csv=args.target_csv,
        max_missing_rate=args.max_missing_rate,
        time_split=not args.no_time_split,
        test_size=args.test_size,
        include_text=args.include_text,
        categorical_encoding=args.categorical_encoding,
        impute_strategy=args.impute_strategy,
        scale_features=not args.no_scale,
    )
    if args.models.lower() == "all":
        models = ["logistic", "lasso", "rf", "gbdt", "svm", "xgb", "lgbm"]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {}
    errors = {}
    for model_name in models:
        try:
            result = tune_model(
                dataset,
                model_name=model_name,
                n_trials=args.n_trials,
                cv_folds=args.cv_folds,
                cv_strategy=args.cv_strategy,
                cv_scoring=args.cv_scoring or None,
                timeout=None if args.timeout == 0 else args.timeout,
            )
            results[model_name] = result.__dict__
        except Exception as exc:
            errors[model_name] = str(exc)

    payload = {"results": results, "errors": errors}
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTG NotebookLM ML pipeline (framework)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="scan NotebookLM outputs and report completeness")
    status.add_argument("--group-dir", required=True)
    status.add_argument("--output-json", default="")
    status.set_defaults(func=cmd_status)

    merge = sub.add_parser("merge", help="merge NotebookLM tables into a single dataset")
    merge.add_argument("--group-dir", required=True)
    merge.add_argument("--mode", choices=["study", "arm"], default="study")
    merge.add_argument("--output-csv", required=True)
    merge.add_argument("--manifest-json", default="")
    merge.add_argument("--no-notebooklm", action="store_true")
    merge.set_defaults(func=cmd_merge)

    merge_tables = sub.add_parser(
        "merge-tables",
        help="merge trials per-table (no join across tables)",
    )
    merge_tables.add_argument("--group-dir", required=True)
    merge_tables.add_argument("--tables", default="", help="Comma-separated table names (default: all 7)")
    merge_tables.add_argument("--output-dir", required=True)
    merge_tables.add_argument(
        "--source",
        choices=["auto", "notebooklm", "base"],
        default="auto",
        help="Which CSV to use per trial table",
    )
    merge_tables.add_argument("--manifest-json", default="")
    merge_tables.add_argument("--no-study-id", action="store_true")
    merge_tables.add_argument(
        "--consistent",
        action="store_true",
        help="Only merge trials with complete notebooklm extraction (all 7 tables same trials)",
    )
    merge_tables.set_defaults(func=cmd_merge_tables)

    backfill = sub.add_parser("backfill-group-type", help="backfill Group_Type in NotebookLM R_Arm_Study")
    backfill.add_argument("--group-dir", required=True)
    backfill.add_argument("--nctids", default="", help="Comma-separated NCT IDs to target (optional)")
    backfill.add_argument("--nctids-json", default="", help="JSON file with list of NCT IDs (optional)")
    backfill.add_argument("--allow-empty-base", action="store_true", help="also backfill when base has only empty values")
    backfill.add_argument("--keep-empty", action="store_true", help="treat empty Group_Type as present (no backfill)")
    backfill.add_argument("--dry-run", action="store_true")
    backfill.add_argument("--output-json", default="")
    backfill.set_defaults(func=cmd_backfill_group_type)

    viz = sub.add_parser("viz", help="feature missing-rate + ranges summary")
    viz.add_argument("--input-csv", required=True)
    viz.add_argument("--output-dir", required=True)
    viz.add_argument("--exclude", nargs="*", default=[])
    viz.set_defaults(func=cmd_viz)

    missing = sub.add_parser("missingness", help="plot features ranked by missingness")
    missing.add_argument("--input-csv", required=True)
    missing.add_argument("--output-png", required=True)
    missing.add_argument("--labels-csv", default="", help="Optional labels CSV to filter StudyID")
    missing.add_argument("--exclude", nargs="*", default=[])
    missing.set_defaults(func=cmd_missingness)

    missing_tables = sub.add_parser("missingness-tables", help="plot missingness ranking per table")
    missing_tables.add_argument("--tables-dir", required=True, help="Directory of merged table CSVs")
    missing_tables.add_argument("--output-dir", required=True, help="Output directory for PNGs")
    missing_tables.add_argument("--labels-csv", default="", help="Optional labels CSV to filter StudyID")
    missing_tables.add_argument("--exclude", nargs="*", default=[])
    missing_tables.set_defaults(func=cmd_missingness_tables)

    impute = sub.add_parser("impute", help="simple imputation for missing values")
    impute.add_argument("--input-csv", required=True)
    impute.add_argument("--output-csv", required=True)
    impute.add_argument("--exclude", nargs="*", default=[])
    impute.add_argument("--numeric-strategy", choices=["median", "mean"], default="median")
    impute.add_argument("--categorical-strategy", choices=["mode"], default="mode")
    impute.set_defaults(func=cmd_impute)

    select = sub.add_parser("select", help="two-stage feature selection (filter/embedded)")
    select.add_argument("--input-csv", required=True)
    select.add_argument("--target", required=True)
    select.add_argument("--stage", choices=["filter", "embedded"], required=True)
    select.add_argument("--method", required=True)
    select.add_argument("--top-ratio", type=float, default=0.2)
    select.add_argument("--output-json", required=True)
    select.set_defaults(func=cmd_select)

    train = sub.add_parser("train", help="train baseline classifier and report metrics")
    train.add_argument("--input-csv", required=True)
    train.add_argument("--target", required=True)
    train.add_argument("--model", choices=["logistic", "lasso", "rf", "gbdt"], default="logistic")
    train.add_argument("--output-json", required=True)
    train.set_defaults(func=cmd_train)

    eda = sub.add_parser("eda", help="comprehensive EDA: schema, missing, keys, cross-table, leakage")
    eda.add_argument("--tables-dir", required=True, help="Directory containing merged table CSVs")
    eda.add_argument("--output-dir", required=True, help="Directory for EDA report outputs")
    eda.add_argument("--timepoint-excel", default="", help="Excel with Variable->Availability_Timepoint mapping")
    eda.set_defaults(func=cmd_eda)

    labels = sub.add_parser("build-labels", help="build target_labels.csv from Excel + Missing_map.csv")
    labels.add_argument("--excel-path", required=True, help="Path to NSCLC_Trialpanorama.xlsx")
    labels.add_argument("--missing-map", required=True, help="Path to Missing_map.csv")
    labels.add_argument("--output-csv", required=True, help="Output target_labels.csv path")
    labels.add_argument("--sheet-name", default="", help="Excel sheet name (optional)")
    labels.set_defaults(func=cmd_build_labels)

    tune = sub.add_parser("tune", help="optuna hyperparameter tuning")
    tune.add_argument("--group-dir", required=True)
    tune.add_argument("--target-csv", required=True)
    tune.add_argument("--models", default="logistic,lasso,rf,gbdt", help="Comma-separated list or 'all'")
    tune.add_argument("--n-trials", type=int, default=50)
    tune.add_argument("--cv-folds", type=int, default=5)
    tune.add_argument("--cv-strategy", choices=["kfold", "loo"], default="kfold")
    tune.add_argument("--cv-scoring", default="", help="sklearn scoring (default: roc_auc or accuracy for loo)")
    tune.add_argument("--timeout", type=int, default=0, help="Optuna timeout seconds (0 = no limit)")
    tune.add_argument("--max-missing-rate", type=float, default=0.5)
    tune.add_argument("--test-size", type=float, default=0.2)
    tune.add_argument("--no-time-split", action="store_true")
    tune.add_argument("--include-text", action="store_true")
    tune.add_argument("--categorical-encoding", choices=["label", "onehot"], default="label")
    tune.add_argument("--impute-strategy", choices=["median", "mean", "zero", "none"], default="median")
    tune.add_argument("--no-scale", action="store_true")
    tune.add_argument("--output-json", default="")
    tune.set_defaults(func=cmd_tune)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
