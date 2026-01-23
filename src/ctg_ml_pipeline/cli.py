from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ctg_ml_pipeline.status import scan_notebooklm_group, summarize_status
from ctg_ml_pipeline.merge import merge_group_tables, merge_group_tables_by_table
from ctg_ml_pipeline.config import ALL_TABLES
from ctg_ml_pipeline.visualize import export_summaries
from ctg_ml_pipeline.impute import impute_simple
from ctg_ml_pipeline.selection import filter_stage, embedded_stage
from ctg_ml_pipeline.modeling import train_classifier


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


def cmd_viz(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.input_csv))
    export_summaries(df, args.output_dir, exclude=args.exclude)


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

    viz = sub.add_parser("viz", help="feature missing-rate + ranges summary")
    viz.add_argument("--input-csv", required=True)
    viz.add_argument("--output-dir", required=True)
    viz.add_argument("--exclude", nargs="*", default=[])
    viz.set_defaults(func=cmd_viz)

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
    train.add_argument("--model", choices=["logistic", "rf", "gbdt"], default="logistic")
    train.add_argument("--output-json", required=True)
    train.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
