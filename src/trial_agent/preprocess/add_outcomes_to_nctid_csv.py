import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.config import settings
from trial_agent.ingest.parse_ctgov import normalize_trial
from trial_agent.retrieval.trial_store import TrialStore


ID_COLUMNS = ("nctid", "nct_id", "trial_id", "study_id")


def _detect_id_column(fieldnames: List[str]) -> str:
    lower_map = {name.lower(): name for name in fieldnames}
    for key in ID_COLUMNS:
        if key in lower_map:
            return lower_map[key]
    return fieldnames[0]


def _collect_outcomes(trial: Dict) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    outcomes = trial.get("outcomes") or []
    if isinstance(outcomes, dict):
        outcomes = [outcomes]
    if isinstance(outcomes, list):
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            entry = {
                key: outcome.get(key, "")
                for key in ("overall_status", "outcome_type", "why_terminated")
                if outcome.get(key)
            }
            if entry:
                items.append(entry)
    if items:
        return items
    summary = trial.get("outcomes_summary") or {}
    if isinstance(summary, dict):
        entry = {
            key: summary.get(key, "")
            for key in ("overall_status", "outcome_type", "why_terminated")
            if summary.get(key)
        }
        if entry:
            return [entry]
    return []


def _join_values(values: List[str]) -> str:
    seen = set()
    out: List[str] = []
    for value in values:
        v = (value or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return " | ".join(out)


def _iter_rows(input_path: Path) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {input_path}")
        return reader.fieldnames, list(reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append outcomes column to an NCTID CSV.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with NCTIDs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: add _with_outcomes suffix).",
    )
    parser.add_argument(
        "--outcome-type-column",
        type=str,
        default="outcome_type",
        help="Column name for outcome_type.",
    )
    parser.add_argument(
        "--why-terminated-column",
        type=str,
        default="why_terminated",
        help="Column name for why_terminated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    output_path = args.output_csv
    if output_path is None:
        output_path = args.input_csv.with_name(
            args.input_csv.stem + "_with_outcomes" + args.input_csv.suffix
        )

    index_path = settings.processed_trials.with_suffix(".index.json")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run build_jsonl_index.py first."
        )
    store = TrialStore(settings.processed_trials, index_path)

    fieldnames, rows = _iter_rows(args.input_csv)
    id_column = _detect_id_column(fieldnames)
    if "outcomes" in fieldnames:
        fieldnames = [name for name in fieldnames if name != "outcomes"]
    new_columns = []
    if args.outcome_type_column not in fieldnames:
        new_columns.append(args.outcome_type_column)
    if args.why_terminated_column not in fieldnames:
        new_columns.append(args.why_terminated_column)
    if new_columns:
        fieldnames = list(fieldnames) + new_columns

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "outcomes" in row:
                row.pop("outcomes", None)
            trial_id = (row.get(id_column) or "").strip()
            outcome_type_value = ""
            why_terminated_value = ""
            if trial_id:
                trial = store.get(trial_id)
                if trial:
                    trial = normalize_trial(trial)
                    outcomes = _collect_outcomes(trial)
                    if outcomes:
                        outcome_type_value = _join_values(
                            [o.get("outcome_type", "") for o in outcomes if isinstance(o, dict)]
                        )
                        why_terminated_value = _join_values(
                            [o.get("why_terminated", "") for o in outcomes if isinstance(o, dict)]
                        )
            row[args.outcome_type_column] = outcome_type_value
            row[args.why_terminated_column] = why_terminated_value
            writer.writerow(row)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
