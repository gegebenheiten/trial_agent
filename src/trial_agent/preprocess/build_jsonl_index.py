"""
Build a trial_id -> byte offset index for a JSONL corpus.

This lets the retrieval pipeline fetch specific trials without loading
the entire corpus into memory.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL offset index.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.processed_trials,
        help="Path to the JSONL corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write index JSON (defaults to <jsonl>.index.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    output = args.output or jsonl_path.with_suffix(".index.json")

    index = {}
    with jsonl_path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                record = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            trial_id = record.get("trial_id") or record.get("nct_id")
            if trial_id:
                index[str(trial_id)] = offset

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=True))
    print(f"Wrote {len(index)} offsets to {output}")


if __name__ == "__main__":
    main()
