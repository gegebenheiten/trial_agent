"""
Build a SQLite keyword index for fast structured retrieval.

Stores token -> trial_id mappings per field and per-trial token counts.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.config import settings
from trial_agent.ingest.parse_ctgov import normalize_trial
from trial_agent.retrieval.embed import tokenize
from trial_agent.retrieval.keyword_retrieve import (
    _extract_biomarkers,
    _extract_conditions,
    _extract_drugs,
    _extract_endpoints,
    _extract_phase,
    _extract_trial_type,
)


FIELDS = ("condition", "drug", "biomarker", "endpoint")


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Dict]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line


def _extract_field_tokens(trial: Dict) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {
        "condition": _extract_conditions(trial),
        "drug": _extract_drugs(trial),
        "biomarker": _extract_biomarkers(trial),
        "endpoint": _extract_endpoints(trial),
    }
    tokens: Dict[str, List[str]] = {}
    for field, vals in values.items():
        if not vals:
            continue
        token_set = set(tokenize(" ".join([v for v in vals if v])))
        if token_set:
            tokens[field] = sorted(token_set)
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite keyword index.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.processed_trials,
        help="Processed trials JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.keyword_index_path,
        help="Output SQLite index path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of trials to index (0 = all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop existing tables before indexing.",
    )
    return parser.parse_args()


def _create_schema(conn: sqlite3.Connection, overwrite: bool) -> None:
    if overwrite:
        conn.executescript(
            """
            DROP TABLE IF EXISTS token_index;
            DROP TABLE IF EXISTS token_count;
            DROP TABLE IF EXISTS trials_meta;
            """
        )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS token_index (
            field TEXT NOT NULL,
            token TEXT NOT NULL,
            trial_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_token_field ON token_index(field, token);
        CREATE INDEX IF NOT EXISTS idx_token_trial ON token_index(trial_id);

        CREATE TABLE IF NOT EXISTS token_count (
            trial_id TEXT NOT NULL,
            field TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY (trial_id, field)
        );

        CREATE TABLE IF NOT EXISTS trials_meta (
            trial_id TEXT PRIMARY KEY,
            phase TEXT,
            trial_type TEXT
        );
        """
    )


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.output))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        _create_schema(conn, args.overwrite)

        token_rows: List[Tuple[str, str, str]] = []
        count_rows: List[Tuple[str, str, int]] = []
        meta_rows: List[Tuple[str, str, str]] = []
        total = 0

        for line in _iter_jsonl(args.jsonl, args.limit):
            total += 1
            try:
                raw = json.loads(line)
            except Exception:
                continue
            trial = normalize_trial(raw)
            trial_id = str(trial.get("trial_id") or trial.get("study_id") or "").strip()
            if not trial_id:
                continue
            tokens_by_field = _extract_field_tokens(trial)
            for field in FIELDS:
                tokens = tokens_by_field.get(field, [])
                if not tokens:
                    continue
                for token in tokens:
                    token_rows.append((field, token, trial_id))
                count_rows.append((trial_id, field, len(tokens)))
            phase = _extract_phase(trial)
            trial_type = _extract_trial_type(trial)
            meta_rows.append((trial_id, phase, trial_type))

            if total % 5000 == 0:
                conn.executemany(
                    "INSERT INTO token_index(field, token, trial_id) VALUES (?, ?, ?)",
                    token_rows,
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO token_count(trial_id, field, token_count) VALUES (?, ?, ?)",
                    count_rows,
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO trials_meta(trial_id, phase, trial_type) VALUES (?, ?, ?)",
                    meta_rows,
                )
                conn.commit()
                token_rows.clear()
                count_rows.clear()
                meta_rows.clear()

        if token_rows:
            conn.executemany(
                "INSERT INTO token_index(field, token, trial_id) VALUES (?, ?, ?)",
                token_rows,
            )
        if count_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO token_count(trial_id, field, token_count) VALUES (?, ?, ?)",
                count_rows,
            )
        if meta_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO trials_meta(trial_id, phase, trial_type) VALUES (?, ?, ?)",
                meta_rows,
            )
        conn.commit()
        print(f"Indexed {total} trials into {args.output}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
