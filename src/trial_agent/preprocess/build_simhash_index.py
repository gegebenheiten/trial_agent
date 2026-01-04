import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

from trial_agent.config import settings
from trial_agent.retrieval.embed import tokenize
from trial_agent.retrieval.index import trial_to_corpus_text
from trial_agent.retrieval.simhash_index import buckets, simhash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SimHash index for fast retrieval.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.processed_trials,
        help="JSONL corpus to index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.processed_trials.with_suffix(".simhash.sqlite"),
        help="Output SQLite file for SimHash buckets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after N records (debug).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for SQLite inserts.",
    )
    return parser.parse_args()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS simhash (
            trial_id TEXT PRIMARY KEY,
            hash TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS buckets (
            bucket INTEGER,
            trial_id TEXT,
            hash TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buckets ON buckets(bucket)")
    conn.commit()


def _iter_jsonl(path: Path, limit: int) -> Iterable[Tuple[int, dict]]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Index exists: {args.output}. Use --overwrite to rebuild."
            )
        args.output.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.output))
    _init_db(conn)
    cur = conn.cursor()

    batch_hash = []
    batch_bucket = []
    total = 0

    for idx, trial in _iter_jsonl(args.jsonl, args.limit):
        trial_id = str(trial.get("trial_id") or "").strip()
        if not trial_id:
            continue
        text = trial_to_corpus_text(trial)
        tokens = tokenize(text)
        if not tokens:
            continue
        signature = simhash(tokens)
        signature_str = str(signature)
        batch_hash.append((trial_id, signature_str))
        for bucket_id in buckets(signature):
            batch_bucket.append((bucket_id, trial_id, signature_str))
        total += 1

        if len(batch_hash) >= args.batch_size:
            cur.executemany(
                "INSERT OR REPLACE INTO simhash (trial_id, hash) VALUES (?, ?)",
                batch_hash,
            )
            cur.executemany(
                "INSERT INTO buckets (bucket, trial_id, hash) VALUES (?, ?, ?)",
                batch_bucket,
            )
            conn.commit()
            batch_hash.clear()
            batch_bucket.clear()
        if total and total % 50000 == 0:
            print(f"Indexed {total} trials...")

    if batch_hash:
        cur.executemany(
            "INSERT OR REPLACE INTO simhash (trial_id, hash) VALUES (?, ?)",
            batch_hash,
        )
        cur.executemany(
            "INSERT INTO buckets (bucket, trial_id, hash) VALUES (?, ?, ?)",
            batch_bucket,
        )
        conn.commit()

    conn.close()
    print(f"Wrote SimHash index for {total} trials to {args.output}")


if __name__ == "__main__":
    main()
