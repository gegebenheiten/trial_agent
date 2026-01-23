from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ctg_ml_pipeline.config import DEFAULT_TABLES


@dataclass
class TableStatus:
    table: str
    has_csv: bool
    has_responses: bool
    has_evidence_jsonl: bool
    has_evidence_json: bool
    response_rows: int
    response_errors: int


@dataclass
class NctStatus:
    nct_id: str
    notebooklm_dir: Path
    table_statuses: dict[str, TableStatus]

    @property
    def missing_tables(self) -> list[str]:
        missing = []
        for name, status in self.table_statuses.items():
            if not status.has_csv:
                missing.append(name)
        return missing

    @property
    def has_all_csv(self) -> bool:
        return all(status.has_csv for status in self.table_statuses.values())

    @property
    def has_all_responses(self) -> bool:
        return all(status.has_responses for status in self.table_statuses.values())

    @property
    def total_response_errors(self) -> int:
        return sum(status.response_errors for status in self.table_statuses.values())


def _count_response_errors(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    errors = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                errors += 1
    return total, errors


def scan_notebooklm_group(group_dir: str | Path, tables: Iterable[str] = DEFAULT_TABLES) -> list[NctStatus]:
    group_path = Path(group_dir)
    results: list[NctStatus] = []

    for nct_dir in sorted(group_path.glob("NCT*")):
        nb_dir = nct_dir / "notebooklm"
        table_statuses: dict[str, TableStatus] = {}
        for table in tables:
            csv_path = nb_dir / f"{table}_notebooklm.csv"
            resp_path = nb_dir / f"{table}_notebooklm_responses.jsonl"
            evidence_jsonl_path = nb_dir / f"{table}_notebooklm_evidence.jsonl"
            evidence_json_path = nb_dir / f"{table}_notebooklm_evidence.json"
            rows, errors = _count_response_errors(resp_path)
            table_statuses[table] = TableStatus(
                table=table,
                has_csv=csv_path.exists(),
                has_responses=resp_path.exists(),
                has_evidence_jsonl=evidence_jsonl_path.exists(),
                has_evidence_json=evidence_json_path.exists(),
                response_rows=rows,
                response_errors=errors,
            )
        results.append(
            NctStatus(
                nct_id=nct_dir.name,
                notebooklm_dir=nb_dir,
                table_statuses=table_statuses,
            )
        )
    return results


def summarize_status(statuses: list[NctStatus]) -> dict[str, list[str]]:
    complete = []
    complete_clean = []
    partial = []
    for status in statuses:
        if status.has_all_csv and status.has_all_responses:
            complete.append(status.nct_id)
            if status.total_response_errors == 0:
                complete_clean.append(status.nct_id)
        else:
            partial.append(status.nct_id)
    return {
        "complete": complete,
        "complete_clean": complete_clean,
        "partial": partial,
    }
