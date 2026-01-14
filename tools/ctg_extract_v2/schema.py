from __future__ import annotations

from pathlib import Path
from typing import List

from schema_fields import TABLE_FIELDS, TABLES


def parse_tables(raw: str) -> tuple[str, ...]:
    if not raw:
        return TABLES
    wanted = [value.strip() for value in raw.split(",") if value.strip()]
    return tuple(table for table in TABLES if table in wanted)


def load_sheet_fields(_: Path, sheet_name: str) -> List[str]:
    if sheet_name not in TABLE_FIELDS:
        raise ValueError(f"Unknown table: {sheet_name}")
    return list(TABLE_FIELDS[sheet_name])
