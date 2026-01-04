import csv
from pathlib import Path
from typing import Set


_CANDIDATE_COLUMNS = ("nctid", "nct_id", "trial_id", "study_id")


def load_trial_id_allowlist(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Allowlist CSV not found: {path}")

    ids: Set[str] = set()
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lower_map = {name.lower(): name for name in reader.fieldnames}
            column = next((lower_map[c] for c in _CANDIDATE_COLUMNS if c in lower_map), None)
            if column:
                for row in reader:
                    value = (row.get(column) or "").strip()
                    if value:
                        ids.add(value)
                return ids

    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if not value:
                continue
            if value.lower() in _CANDIDATE_COLUMNS:
                continue
            ids.add(value)

    return ids
