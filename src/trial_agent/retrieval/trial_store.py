import json
from pathlib import Path
from typing import Dict, Iterable, Optional


class TrialStore:
    def __init__(self, jsonl_path: Path, index_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.index_path = index_path
        self._index: Optional[Dict[str, int]] = None

    def _load_index(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"JSONL index not found: {self.index_path}")
        data = json.loads(self.index_path.read_text())
        self._index = {str(k): int(v) for k, v in data.items()}

    def get(self, trial_id: str) -> Optional[Dict]:
        if not trial_id:
            return None
        self._load_index()
        assert self._index is not None
        offset = self._index.get(trial_id)
        if offset is None:
            return None
        with self.jsonl_path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    def get_many(self, trial_ids: Iterable[str]) -> Dict[str, Dict]:
        self._load_index()
        assert self._index is not None
        results: Dict[str, Dict] = {}
        with self.jsonl_path.open("rb") as f:
            for trial_id in trial_ids:
                offset = self._index.get(trial_id)
                if offset is None:
                    continue
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue
                try:
                    results[trial_id] = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
        return results
