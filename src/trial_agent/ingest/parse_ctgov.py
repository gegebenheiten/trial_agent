import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

from trial_agent.ingest.clean_text import normalize_whitespace


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a newline-delimited JSON file."""
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Persist records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_trial(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Align raw CT.gov style JSON into the minimal schema expected by the agent.
    Missing fields are filled with safe defaults to keep the pipeline robust.
    """
    trial = deepcopy(raw)
    if trial.get("source") == "trialpanorama" or "study" in trial:
        trial.setdefault("trial_id", trial.get("study_id", trial.get("nct_id", "unknown")))
        return trial
    trial.setdefault("trial_id", trial.get("nct_id", "unknown"))
    trial.setdefault("condition", [])
    trial.setdefault("phase", trial.get("study_phase", ""))
    trial.setdefault("interventions", [])

    design = trial.setdefault("design", {})
    design.setdefault("allocation", "")
    design.setdefault("intervention_model", "")
    design.setdefault("masking", "")
    design.setdefault("primary_purpose", "")
    design.setdefault("arms", [])
    design.setdefault("dose", "")

    criteria = trial.setdefault("criteria", {})
    criteria.setdefault("inclusion_text", "")
    criteria.setdefault("exclusion_text", "")
    parsed = criteria.setdefault("parsed", {})
    parsed.setdefault("age_min", None)
    parsed.setdefault("age_max", None)
    parsed.setdefault("ecog_max", None)
    parsed.setdefault("prior_lines_max", None)
    parsed.setdefault("key_flags", [])

    endpoints = trial.setdefault("endpoints", {})
    endpoints.setdefault("primary", [])
    endpoints.setdefault("secondary", [])
    ep_parsed = endpoints.setdefault("parsed", {})
    ep_parsed.setdefault("primary_type", "")

    outcome = trial.setdefault("outcome_label", {})
    outcome.setdefault("status", "unknown")
    outcome.setdefault("source", "manual")
    outcome.setdefault("notes", "")

    # Light cleanup for long texts.
    criteria["inclusion_text"] = normalize_whitespace(criteria["inclusion_text"])
    criteria["exclusion_text"] = normalize_whitespace(criteria["exclusion_text"])
    for ep in endpoints["primary"]:
        ep["name"] = normalize_whitespace(ep.get("name", ""))
        ep["description"] = normalize_whitespace(ep.get("description", ""))
        ep["time_frame"] = normalize_whitespace(ep.get("time_frame", ""))
    for ep in endpoints["secondary"]:
        ep["name"] = normalize_whitespace(ep.get("name", ""))
        ep["description"] = normalize_whitespace(ep.get("description", ""))
        ep["time_frame"] = normalize_whitespace(ep.get("time_frame", ""))
    return trial


def normalize_trials(raw_trials: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_trial(t) for t in raw_trials]
