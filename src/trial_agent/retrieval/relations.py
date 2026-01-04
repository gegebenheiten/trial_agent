import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from trial_agent.ingest.clean_text import short_snippet
from trial_agent.retrieval.trial_store import TrialStore


def load_relation_graph(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Relation graph not found: {path}")
    return json.loads(path.read_text())


def related_studies(
    trial_id: str,
    graph: Dict[str, List[str]],
    max_hops: int,
    max_results: int,
) -> List[Tuple[str, int]]:
    if not trial_id:
        return []
    start = f"study:{trial_id}"
    if start not in graph:
        return []

    visited = {start}
    queue: Deque[Tuple[str, int]] = deque([(start, 0)])
    found: List[Tuple[str, int]] = []

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in graph.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            next_depth = depth + 1
            if neighbor.startswith("study:") and neighbor != start:
                found.append((neighbor.split(":", 1)[1], next_depth))
                if len(found) >= max_results and next_depth == 1:
                    # Enough direct neighbors; no need to expand further.
                    return found
            queue.append((neighbor, next_depth))
    return found


def _pick_snippet(trial: Dict) -> str:
    endpoints = trial.get("endpoints", {})
    if isinstance(endpoints, dict):
        primary = endpoints.get("primary", [])
        if primary:
            ep = primary[0]
            return short_snippet(
                ep.get("description")
                or ep.get("name", "")
                or trial.get("design", {}).get("arms", [{}])[0].get("description", "")
            )
    if isinstance(endpoints, list) and endpoints:
        ep = endpoints[0]
        if isinstance(ep, dict):
            return short_snippet(
                ep.get("primary_endpoint")
                or ep.get("primary_endpoint_domain", "")
                or ep.get("primary_endpoint_subdomain", "")
            )

    criteria_texts = [
        trial.get("criteria", {}).get("inclusion_text", ""),
        trial.get("criteria", {}).get("exclusion_text", ""),
    ]
    for text in criteria_texts:
        if text:
            return short_snippet(text)

    study = trial.get("study", {}) or {}
    if study.get("abstract") or study.get("title"):
        return short_snippet(study.get("abstract") or study.get("title"))

    outcomes = trial.get("outcomes", []) or []
    if outcomes and isinstance(outcomes[0], dict):
        return short_snippet(
            outcomes[0].get("outcome_type", "")
            or outcomes[0].get("why_terminated", "")
            or outcomes[0].get("overall_status", "")
        )

    return short_snippet(trial.get("design", {}).get("primary_purpose", ""))


def retrieve_related_trials(
    trial_id: str,
    graph: Dict[str, List[str]],
    store: TrialStore,
    top_k: int,
    max_hops: int,
) -> List[Dict]:
    related = related_studies(trial_id, graph, max_hops=max_hops, max_results=top_k)
    if not related:
        return []
    trial_ids = [item[0] for item in related]
    trial_map = store.get_many(trial_ids)
    results: List[Dict] = []
    for related_id, hop in related:
        trial = trial_map.get(related_id)
        if not trial:
            continue
        results.append(
            {
                "trial": trial,
                "score": round(1.0 / hop, 4),
                "snippet": _pick_snippet(trial),
                "hop": hop,
            }
        )
        if len(results) >= top_k:
            break
    return results
