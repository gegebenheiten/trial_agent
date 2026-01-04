import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.config import settings
from trial_agent.ingest.clean_text import short_snippet
from trial_agent.ingest.parse_ctgov import normalize_trial
from trial_agent.retrieval.embed import tokenize
from trial_agent.retrieval.index import trial_to_field_text
from trial_agent.retrieval.trial_store import TrialStore


def _snippet_for_keyword(text: str, keyword: str, window: int = 180, limit: int = 360) -> str:
    if not text:
        return ""
    lower = text.lower()
    idx = lower.find(keyword.lower()) if keyword else -1
    if idx == -1:
        return short_snippet(text, limit=limit)
    left = max(0, idx - window)
    right = min(len(text), idx + len(keyword) + window)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return short_snippet(snippet, limit=limit)


def _truncate_list(values: List[str], max_items: int) -> List[str]:
    if max_items <= 0:
        return values
    return values[:max_items]


def build_match_basis(query_trial: Dict, focus_parts: List[str], max_items: int = 10) -> Dict[str, Any]:
    features = _build_query_features(query_trial, focus_parts)
    return {
        "focus_parts": focus_parts,
        "conditions": _truncate_list(_dedupe_keywords(features.get("conditions", [])), max_items),
        "drugs": _truncate_list(_dedupe_keywords(features.get("drugs", [])), max_items),
        "biomarkers": _truncate_list(_dedupe_keywords(features.get("biomarkers", [])), max_items),
        "endpoints": _truncate_list(_dedupe_keywords(features.get("endpoints", [])), max_items),
        "phase": features.get("phase", ""),
        "trial_type": features.get("trial_type", ""),
    }


def _token_set(values: List[str]) -> set:
    if not values:
        return set()
    return set(tokenize(" ".join([v for v in values if v])))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyword-based trial retrieval (no embeddings).")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.processed_trials,
        help="Processed trials JSONL.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional JSONL offset index (defaults to <jsonl>.index.json if present).",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="condition",
        help="Field(s) to search against (comma-separated).",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default="",
        help="Use features from this trial_id as keywords.",
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Keyword or phrase (repeatable).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--min-match",
        type=int,
        default=1,
        help="Minimum keyword matches required.",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=0,
        help="Limit number of extracted keywords (0 = no limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of trials scanned (0 = all).",
    )
    return parser.parse_args()


FOCUS_ALIASES = {
    "conditions": "condition",
    "endpoints": "endpoint",
    "drugs": "drug",
    "biomarkers": "biomarker",
    "all": "full",
    "default": "full",
}
ALLOWED_FOCUS = {
    "condition",
    "endpoint",
    "drug",
    "biomarker",
    "study",
    "full",
}


def _normalize_focus(value: str) -> str:
    key = value.strip().lower()
    return FOCUS_ALIASES.get(key, key)


def normalize_focus_parts(focus: str) -> List[str]:
    parts = [_normalize_focus(part) for part in (focus or "").split(",") if part.strip()]
    if not parts:
        return []
    invalid = [part for part in parts if part not in ALLOWED_FOCUS]
    if invalid:
        raise ValueError(f"Unsupported focus value(s): {', '.join(sorted(set(invalid)))}")
    return parts


def _dedupe_keywords(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if not value:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _extract_conditions(trial: Dict) -> List[str]:
    if "condition" in trial:
        return trial.get("condition", []) or []
    conditions = trial.get("conditions", []) or []
    if conditions and isinstance(conditions[0], dict):
        return [c.get("condition_name", "") for c in conditions if c.get("condition_name")]
    return conditions


def _extract_endpoints(trial: Dict) -> List[str]:
    endpoints = trial.get("endpoints", {})
    if isinstance(endpoints, dict):
        names = [ep.get("name", "") for ep in endpoints.get("primary", [])]
        parsed = endpoints.get("parsed", {}) or {}
        if parsed.get("primary_type"):
            names.append(parsed.get("primary_type", ""))
        return names
    if isinstance(endpoints, list):
        names: List[str] = []
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            names.extend(
                [
                    ep.get("primary_endpoint", ""),
                    ep.get("primary_endpoint_domain", ""),
                    ep.get("primary_endpoint_subdomain", ""),
                ]
            )
        return names
    return []


def _extract_drugs(trial: Dict) -> List[str]:
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial
    if is_trialpanorama:
        drugs = trial.get("drugs", []) or []
        if drugs and isinstance(drugs[0], dict):
            names: List[str] = []
            for drug in drugs:
                names.extend(
                    [
                        drug.get("drug_name", ""),
                        drug.get("rx_normalized_name", ""),
                        drug.get("drugbank_name", ""),
                    ]
                )
            return names
        return drugs
    names: List[str] = []
    for intervention in trial.get("interventions", []) or []:
        if not isinstance(intervention, dict):
            continue
        names.append(intervention.get("name", ""))
        names.append(intervention.get("type", ""))
    return names


def _extract_biomarkers(trial: Dict) -> List[str]:
    biomarkers = trial.get("biomarkers", []) or []
    if biomarkers and isinstance(biomarkers[0], dict):
        names: List[str] = []
        for biomarker in biomarkers:
            names.extend(
                [
                    biomarker.get("biomarker_name", ""),
                    biomarker.get("biomarker_genes", ""),
                ]
            )
        return names
    return biomarkers


def _extract_study(trial: Dict) -> List[str]:
    out: List[str] = []
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial
    if is_trialpanorama:
        study = trial.get("study", {}) or {}
        if isinstance(study, dict):
            out.extend(
                [
                    study.get("trial_type", ""),
                    study.get("recruitment_status", ""),
                    study.get("phase", ""),
                    study.get("sex", ""),
                    str(study.get("actual_accrual", "") or ""),
                    str(study.get("target_accrual", "") or ""),
                ]
            )
        return out
    design = trial.get("design", {}) or {}
    if isinstance(design, dict):
        out.extend(
            [
                design.get("allocation", ""),
                design.get("intervention_model", ""),
                design.get("masking", ""),
                design.get("primary_purpose", ""),
                design.get("dose", ""),
            ]
        )
        for arm in design.get("arms", []) or []:
            if not isinstance(arm, dict):
                continue
            out.extend([arm.get("name", ""), arm.get("description", "")])
    return out


def _extract_outcomes(trial: Dict) -> List[str]:
    outcomes = trial.get("outcomes", []) or []
    if outcomes and isinstance(outcomes[0], dict):
        out: List[str] = []
        for outcome in outcomes:
            out.extend(
                [
                    outcome.get("overall_status", ""),
                    outcome.get("outcome_type", ""),
                    outcome.get("why_terminated", ""),
                ]
            )
        return out
    outcomes_summary = trial.get("outcomes_summary", {}) or {}
    if isinstance(outcomes_summary, dict) and outcomes_summary:
        out = []
        out.extend(outcomes_summary.get("overall_status", []) or [])
        out.extend(outcomes_summary.get("outcome_type", []) or [])
        out.extend(outcomes_summary.get("why_terminated", []) or [])
        return out
    return []


def _extract_phase(trial: Dict) -> str:
    return trial.get("phase", "") or (trial.get("study", {}) or {}).get("phase", "")


def _extract_trial_type(trial: Dict) -> str:
    study = trial.get("study", {}) or {}
    return study.get("trial_type", "") or trial.get("trial_type", "")


def _extract_keywords(trial: Dict, focus: str) -> List[str]:
    focus_key = _normalize_focus(focus)
    if focus_key in {"condition", "conditions"}:
        return _extract_conditions(trial)
    if focus_key in {"endpoint", "endpoints"}:
        return _extract_endpoints(trial)
    if focus_key in {"drug", "drugs"}:
        return _extract_drugs(trial)
    if focus_key in {"biomarker", "biomarkers"}:
        return _extract_biomarkers(trial)
    if focus_key in {"study"}:
        return _extract_study(trial)
    if focus_key in {"full", "all", "default"}:
        out: List[str] = []
        out.extend(_extract_conditions(trial))
        out.extend(_extract_drugs(trial))
        out.extend(_extract_endpoints(trial))
        out.extend(_extract_biomarkers(trial))
        return out
    return []


def extract_keywords_from_trial(
    trial: Dict, focus_parts: List[str], max_keywords: int = 0
) -> List[str]:
    keywords: List[str] = []
    for focus in focus_parts:
        keywords.extend(_extract_keywords(trial, focus))
    keywords = _dedupe_keywords(keywords)
    if max_keywords and max_keywords > 0:
        keywords = keywords[:max_keywords]
    return keywords


def _load_trial_by_id(jsonl_path: Path, index_path: Path, trial_id: str) -> Dict:
    if not index_path or not index_path.exists():
        raise FileNotFoundError(f"JSONL index not found: {index_path}")
    store = TrialStore(jsonl_path, index_path)
    trial = store.get(trial_id)
    if trial:
        return normalize_trial(trial)
    return {}


def _build_query_features(trial: Dict, focus_parts: List[str]) -> Dict[str, Any]:
    use_full = "full" in focus_parts
    use_condition = use_full or "condition" in focus_parts
    use_drug = use_full or "drug" in focus_parts
    use_biomarker = use_full or "biomarker" in focus_parts
    use_endpoint = use_full or "endpoint" in focus_parts
    use_study = use_full or "study" in focus_parts
    return {
        "conditions": _extract_conditions(trial) if use_condition else [],
        "drugs": _extract_drugs(trial) if use_drug else [],
        "biomarkers": _extract_biomarkers(trial) if use_biomarker else [],
        "endpoints": _extract_endpoints(trial) if use_endpoint else [],
        "phase": _extract_phase(trial) if use_study else "",
        "trial_type": _extract_trial_type(trial) if use_study else "",
    }


def structured_retrieve_trials(
    jsonl_path: Path,
    query_trial: Dict,
    focus_parts: List[str],
    keywords: List[str],
    top_k: int = 5,
    min_match: int = 1,
    limit: int = 0,
    exclude_trial_id: str = "",
    allowlist: Optional[Set[str]] = None,
    index_path: Optional[Path] = None,
    trial_store: Optional[TrialStore] = None,
) -> Tuple[List[Dict], int]:
    if not focus_parts:
        raise ValueError("focus_parts is required")
    if not index_path or not index_path.exists():
        raise FileNotFoundError(
            f"Keyword index not found: {index_path}. Run build_keyword_index.py first."
        )
    if not trial_store:
        raise FileNotFoundError(
            "JSONL index not available. Run build_jsonl_index.py first."
        )
    return _structured_retrieve_trials_sqlite(
        index_path=index_path,
        trial_store=trial_store,
        query_trial=query_trial,
        focus_parts=focus_parts,
        keywords=keywords,
        top_k=top_k,
        min_match=min_match,
        limit=limit,
        exclude_trial_id=exclude_trial_id,
        allowlist=allowlist,
    )


def _structured_retrieve_trials_sqlite(
    index_path: Path,
    trial_store: TrialStore,
    query_trial: Dict,
    focus_parts: List[str],
    keywords: List[str],
    top_k: int = 5,
    min_match: int = 1,
    limit: int = 0,
    exclude_trial_id: str = "",
    allowlist: Optional[Set[str]] = None,
) -> Tuple[List[Dict], int]:
    query_features = _build_query_features(query_trial, focus_parts)
    q_conditions = _token_set(query_features["conditions"])
    q_drugs = _token_set(query_features["drugs"])
    q_biomarkers = _token_set(query_features["biomarkers"])
    q_endpoints = _token_set(query_features["endpoints"])
    q_phase = (query_features.get("phase") or "").strip().lower()
    q_trial_type = (query_features.get("trial_type") or "").strip().lower()

    def _overlap_map(conn: sqlite3.Connection, field: str, tokens: Set[str]) -> Dict[str, int]:
        if not tokens:
            return {}
        placeholders = ",".join(["?"] * len(tokens))
        query = (
            "SELECT trial_id, COUNT(DISTINCT token) AS overlap "
            "FROM token_index WHERE field = ? AND token IN ("
            + placeholders
            + ") GROUP BY trial_id"
        )
        rows = conn.execute(query, [field, *sorted(tokens)]).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    conn = sqlite3.connect(str(index_path))
    try:
        cond_overlap_map = _overlap_map(conn, "condition", q_conditions)
        drug_overlap_map = _overlap_map(conn, "drug", q_drugs)
        biomarker_overlap_map = _overlap_map(conn, "biomarker", q_biomarkers)
        endpoint_overlap_map = _overlap_map(conn, "endpoint", q_endpoints)

        candidate_ids: Set[str] = set()
        candidate_ids.update(cond_overlap_map)
        candidate_ids.update(drug_overlap_map)
        candidate_ids.update(biomarker_overlap_map)
        candidate_ids.update(endpoint_overlap_map)
        if exclude_trial_id:
            candidate_ids.discard(exclude_trial_id)
        if allowlist:
            candidate_ids.intersection_update(allowlist)
        if not candidate_ids:
            return [], 0

        if limit and limit > 0 and len(candidate_ids) > limit:
            candidate_ids = set(list(candidate_ids)[:limit])

        placeholders = ",".join(["?"] * len(candidate_ids))
        count_rows = conn.execute(
            "SELECT trial_id, field, token_count FROM token_count "
            f"WHERE trial_id IN ({placeholders})",
            list(candidate_ids),
        ).fetchall()
        token_counts: Dict[str, Dict[str, int]] = {}
        for trial_id, field, count in count_rows:
            token_counts.setdefault(trial_id, {})[field] = int(count)

        meta_rows = conn.execute(
            "SELECT trial_id, phase, trial_type FROM trials_meta "
            f"WHERE trial_id IN ({placeholders})",
            list(candidate_ids),
        ).fetchall()
        meta_map = {
            row[0]: {"phase": (row[1] or "").lower(), "trial_type": (row[2] or "").lower()}
            for row in meta_rows
        }
    finally:
        conn.close()

    has_conditions = bool(q_conditions)
    has_drugs = bool(q_drugs)
    has_biomarkers = bool(q_biomarkers)
    has_endpoints = bool(q_endpoints)
    has_phase = bool(q_phase)
    has_trial_type = bool(q_trial_type)

    def _levels() -> List[Dict[str, bool]]:
        if has_conditions:
            levels = [
                {
                    "require_condition": True,
                    "require_drug": has_drugs,
                    "require_biomarker": has_biomarkers,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": True,
                    "require_drug": has_drugs,
                    "require_biomarker": False,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": True,
                    "require_drug": False,
                    "require_biomarker": False,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": True,
                    "require_drug": False,
                    "require_biomarker": False,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": False,
                },
                {
                    "require_condition": True,
                    "require_drug": False,
                    "require_biomarker": False,
                    "require_endpoint": False,
                    "require_trial_type": False,
                    "require_phase": False,
                },
            ]
            if has_endpoints:
                levels.append(
                    {
                        "require_condition": False,
                        "require_drug": False,
                        "require_biomarker": False,
                        "require_endpoint": True,
                        "require_trial_type": has_trial_type,
                        "require_phase": False,
                    }
                )
            return levels
        if has_endpoints:
            return [
                {
                    "require_condition": False,
                    "require_drug": has_drugs,
                    "require_biomarker": has_biomarkers,
                    "require_endpoint": True,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": False,
                    "require_drug": has_drugs,
                    "require_biomarker": False,
                    "require_endpoint": True,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": False,
                    "require_drug": False,
                    "require_biomarker": False,
                    "require_endpoint": True,
                    "require_trial_type": has_trial_type,
                    "require_phase": False,
                },
            ]
        if has_drugs:
            return [
                {
                    "require_condition": False,
                    "require_drug": True,
                    "require_biomarker": has_biomarkers,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                },
                {
                    "require_condition": False,
                    "require_drug": True,
                    "require_biomarker": False,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": False,
                },
            ]
        if has_biomarkers:
            return [
                {
                    "require_condition": False,
                    "require_drug": False,
                    "require_biomarker": True,
                    "require_endpoint": False,
                    "require_trial_type": has_trial_type,
                    "require_phase": has_phase,
                }
            ]
        return [
            {
                "require_condition": False,
                "require_drug": False,
                "require_biomarker": False,
                "require_endpoint": False,
                "require_trial_type": False,
                "require_phase": False,
            }
        ]

    levels = _levels()
    buckets: List[List[Tuple[float, str]]] = [[] for _ in levels]
    scanned = 0
    for trial_id in candidate_ids:
        scanned += 1
        cond_overlap = cond_overlap_map.get(trial_id, 0)
        drug_overlap = drug_overlap_map.get(trial_id, 0)
        biomarker_overlap = biomarker_overlap_map.get(trial_id, 0)
        endpoint_overlap = endpoint_overlap_map.get(trial_id, 0)

        cond_match = cond_overlap > 0 if has_conditions else False
        drug_match = drug_overlap > 0 if has_drugs else False
        biomarker_match = biomarker_overlap > 0 if has_biomarkers else False
        endpoint_match = endpoint_overlap > 0 if has_endpoints else False

        meta = meta_map.get(trial_id, {})
        phase_match = bool(q_phase) and q_phase == (meta.get("phase") or "") if has_phase else False
        trial_type_match = (
            bool(q_trial_type) and q_trial_type == (meta.get("trial_type") or "")
            if has_trial_type
            else False
        )

        match_count = int(cond_match) + int(drug_match) + int(biomarker_match) + int(endpoint_match)
        if min_match and match_count < min_match:
            continue

        counts = token_counts.get(trial_id, {})
        cond_j = (
            cond_overlap / (len(q_conditions) + counts.get("condition", 0) - cond_overlap)
            if cond_overlap
            else 0.0
        )
        drug_j = (
            drug_overlap / (len(q_drugs) + counts.get("drug", 0) - drug_overlap)
            if drug_overlap
            else 0.0
        )
        biomarker_j = (
            biomarker_overlap
            / (len(q_biomarkers) + counts.get("biomarker", 0) - biomarker_overlap)
            if biomarker_overlap
            else 0.0
        )

        score = (0.45 * cond_j) + (0.25 * drug_j) + (0.20 * biomarker_j)
        if phase_match:
            score += 0.10
        if endpoint_match:
            score += 0.05

        for idx, level in enumerate(levels):
            if level["require_condition"] and not cond_match:
                continue
            if level["require_drug"] and not drug_match:
                continue
            if level["require_biomarker"] and not biomarker_match:
                continue
            if level["require_endpoint"] and not endpoint_match:
                continue
            if level["require_trial_type"] and not trial_type_match:
                continue
            if level["require_phase"] and not phase_match:
                continue
            bucket = buckets[idx]
            bucket.append((score, trial_id))
            bucket.sort(key=lambda x: x[0])
            if len(bucket) > top_k:
                bucket.pop(0)
            break

    results: List[Dict] = []
    seen: Set[str] = set()
    for bucket in buckets:
        bucket.sort(key=lambda x: x[0], reverse=True)
        for score, trial_id in bucket:
            if trial_id in seen:
                continue
            seen.add(trial_id)
            results.append({"trial_id": trial_id, "score": float(score)})
            if len(results) >= top_k:
                break
        if len(results) >= top_k:
            break

    if not results:
        return [], scanned

    trial_ids = [item["trial_id"] for item in results]
    trial_map = trial_store.get_many(trial_ids)
    final_results: List[Dict] = []
    for item in results:
        trial = trial_map.get(item["trial_id"])
        if not trial:
            continue
        trial = normalize_trial(trial)
        text_parts = []
        for focus in focus_parts:
            text_parts.append(trial_to_field_text(trial, focus))
        text = " ".join([p for p in text_parts if p])
        keyword = keywords[0] if keywords else ""
        snippet = _snippet_for_keyword(text, keyword)
        final_results.append(
            {
                "trial": trial,
                "score": float(item["score"]),
                "snippet": snippet,
            }
        )
    return final_results, scanned


def main() -> None:
    args = _parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")
    focus_parts = normalize_focus_parts(args.focus)
    if not focus_parts:
        raise ValueError("--focus is required.")
    keywords = [q.strip() for q in (args.query or []) if q.strip()]
    query_trial_id = (args.trial_id or "").strip()
    if not query_trial_id:
        raise ValueError("--trial-id is required for SQLite retrieval.")
    index_path = args.index or args.jsonl.with_suffix(".index.json")
    trial = _load_trial_by_id(args.jsonl, index_path, query_trial_id)
    if not trial:
        raise ValueError(f"trial_id not found: {query_trial_id}")
    query_trial: Optional[Dict[str, Any]] = trial
    keywords.extend(extract_keywords_from_trial(trial, focus_parts, args.max_keywords))
    keywords = _dedupe_keywords(keywords)
    if args.max_keywords and args.max_keywords > 0:
        keywords = keywords[: args.max_keywords]
    if not keywords:
        raise ValueError("No keywords found. Add --query or ensure the trial has fields.")
    trial_store = TrialStore(args.jsonl, index_path)
    keyword_index = settings.keyword_index_path if settings.keyword_index_path.exists() else None
    results, scanned = structured_retrieve_trials(
        args.jsonl,
        query_trial=query_trial,
        focus_parts=focus_parts,
        keywords=keywords,
        top_k=args.top_k,
        min_match=args.min_match,
        limit=args.limit,
        exclude_trial_id=query_trial_id,
        index_path=keyword_index,
        trial_store=trial_store,
    )
    payload = {
        "focus": ",".join(focus_parts),
        "query": keywords,
        "query_trial_id": query_trial_id or None,
        "scanned": scanned,
        "hits": len(results),
        "match_basis": build_match_basis(query_trial, focus_parts) if query_trial else None,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
