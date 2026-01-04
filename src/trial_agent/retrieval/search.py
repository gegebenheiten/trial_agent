from typing import Dict, List, Optional, Set

from trial_agent.ingest.clean_text import short_snippet
from trial_agent.retrieval.embed import jaccard_similarity, vectorize
from trial_agent.retrieval.index import trial_to_corpus_text, trial_to_field_text
from trial_agent.retrieval.relations import related_studies
from trial_agent.retrieval.simhash_index import SimHashIndex
from trial_agent.retrieval.trial_store import TrialStore
from trial_agent.retrieval.vector_store import VectorStore


def _overlap_score(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set([x.lower() for x in a])
    sb = set([x.lower() for x in b])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _endpoint_type(trial: Dict) -> str:
    endpoints = trial.get("endpoints", {})
    if isinstance(endpoints, dict):
        return (endpoints.get("parsed", {}) or {}).get("primary_type", "")
    if isinstance(endpoints, list) and endpoints:
        ep = endpoints[0]
        if isinstance(ep, dict):
            return (
                ep.get("primary_endpoint_domain")
                or ep.get("primary_endpoint_subdomain")
                or ep.get("primary_endpoint")
                or ""
            )
    return ""


def _phase(trial: Dict) -> str:
    return trial.get("phase", "") or (trial.get("study", {}) or {}).get("phase", "")


def _conditions(trial: Dict) -> List[str]:
    if "condition" in trial:
        return trial.get("condition", []) or []
    conditions = trial.get("conditions", []) or []
    if conditions and isinstance(conditions[0], dict):
        return [c.get("condition_name", "") for c in conditions if c.get("condition_name")]
    return conditions


def _split_tokens(value: str) -> List[str]:
    if not value:
        return []
    tokens = []
    for part in value.replace(";", "|").replace(",", "|").split("|"):
        part = part.strip()
        if part:
            tokens.append(part)
    return tokens


def _biomarkers(trial: Dict) -> List[str]:
    biomarkers = trial.get("biomarkers", []) or []
    if biomarkers and isinstance(biomarkers[0], dict):
        out: List[str] = []
        for biomarker in biomarkers:
            out.extend(_split_tokens(biomarker.get("biomarker_name", "")))
            out.extend(_split_tokens(biomarker.get("biomarker_genes", "")))
        return out
    return biomarkers


def _target_genes(trial: Dict) -> List[str]:
    drug_moa = trial.get("drug_moa", {}) or {}
    if isinstance(drug_moa, dict):
        return drug_moa.get("target_genes", []) or []
    if isinstance(drug_moa, list):
        out: List[str] = []
        for item in drug_moa:
            if not isinstance(item, dict):
                continue
            out.extend(_split_tokens(item.get("gene", "")))
        return out
    return []


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


def score_trial(query_trial: Dict, candidate: Dict) -> float:
    """Combine lexical and structured similarity into a single score."""
    query_vec = vectorize([trial_to_corpus_text(query_trial)])
    lexical = jaccard_similarity(query_vec, candidate["vector"])
    phase_bonus = 0.15 if _phase(query_trial) == _phase(candidate["trial"]) else 0.0
    condition_bonus = 0.4 * _overlap_score(
        _conditions(query_trial), _conditions(candidate["trial"])
    )
    endpoint_bonus = 0.2 if _endpoint_type(query_trial) == _endpoint_type(candidate["trial"]) else 0.0
    biomarker_bonus = 0.25 * _overlap_score(_biomarkers(query_trial), _biomarkers(candidate["trial"]))
    gene_bonus = 0.35 * _overlap_score(_target_genes(query_trial), _target_genes(candidate["trial"]))
    return lexical + phase_bonus + condition_bonus + endpoint_bonus + biomarker_bonus + gene_bonus


def _lexical_rank(query_trial: Dict, indexed_trials: List[Dict]) -> List[Dict]:
    scored = []
    for candidate in indexed_trials:
        score = score_trial(query_trial, candidate)
        scored.append(
            {
                "trial": candidate["trial"],
                "score": score,
                "snippet": _pick_snippet(candidate["trial"]),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _relation_rank(
    query_trial: Dict,
    indexed_trials: List[Dict],
    relation_graph: Dict[str, List[str]],
    top_k: int,
    max_hops: int,
) -> List[Dict]:
    trial_id = (query_trial.get("trial_id") or "").strip()
    if not trial_id:
        return []
    related = related_studies(trial_id, relation_graph, max_hops=max_hops, max_results=top_k)
    if not related:
        return []
    related.sort(key=lambda x: x[1])
    index_by_id = {
        entry["trial"].get("trial_id"): entry
        for entry in indexed_trials
        if entry.get("trial", {}).get("trial_id")
    }
    results: List[Dict] = []
    for related_id, hop in related:
        entry = index_by_id.get(related_id)
        if not entry:
            continue
        results.append(
            {
                "trial": entry["trial"],
                "score": round(1.0 / hop, 4),
                "snippet": _pick_snippet(entry["trial"]),
                "hop": hop,
            }
        )
        if len(results) >= top_k:
            break
    return results


def search_trials(
    query_trial: Dict,
    indexed_trials: List[Dict],
    top_k: int = 5,
    relation_graph: Optional[Dict[str, List[str]]] = None,
    max_hops: int = 3,
) -> List[Dict]:
    """Return ranked list of trials with snippets for evidence-aware prompts."""
    if relation_graph:
        trial_id = (query_trial.get("trial_id") or "").strip()
        if trial_id and f"study:{trial_id}" in relation_graph:
            return _relation_rank(
                query_trial, indexed_trials, relation_graph, top_k=top_k, max_hops=max_hops
            )
    return _lexical_rank(query_trial, indexed_trials)[:top_k]


def search_trials_simhash(
    query_trial: Dict,
    trial_store: TrialStore,
    simhash_index: SimHashIndex,
    top_k: int = 5,
    allowlist: Optional[Set[str]] = None,
) -> List[Dict]:
    query_text = trial_to_corpus_text(query_trial)
    trial_id = (query_trial.get("trial_id") or "").strip()
    candidate_k = max(top_k * 8, top_k)
    candidates = simhash_index.search(
        query_text,
        top_k=candidate_k,
        exclude_id=trial_id or None,
    )
    if not candidates:
        return []

    if allowlist:
        filtered = []
        for trial_id_candidate, score, distance in candidates:
            if trial_id_candidate in allowlist:
                filtered.append((trial_id_candidate, score, distance))
        candidates = filtered
        if not candidates:
            return []

    trial_ids = [trial_id_candidate for trial_id_candidate, _, _ in candidates]
    trial_map = trial_store.get_many(trial_ids)
    scored: List[Dict] = []
    for trial_id_candidate, _, _ in candidates:
        trial = trial_map.get(trial_id_candidate)
        if not trial:
            continue
        vector = vectorize([trial_to_corpus_text(trial)])
        score = score_trial(query_trial, {"trial": trial, "vector": vector})
        scored.append(
            {
                "trial": trial,
                "score": score,
                "snippet": _pick_snippet(trial),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def search_trials_vector(
    query_trial: Dict,
    trial_store: TrialStore,
    vector_store: VectorStore,
    top_k: int = 5,
    allowlist: Optional[Set[str]] = None,
    focus: str = "full",
) -> List[Dict]:
    query_text = trial_to_field_text(query_trial, focus)
    trial_id = (query_trial.get("trial_id") or "").strip()
    candidate_k = max(top_k * 5, top_k)
    candidates = vector_store.search(
        query_text,
        top_k=candidate_k,
        exclude_id=trial_id or None,
    )
    if not candidates:
        return []

    if allowlist:
        candidates = [(tid, score) for tid, score in candidates if tid in allowlist]
        if not candidates:
            return []

    trial_ids = [trial_id_candidate for trial_id_candidate, _ in candidates]
    trial_map = trial_store.get_many(trial_ids)
    results: List[Dict] = []
    for trial_id_candidate, score in candidates:
        trial = trial_map.get(trial_id_candidate)
        if not trial:
            continue
        results.append(
            {
                "trial": trial,
                "score": float(score),
                "snippet": _pick_snippet(trial),
            }
        )
        if len(results) >= top_k:
            break
    return results
