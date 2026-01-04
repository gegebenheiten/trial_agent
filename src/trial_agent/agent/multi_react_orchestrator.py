import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.agent.prompts import FOCUS_GUIDE
from trial_agent.config import settings
from trial_agent.ingest.clean_text import short_snippet
from trial_agent.ingest.parse_ctgov import load_jsonl, normalize_trial
from trial_agent.knowledge.drugbank import load_drugbank_index, lookup as drugbank_lookup
from trial_agent.formatting import compact_trial_for_prompt
from trial_agent.llm import DifyClient
from trial_agent.retrieval.keyword_retrieve import (
    build_match_basis,
    extract_keywords_from_trial,
    normalize_focus_parts,
    structured_retrieve_trials,
)
from trial_agent.retrieval.index import build_in_memory_index
from trial_agent.retrieval.relations import load_relation_graph, retrieve_related_trials
from trial_agent.retrieval.search import search_trials, search_trials_simhash, search_trials_vector
from trial_agent.retrieval.simhash_index import SimHashIndex
from trial_agent.retrieval.trial_store import TrialStore
from trial_agent.retrieval.vector_store import (
    VectorStore,
    resolve_vector_paths,
    vector_index_available,
)


@dataclass
class MultiAgentState:
    trial: Dict[str, Any]
    corpus: List[Dict[str, Any]]
    index: List[Dict[str, Any]]
    trial_by_id: Dict[str, Dict[str, Any]]
    search_blob_by_id: Dict[str, str]
    relation_graph: Optional[Dict[str, List[str]]] = None
    trial_store: Optional[TrialStore] = None
    simhash_index: Optional[SimHashIndex] = None
    vector_stores: Dict[str, VectorStore] = field(default_factory=dict)
    allowlist_path: Optional[Path] = None
    relation_only: bool = False

    active_agent: Optional[str] = None
    retrieved: Optional[List[Dict[str, Any]]] = None
    retrieved_by_agent: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    drug_biomarker: Optional[Dict[str, Any]] = None
    design_analysis: Optional[Dict[str, Any]] = None
    outcome_summary: Optional[Dict[str, Any]] = None

    traces: Dict[str, Any] = field(default_factory=dict)


def _read_trial_ids_from_csv(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    candidates = ("nctid", "nct_id", "trial_id", "study_id")
    ids: List[str] = []
    seen = set()
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lower_map = {name.lower(): name for name in reader.fieldnames}
            column = next((lower_map[c] for c in candidates if c in lower_map), None)
            if column:
                for row in reader:
                    value = (row.get(column) or "").strip()
                    if value and value not in seen:
                        ids.append(value)
                        seen.add(value)
                return ids
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value and value.lower() not in candidates and value not in seen:
                ids.append(value)
                seen.add(value)
    return ids


def _read_trial_ids_from_excel(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input Excel not found: {path}")
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to read .xlsx files") from exc
    candidates = ("nctid", "nct_id", "trial_id", "study_id")
    df = pd.read_excel(path)
    if df.empty:
        return []
    lower_map = {str(name).lower(): name for name in df.columns}
    column = next((lower_map[c] for c in candidates if c in lower_map), None)
    if column is None:
        column = df.columns[0]
    ids = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    seen = set()
    out: List[str] = []
    for value in ids:
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _load_trial_from_corpus(trial_id: str) -> Dict[str, Any]:
    index_path = settings.processed_trials.with_suffix(".index.json")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run build_jsonl_index.py first."
        )
    store = TrialStore(settings.processed_trials, index_path)
    trial = store.get(trial_id)
    if not trial:
        raise KeyError(f"trial_id not found in corpus: {trial_id}")
    return trial


def load_corpus(path: Path, allowlist_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed trial corpus not found: {path}")
    trials = [normalize_trial(t) for t in load_jsonl(path)]
    if allowlist_path and allowlist_path.exists():
        from trial_agent.ingest.allowlist import load_trial_id_allowlist

        allowlist = load_trial_id_allowlist(allowlist_path)
        if allowlist:
            trials = [t for t in trials if t.get("trial_id") in allowlist]
    return trials


def _get_vector_store(state: MultiAgentState, focus: str) -> Tuple[Optional[VectorStore], str]:
    focus_key = (focus or "full").strip().lower()
    if focus_key in state.vector_stores:
        return state.vector_stores[focus_key], focus_key
    index_path, ids_path = resolve_vector_paths(focus_key)
    if not index_path.exists() or not ids_path.exists():
        if focus_key != "full":
            index_path, ids_path = resolve_vector_paths("full")
            if not index_path.exists() or not ids_path.exists():
                return None, focus_key
            focus_key = "full"
        else:
            return None, focus_key
    store = VectorStore(
        index_path,
        ids_path,
        model_name=settings.embedding_model_name,
        trust_remote_code=settings.embedding_trust_remote_code,
    )
    state.vector_stores[focus_key] = store
    return store, focus_key


def _trial_blob(trial: Dict[str, Any], text_limit: int = 2000) -> str:
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial
    if is_trialpanorama:
        conditions = " ".join(
            [
                c.get("condition_name", "")
                for c in (trial.get("conditions", []) or [])
                if isinstance(c, dict) and c.get("condition_name")
            ]
        )
        interventions = " ".join(
            [
                d.get("drug_name", "")
                for d in (trial.get("drugs", []) or [])
                if isinstance(d, dict) and d.get("drug_name")
            ]
        )
        endpoints = " ".join(
            [
                ep.get("primary_endpoint", "")
                for ep in (trial.get("endpoints", []) or [])
                if isinstance(ep, dict) and ep.get("primary_endpoint")
            ]
        )
        outcomes = " ".join(
            [
                o.get("outcome_type", "")
                for o in (trial.get("outcomes", []) or [])
                if isinstance(o, dict) and o.get("outcome_type")
            ]
        )
        study = trial.get("study", {}) or {}
        abstract = study.get("abstract", "") or study.get("title", "")
        text = "\n".join([conditions, interventions, endpoints, outcomes, abstract])
        return text.lower()

    conditions = " ".join(trial.get("condition", []) or [])
    interventions = " ".join(
        [(iv.get("name") or "") for iv in (trial.get("interventions", []) or []) if iv.get("name")]
    )
    endpoints = trial.get("endpoints", {}) or {}
    primary_names = " ".join([(ep.get("name") or "") for ep in (endpoints.get("primary", []) or [])])
    criteria = trial.get("criteria", {}) or {}
    inc = criteria.get("inclusion_text", "") or ""
    exc = criteria.get("exclusion_text", "") or ""
    crit = (inc + "\n" + exc)[:text_limit]
    design = trial.get("design", {}) or {}
    arms = " ".join([(a.get("name") or "") for a in (design.get("arms", []) or [])])
    text = "\n".join([conditions, interventions, primary_names, arms, crit])
    return text.lower()


def _format_retrieved(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in retrieved:
        trial = item["trial"]
        payload.append(
            {
                "trial_id": trial.get("trial_id"),
                "score": round(item.get("score", 0.0), 3),
                "relation_hop": item.get("hop"),
                "snippet": item.get("snippet", ""),
                "trial_compact": compact_trial_for_prompt(trial, include_outcomes=True),
            }
        )
    return payload


def tool_retrieve_similar_trials(state: MultiAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    top_k = int(args.get("top_k", settings.default_top_k))
    max_hops = int(args.get("max_hops", 3))
    focus = str(args.get("focus") or "full").strip().lower()
    trial_id = (state.trial.get("trial_id") or "").strip()

    if state.trial_store and state.relation_graph and trial_id:
        if f"study:{trial_id}" in state.relation_graph:
            retrieved = retrieve_related_trials(
                trial_id=trial_id,
                graph=state.relation_graph,
                store=state.trial_store,
                top_k=top_k,
                max_hops=max_hops,
            )
            for item in retrieved:
                item["trial"] = normalize_trial(item["trial"])
            state.retrieved = retrieved
            return {"retrieved_trials": _format_retrieved(retrieved)}

    if state.relation_only:
        state.retrieved = []
        return {
            "retrieved_trials": [],
            "warning": "relation_only_no_match",
        }

    vector_store, focus_used = _get_vector_store(state, focus)
    if state.trial_store and vector_store:
        allowlist = None
        if state.allowlist_path and state.allowlist_path.exists():
            from trial_agent.ingest.allowlist import load_trial_id_allowlist

            allowlist = load_trial_id_allowlist(state.allowlist_path)
        try:
            retrieved = search_trials_vector(
                state.trial,
                trial_store=state.trial_store,
                vector_store=vector_store,
                top_k=top_k,
                allowlist=allowlist,
                focus=focus_used,
            )
        except Exception:
            retrieved = []
        if retrieved:
            state.retrieved = retrieved
            return {"retrieved_trials": _format_retrieved(retrieved)}

    if state.trial_store and state.simhash_index:
        allowlist = None
        if state.allowlist_path and state.allowlist_path.exists():
            from trial_agent.ingest.allowlist import load_trial_id_allowlist

            allowlist = load_trial_id_allowlist(state.allowlist_path)
        retrieved = search_trials_simhash(
            state.trial,
            trial_store=state.trial_store,
            simhash_index=state.simhash_index,
            top_k=top_k,
            allowlist=allowlist,
        )
        state.retrieved = retrieved
        return {"retrieved_trials": _format_retrieved(retrieved)}

    if not state.index:
        allowlist_path = state.allowlist_path if state.allowlist_path else None
        state.corpus = load_corpus(settings.processed_trials, allowlist_path)
        state.index = build_in_memory_index(state.corpus)
        state.trial_by_id = {t.get("trial_id", ""): t for t in state.corpus if t.get("trial_id")}
        state.search_blob_by_id = {tid: _trial_blob(t) for tid, t in state.trial_by_id.items()}

    retrieved = search_trials(
        state.trial,
        state.index,
        top_k=top_k,
        relation_graph=state.relation_graph,
        max_hops=max_hops,
    )
    state.retrieved = retrieved
    return {"retrieved_trials": _format_retrieved(retrieved)}


def tool_retrieve_keyword_trials(state: MultiAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    top_k = int(args.get("top_k", settings.default_top_k))
    min_match = int(args.get("min_match", 1))
    max_keywords = int(args.get("max_keywords", 0))
    limit = int(args.get("limit", 0))
    focus = str(args.get("focus") or "condition").strip().lower()
    try:
        focus_parts = normalize_focus_parts(focus)
    except ValueError as exc:
        return {"error": str(exc)}
    if not focus_parts:
        return {"error": "focus is required"}
    keywords = extract_keywords_from_trial(state.trial, focus_parts, max_keywords=max_keywords)
    if not keywords:
        return {"error": "no_keywords_from_trial"}
    allowlist = None
    if state.allowlist_path and state.allowlist_path.exists():
        from trial_agent.ingest.allowlist import load_trial_id_allowlist

        allowlist = load_trial_id_allowlist(state.allowlist_path)
    trial_id = (state.trial.get("trial_id") or "").strip()
    try:
        retrieved, scanned = structured_retrieve_trials(
            settings.processed_trials,
            query_trial=state.trial,
            focus_parts=focus_parts,
            keywords=keywords,
            top_k=top_k,
            min_match=min_match,
            limit=limit,
            exclude_trial_id=trial_id,
            allowlist=allowlist,
            index_path=settings.keyword_index_path,
            trial_store=state.trial_store,
        )
    except Exception as exc:
        return {"error": str(exc)}
    state.retrieved = retrieved
    agent_key = state.active_agent or "default"
    state.retrieved_by_agent[agent_key] = retrieved
    match_basis = build_match_basis(state.trial, focus_parts)
    return {
        "retrieved_trials": _format_retrieved(retrieved),
        "keywords_used": keywords,
        "scanned": scanned,
        "match_basis": match_basis,
    }


def tool_get_trial_details(state: MultiAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    trial_id = (args.get("trial_id") or "").strip()
    if not trial_id:
        return {"error": "trial_id is required"}
    trial = state.trial_by_id.get(trial_id)
    if not trial and state.trial_store:
        trial = state.trial_store.get(trial_id)
        if trial:
            trial = normalize_trial(trial)
    if not trial:
        return {"error": f"trial_id not found: {trial_id}"}

    # Keep outputs compact by default.
    limit = int(args.get("text_limit", 1200))
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial
    if is_trialpanorama:
        study = dict(trial.get("study", {}) or {})
        if study.get("abstract"):
            study["abstract"] = short_snippet(study.get("abstract", ""), limit=limit)
        if study.get("title"):
            study["title"] = short_snippet(study.get("title", ""), limit=360)

        def trim_list(items: List[Dict], fields: List[str]) -> List[Dict]:
            out: List[Dict] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                trimmed = dict(item)
                for field in fields:
                    if field in trimmed and isinstance(trimmed[field], str):
                        trimmed[field] = short_snippet(trimmed[field], limit=limit)
                out.append(trimmed)
            return out

        return {
            "trial": {
                "trial_id": trial.get("trial_id"),
                "study": study,
                "conditions": trial.get("conditions", []),
                "drugs": trim_list(trial.get("drugs", []) or [], ["drug_description"]),
                "biomarkers": trial.get("biomarkers", []),
                "endpoints": trial.get("endpoints", []),
                "outcomes": trial.get("outcomes", []),
                "results": trim_list(
                    trial.get("results", []) or [],
                    ["population", "interventions", "outcomes"],
                ),
                "disposition": trim_list(
                    trial.get("disposition", []) or [],
                    ["intervention_description"],
                ),
                "adverse_events": trim_list(
                    trial.get("adverse_events", []) or [],
                    ["adverse_event_description"],
                ),
                "drug_moa": trial.get("drug_moa", []),
            }
        }

    criteria = trial.get("criteria", {}) or {}
    inclusion = criteria.get("inclusion_text", "")
    exclusion = criteria.get("exclusion_text", "")
    endpoints = trial.get("endpoints", {}) or {}
    design = trial.get("design", {}) or {}

    return {
        "trial": {
            "trial_id": trial.get("trial_id"),
            "condition": trial.get("condition", []),
            "phase": trial.get("phase", ""),
            "interventions": trial.get("interventions", []),
            "design": {
                "allocation": design.get("allocation", ""),
                "intervention_model": design.get("intervention_model", ""),
                "masking": design.get("masking", ""),
                "primary_purpose": design.get("primary_purpose", ""),
                "arms": [
                    {
                        "name": arm.get("name", ""),
                        "description": short_snippet(arm.get("description", ""), limit=320),
                    }
                    for arm in (design.get("arms", []) or [])
                ],
                "dose": design.get("dose", ""),
            },
            "criteria": {
                "inclusion_text": short_snippet(inclusion, limit=limit),
                "exclusion_text": short_snippet(exclusion, limit=limit),
                "parsed": criteria.get("parsed", {}),
            },
            "endpoints": {
                "primary": [
                    {
                        "name": ep.get("name", ""),
                        "time_frame": ep.get("time_frame", ""),
                        "description": short_snippet(ep.get("description", ""), limit=360),
                    }
                    for ep in (endpoints.get("primary", []) or [])
                ],
                "secondary": [
                    {
                        "name": ep.get("name", ""),
                        "time_frame": ep.get("time_frame", ""),
                        "description": short_snippet(ep.get("description", ""), limit=240),
                    }
                    for ep in (endpoints.get("secondary", []) or [])
                ],
                "parsed": endpoints.get("parsed", {}),
            },
        }
    }


def tool_keyword_search(state: MultiAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    keyword = (args.get("keyword") or "").strip().lower()
    if not keyword:
        return {"error": "keyword is required"}
    top_k = int(args.get("top_k", 5))

    if not state.search_blob_by_id:
        return {"error": "keyword_search_unavailable_in_fast_mode"}

    results: List[Dict[str, Any]] = []
    for trial_id, blob in state.search_blob_by_id.items():
        if keyword not in blob:
            continue
        # Cheap relevance proxy: count matches.
        count = blob.count(keyword)
        results.append({"trial_id": trial_id, "match_count": count})

    results.sort(key=lambda x: x["match_count"], reverse=True)
    out: List[Dict[str, Any]] = []
    for hit in results[:top_k]:
        trial = state.trial_by_id.get(hit["trial_id"], {})
        criteria = trial.get("criteria", {}) or {}
        inclusion = criteria.get("inclusion_text", "") or ""
        exclusion = criteria.get("exclusion_text", "") or ""
        source_text = " ".join([inclusion, exclusion]).strip()
        snippet = ""
        if source_text:
            lower = source_text.lower()
            idx = lower.find(keyword)
            if idx != -1:
                left = max(0, idx - 180)
                right = min(len(source_text), idx + len(keyword) + 180)
                snippet = source_text[left:right].strip()
                if left > 0:
                    snippet = "..." + snippet
                if right < len(source_text):
                    snippet = snippet + "..."
                snippet = short_snippet(snippet, limit=360)
            else:
                snippet = short_snippet(source_text, limit=360)
        out.append(
            {
                "trial_id": hit["trial_id"],
                "match_count": hit["match_count"],
                "conditions": trial.get("condition", []),
                "phase": trial.get("phase", ""),
                "interventions": trial.get("interventions", []),
                "evidence_snippet": snippet,
            }
        )

    return {"keyword_hits": out}


def tool_drugbank_lookup(state: MultiAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    query = (args.get("query") or args.get("name") or "").strip()
    if not query:
        return {"error": "query is required"}
    top_k = int(args.get("top_k", 5))
    try:
        index = load_drugbank_index(str(settings.drugbank_minimal_index))
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return drugbank_lookup(index=index, query=query, top_k=top_k)


TOOLS: Dict[str, Dict[str, Any]] = {
    "retrieve_similar_trials": {
        "description": "Retrieve top-K similar trials from the local corpus.",
        "args": {
            "top_k": "int",
            "max_hops": "int",
            "focus": "str (full/condition/drug/biomarker/study/endpoint/outcome)",
        },
        "fn": tool_retrieve_similar_trials,
    },
    "retrieve_keyword_trials": {
        "description": "Retrieve trials by structured overlap on conditions/drugs/biomarkers/endpoints/phase.",
        "args": {
            "top_k": "int",
            "min_match": "int",
            "max_keywords": "int",
            "limit": "int",
            "focus": "str (comma-separated: condition,endpoint,drug,biomarker,study,full)",
        },
        "fn": tool_retrieve_keyword_trials,
    },
    "get_trial_details": {
        "description": "Get a compact view of a trial by trial_id (criteria/endpoints/arms are truncated).",
        "args": {"trial_id": "str", "text_limit": "int"},
        "fn": tool_get_trial_details,
    },
    "keyword_search": {
        "description": "Search the local corpus for a keyword (e.g., biomarker) across conditions/interventions/criteria.",
        "args": {"keyword": "str", "top_k": "int"},
        "fn": tool_keyword_search,
    },
    "drugbank_lookup": {
        "description": "Lookup DrugBank by drug name/synonym to get targets/genes and short indication/MoA (requires drugbank_minimal.jsonl).",
        "args": {"query": "str", "top_k": "int"},
        "fn": tool_drugbank_lookup,
    },
}


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.strip().strip("`")
        lines = text.splitlines()
        if lines and lines[0].startswith("json"):
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_llm_action(response: str, allowed_tools: List[str]) -> Tuple[str, Dict[str, Any]]:
    data = _extract_json(response)
    if not data:
        return "invalid", {"error": "unparseable_response", "raw": response}

    action_type = data.get("type")
    if action_type == "final":
        return "final", {"final": data.get("final", data)}

    if action_type == "tool":
        tool_name = data.get("tool", "")
        args = data.get("args", {}) or {}
        return "tool", {"tool": tool_name, "args": args}

    if action_type in allowed_tools:
        return "tool", {"tool": action_type, "args": data.get("args", {}) or {}}

    tool_name = data.get("tool") or data.get("action") or data.get("name")
    if tool_name in allowed_tools:
        return "tool", {"tool": tool_name, "args": data.get("args", {}) or {}}

    if action_type is not None:
        return "invalid", {"error": "invalid_type", "raw": data}

    # No type: treat as final payload.
    return "final", {"final": data}


def parse_final_response(response: str) -> Dict[str, Any]:
    data = _extract_json(response)
    if not data:
        return {"error": "unparseable_response", "raw": response}
    if isinstance(data, dict) and data.get("type") == "final":
        return data.get("final", data)
    if isinstance(data, dict) and "final" in data:
        return data.get("final", data)
    return data


def _truncate_text(text: str, limit: int = 1200) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _summarize_list(values: List[Any], limit: int = 1, str_limit: int = 200) -> Dict[str, Any]:
    sample: List[Any] = []
    for item in values[:limit]:
        if isinstance(item, dict):
            entry: Dict[str, Any] = {}
            for key, val in item.items():
                if isinstance(val, list):
                    entry[key] = f"len={len(val)}"
                elif isinstance(val, str):
                    entry[key] = _truncate_text(val, str_limit)
                else:
                    entry[key] = val
            sample.append(entry)
        elif isinstance(item, str):
            sample.append(_truncate_text(item, str_limit))
        else:
            sample.append(item)
    return {"len": len(values), "sample": sample}


def _summarize_value(value: Any, limit: int = 400) -> Any:
    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(val, list):
                summary[key] = _summarize_list(val)
            elif isinstance(val, dict):
                summary[key] = f"keys={list(val.keys())[:8]}"
            elif isinstance(val, str):
                summary[key] = _truncate_text(val, limit)
            else:
                summary[key] = val
        return summary
    if isinstance(value, list):
        return _summarize_list(value)
    if isinstance(value, str):
        return _truncate_text(value, limit)
    return value


def _summarize_trial_outcomes(trial: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    outcomes = trial.get("outcomes", []) or []
    results = trial.get("results", []) or []
    summary: Dict[str, Any] = {}
    if isinstance(outcomes, dict):
        summary["outcomes"] = {
            key: outcomes.get(key, "")
            for key in ("overall_status", "outcome_type", "why_terminated")
            if outcomes.get(key)
        }
    elif isinstance(outcomes, list):
        items = []
        for item in outcomes[:limit]:
            if not isinstance(item, dict):
                continue
            entry = {
                key: item.get(key, "")
                for key in ("overall_status", "outcome_type", "why_terminated")
                if item.get(key)
            }
            if entry:
                items.append(entry)
        if items:
            summary["outcomes"] = items
    outcomes_summary = trial.get("outcomes_summary", {}) or {}
    if outcomes_summary and "outcomes" not in summary:
        summary["outcomes_summary"] = {
            key: outcomes_summary.get(key, "")
            for key in ("overall_status", "outcome_type", "why_terminated")
            if outcomes_summary.get(key)
        }
    if isinstance(results, list) and results:
        items = []
        for item in results[:limit]:
            if not isinstance(item, dict):
                continue
            entry = {
                key: item.get(key, "")
                for key in ("population", "interventions", "outcomes", "group_type")
                if item.get(key)
            }
            if entry:
                items.append(entry)
        if items:
            summary["results"] = items
    return summary


def _summarize_retrieval_observation(observation: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    retrieved = observation.get("retrieved_trials") or []
    preview = []
    trial_ids = []
    for item in retrieved[:limit]:
        if not isinstance(item, dict):
            continue
        trial_id = item.get("trial_id")
        if trial_id:
            trial_ids.append(trial_id)
        preview.append(
            {
                "trial_id": trial_id,
                "score": item.get("score"),
                "snippet": item.get("snippet"),
            }
        )
    return {
        "retrieved_trial_ids": trial_ids,
        "retrieved_trials_preview": preview,
        "scanned": observation.get("scanned"),
        "keywords_used": observation.get("keywords_used"),
        "match_basis": observation.get("match_basis"),
    }


def _append_trace(trace_events: Optional[List[Dict[str, Any]]], payload: Dict[str, Any]) -> None:
    if trace_events is None:
        return
    trace_events.append(payload)


def build_prompt(
    agent_name: str,
    agent_goal: str,
    output_contract: str,
    allowed_tools: List[str],
    state: MultiAgentState,
    history: List[Dict[str, Any]],
    agent_focus: Optional[str],
    agent_instructions: Optional[str],
) -> str:
    tools_desc = [
        {
            "name": name,
            "description": TOOLS[name]["description"],
            "args": TOOLS[name]["args"],
        }
        for name in allowed_tools
    ]

    agent_key = state.active_agent or ""
    agent_retrieved = (
        state.retrieved_by_agent.get(agent_key)
        if agent_key
        else state.retrieved
    )
    shared = {
        "retrieved_trials": _format_retrieved(agent_retrieved) if agent_retrieved else None,
        "drug_biomarker_analysis": state.drug_biomarker,
        "design_analysis": state.design_analysis,
    }

    focus_note = ""
    if agent_focus:
        focus_note += f"Agent focus: {agent_focus}\n"
    if agent_instructions:
        focus_note += f"Agent instructions:\n{agent_instructions}\n"

    prompt = (
        f"You are {agent_name}.\n"
        f"Goal: {agent_goal}\n\n"
        f"{focus_note}\n"
        "You are running in a tool-using ReAct loop.\n"
        "Respond ONLY in JSON. Use exactly one of:\n"
        '{"type":"tool","tool":"TOOL_NAME","args":{...}}\n'
        '{"type":"final","final":{...}}\n'
        'Do NOT put a tool name into the "type" field.\n\n'
        "Available tools:\n"
        f"{json.dumps(tools_desc, indent=2)}\n\n"
        "Focus guide (use when calling retrieve_keyword_trials):\n"
        f"{FOCUS_GUIDE}\n\n"
        "Current trial schema:\n"
        f"{json.dumps(compact_trial_for_prompt(state.trial, include_outcomes=False), ensure_ascii=False, indent=2)}\n\n"
        "Shared context (from tools/other agents):\n"
        f"{json.dumps(shared, ensure_ascii=False, indent=2)}\n\n"
        "History (most recent last):\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
        'If "retrieved_trials" is null, call retrieve_keyword_trials first.\n'
        "Output contract:\n"
        f"{output_contract}\n"
    )
    return prompt


def run_agent_loop(
    client: DifyClient,
    agent_name: str,
    agent_goal: str,
    output_contract: str,
    allowed_tools: List[str],
    state: MultiAgentState,
    max_steps: int,
    top_k_default: int,
    conversation_id: str,
    agent_focus: Optional[str],
    agent_instructions: Optional[str],
    trace_events: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    state.active_agent = agent_name
    state.retrieved = state.retrieved_by_agent.get(agent_name)

    for step in range(1, max_steps + 1):
        prompt = build_prompt(
            agent_name=agent_name,
            agent_goal=agent_goal,
            output_contract=output_contract,
            allowed_tools=allowed_tools,
            state=state,
            history=history,
            agent_focus=agent_focus,
            agent_instructions=agent_instructions,
        )
        _append_trace(
            trace_events,
            {
                "event": "prompt",
                "agent": agent_name,
                "step": step,
                "ts": time.time(),
                "prompt": prompt,
                "prompt_preview": _truncate_text(prompt),
            },
        )
        response = client.chat(prompt, conversation_id=conversation_id)
        _append_trace(
            trace_events,
            {
                "event": "response",
                "agent": agent_name,
                "step": step,
                "ts": time.time(),
                "response": response or "",
                "response_preview": _truncate_text(response or ""),
            },
        )
        if not response:
            return {"error": "no_response_from_llm"}, history

        action_type, payload = parse_llm_action(response, allowed_tools)
        if action_type == "final":
            final_payload = payload.get("final", payload)
            _append_trace(
                trace_events,
                {
                    "event": "final",
                    "agent": agent_name,
                    "step": step,
                    "ts": time.time(),
                    "final": final_payload,
                    "final_preview": _summarize_value(final_payload),
                },
            )
            return final_payload, history
        if action_type == "invalid":
            history.append({"error": payload.get("error"), "raw": payload.get("raw")})
            continue

        tool_name = payload.get("tool", "")
        tool_args = payload.get("args", {}) or {}

        if tool_name not in allowed_tools:
            history.append({"tool": tool_name, "args": tool_args, "observation": "tool_not_allowed"})
            continue

        if tool_name in ("retrieve_similar_trials", "retrieve_keyword_trials") and "top_k" not in tool_args:
            tool_args["top_k"] = top_k_default
        if tool_name == "retrieve_keyword_trials" and agent_focus and "focus" not in tool_args:
            tool_args["focus"] = agent_focus

        _append_trace(
            trace_events,
            {
                "event": "tool_call",
                "agent": agent_name,
                "step": step,
                "ts": time.time(),
                "tool": tool_name,
                "args": tool_args,
                "args_preview": _summarize_value(tool_args),
            },
        )

        observation = TOOLS[tool_name]["fn"](state, tool_args)
        history.append({"tool": tool_name, "args": tool_args, "observation": observation})
        if tool_name == "retrieve_keyword_trials" and isinstance(observation, dict):
            observation_summary = _summarize_retrieval_observation(observation)
        else:
            observation_summary = _summarize_value(observation)
        _append_trace(
            trace_events,
            {
                "event": "tool_result",
                "agent": agent_name,
                "step": step,
                "ts": time.time(),
                "tool": tool_name,
                "observation": observation,
                "observation_preview": observation_summary,
            },
        )

    return {"error": "max_steps_reached"}, history


def run_multi_agent(
    trial_data: Dict[str, Any],
    top_k: int,
    max_steps: int,
    dify_api_key: Optional[str],
    conversation_id: str,
    allowlist_csv: Optional[Path],
    relation_only: bool,
    drug_focus: Optional[str],
    design_focus: Optional[str],
    outcome_focus: Optional[str],
    trace_path: Optional[Path],
) -> Dict[str, Any]:
    trial = normalize_trial(trial_data)
    allowlist_path = allowlist_csv if allowlist_csv else settings.trial_id_allowlist_csv
    relation_graph = None
    trial_store = None
    simhash_index = None
    index_path = settings.processed_trials.with_suffix(".index.json")
    if index_path.exists():
        trial_store = TrialStore(settings.processed_trials, index_path)
    if settings.trialpanorama_relations.exists():
        try:
            relation_graph = load_relation_graph(settings.trialpanorama_relations)
        except Exception:
            relation_graph = None
    if settings.simhash_index_path.exists():
        try:
            simhash_index = SimHashIndex(settings.simhash_index_path)
        except Exception:
            simhash_index = None

    corpus: List[Dict[str, Any]] = []
    index: List[Dict[str, Any]] = []
    trial_by_id: Dict[str, Dict[str, Any]] = {}
    search_blob_by_id: Dict[str, str] = {}
    trial_id = (trial.get("trial_id") or "").strip()
    should_load_corpus = not relation_only and not (
        trial_store and relation_graph and trial_id and f"study:{trial_id}" in relation_graph
    )
    vector_index_ready = vector_index_available()
    keyword_index_ready = settings.keyword_index_path.exists()
    if should_load_corpus and not (
        trial_store and (simhash_index or vector_index_ready or keyword_index_ready)
    ):
        corpus = load_corpus(settings.processed_trials, allowlist_path)
        index = build_in_memory_index(corpus)
        trial_by_id = {t.get("trial_id", ""): t for t in corpus if t.get("trial_id")}
        search_blob_by_id = {tid: _trial_blob(t) for tid, t in trial_by_id.items()}
    state = MultiAgentState(
        trial=trial,
        corpus=corpus,
        index=index,
        trial_by_id=trial_by_id,
        search_blob_by_id=search_blob_by_id,
        relation_graph=relation_graph,
        trial_store=trial_store,
        simhash_index=simhash_index,
        vector_stores={},
        allowlist_path=allowlist_path,
        relation_only=relation_only,
    )

    client = DifyClient(api_key=dify_api_key)
    trace_events: Optional[List[Dict[str, Any]]] = [] if trace_path else None
    trace_output_path: Optional[Path] = None
    if trace_path:
        trial_id_for_trace = (trial.get("trial_id") or "unknown").strip() or "unknown"
        trace_dir = trace_path.parent if trace_path.suffix else trace_path
        trace_output_path = trace_dir / f"trace_{trial_id_for_trace}.json"
    _append_trace(
        trace_events,
        {
            "event": "input_trial",
            "ts": time.time(),
            "trial_id": trial.get("trial_id"),
            "trial": trial,
            "trial_outcomes": _summarize_trial_outcomes(trial),
        },
    )

    allowed_tools = [
        "retrieve_keyword_trials",
        "get_trial_details",
        "keyword_search",
        "drugbank_lookup",
    ]

    drug_contract = (
        "Return a JSON object with keys:\n"
        "- drug_hypotheses: list of {name, inferred_target_or_moa, confidence (low/medium/high), evidence}\n"
        "- biomarker_hypotheses: list of {biomarker, rationale, confidence, evidence}\n"
        "- population_alignment_notes: list of short bullets\n"
        "- open_questions_for_medical_review: list of questions\n"
        "Rules: use tool drugbank_lookup(drug_name) when possible to ground target/gene hypotheses. "
        "Only infer targets/biomarkers when clearly implied by trial/intervention text, DrugBank, or retrieved evidence; "
        "otherwise mark as open question. Cite trial_ids/snippets when possible."
    )
    design_contract = (
        "Return a JSON object with keys:\n"
        "- design_risk_assessment: list of {issue, why_it_matters, evidence, severity (low/medium/high)}\n"
        "- biomarker_design_alignment: list of short bullets\n"
        "- endpoint_feasibility_notes: list of short bullets\n"
        "- open_questions_for_stat_medical_review: list of questions\n"
        "Rules: do NOT propose protocol edits as recommendations; focus on risks, rationales, and what needs expert review."
    )
    outcome_contract = (
        "Return a JSON object with keys:\n"
        "- success_probability_estimate: number between 0 and 1 (LLM heuristic, not a trained model)\n"
        "- top_drivers: list of {driver, direction (up/down), evidence}\n"
        "- key_uncertainties: list of short bullets\n"
        "- summary: 3-6 sentences\n"
        "Rules: ground on retrieved trials and prior agent outputs; avoid medical claims without evidence."
    )
    final_contract = (
        "Return a JSON object with keys:\n"
        "- success_probability: number between 0 and 1\n"
        "- key_reasons: list of {reason, evidence}\n"
        "- evidence_trials: list of trial_id\n"
        "- open_questions: list of short bullets\n"
        "Rules: base the decision on retrieved evidence and agent outputs only."
    )

    drug_focus = (drug_focus or "drug,biomarker").strip().lower()
    design_focus = (design_focus or "condition,endpoint,study").strip().lower()
    outcome_focus = (outcome_focus or "full").strip().lower()
    drug_instructions = (
        "Prioritize drug/biomarker evidence only. "
        "Use drugbank_lookup when possible. Avoid design/outcome claims."
    )
    design_instructions = (
        "Focus on design/criteria/endpoint feasibility and study attributes. "
        "Avoid outcome prediction."
    )
    outcome_instructions = (
        "Synthesize success estimate using retrieved trials and other agent outputs. "
        "Avoid introducing new facts without evidence."
    )

    drug_out, drug_hist = run_agent_loop(
        client=client,
        agent_name="DrugBiomarkerAgent",
        agent_goal="Analyze drug/target/biomarker hypotheses implied by the trial and local evidence; surface key uncertainties.",
        output_contract=drug_contract,
        allowed_tools=allowed_tools,
        state=state,
        max_steps=max_steps,
        top_k_default=top_k,
        conversation_id=conversation_id,
        agent_focus=drug_focus,
        agent_instructions=drug_instructions,
        trace_events=trace_events,
    )
    state.drug_biomarker = drug_out
    state.traces["DrugBiomarkerAgent"] = drug_hist

    design_out, design_hist = run_agent_loop(
        client=client,
        agent_name="DesignAgent",
        agent_goal="Assess whether trial design choices are coherent given the drug/biomarker context and local evidence.",
        output_contract=design_contract,
        allowed_tools=allowed_tools,
        state=state,
        max_steps=max_steps,
        top_k_default=top_k,
        conversation_id=conversation_id,
        agent_focus=design_focus,
        agent_instructions=design_instructions,
        trace_events=trace_events,
    )
    state.design_analysis = design_out
    state.traces["DesignAgent"] = design_hist

    outcome_out, outcome_hist = run_agent_loop(
        client=client,
        agent_name="OutcomeSummaryAgent",
        agent_goal="Synthesize the overall success probability estimate with evidence and key uncertainties.",
        output_contract=outcome_contract,
        allowed_tools=allowed_tools,
        state=state,
        max_steps=max_steps,
        top_k_default=top_k,
        conversation_id=conversation_id,
        agent_focus=outcome_focus,
        agent_instructions=outcome_instructions,
        trace_events=trace_events,
    )
    state.outcome_summary = outcome_out
    state.traces["OutcomeSummaryAgent"] = outcome_hist

    retrieved_by_agent = {
        name: _format_retrieved(items) for name, items in state.retrieved_by_agent.items()
    }
    final_prompt = (
        "You are FinalDecisionAgent.\n"
        "Goal: provide the final success probability estimate using all available evidence.\n"
        "Respond ONLY in JSON as {\"type\":\"final\",\"final\":{...}}.\n\n"
        "Current trial schema:\n"
        f"{json.dumps(compact_trial_for_prompt(state.trial, include_outcomes=False), ensure_ascii=False, indent=2)}\n\n"
        "Retrieved evidence by agent:\n"
        f"{json.dumps(retrieved_by_agent, ensure_ascii=False, indent=2)}\n\n"
        "Agent outputs:\n"
        f"{json.dumps({'drug_biomarker_analysis': state.drug_biomarker, 'design_analysis': state.design_analysis, 'outcome_summary': state.outcome_summary}, ensure_ascii=False, indent=2)}\n\n"
        "Output contract:\n"
        f"{final_contract}\n"
    )
    final_response = client.chat(final_prompt, conversation_id=conversation_id)
    _append_trace(
        trace_events,
        {
            "event": "final_prompt",
            "agent": "FinalDecisionAgent",
            "step": 1,
            "ts": time.time(),
            "prompt": final_prompt,
            "prompt_preview": _truncate_text(final_prompt),
        },
    )
    _append_trace(
        trace_events,
        {
            "event": "final_response",
            "agent": "FinalDecisionAgent",
            "step": 1,
            "ts": time.time(),
            "response": final_response or "",
            "response_preview": _truncate_text(final_response or ""),
        },
    )
    final_decision = parse_final_response(final_response or "")
    _append_trace(
        trace_events,
        {
            "event": "final_decision",
            "agent": "FinalDecisionAgent",
            "step": 1,
            "ts": time.time(),
            "final": final_decision,
            "final_preview": _summarize_value(final_decision),
        },
    )
    if trace_output_path and trace_events is not None:
        trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_output_path.write_text(json.dumps(trace_events, ensure_ascii=False, indent=2))

    # Aggregate common outputs for the caller.
    result = {
        "retrieved_trials": _format_retrieved(state.retrieved) if state.retrieved else None,
        "retrieved_by_agent": retrieved_by_agent,
        "drug_biomarker_analysis": state.drug_biomarker,
        "design_analysis": state.design_analysis,
        "outcome_summary": state.outcome_summary,
        "final_decision": final_decision,
        "trace": state.traces,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent ReAct (Python) with Dify as the LLM.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.processed_trials.parent / "sample_input.json",
        help="Path to a JSON file containing the trial schema.",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default="",
        help="Load trial by ID from processed_trials using the JSONL index.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV of trial IDs (nctid) to select from.",
    )
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=None,
        help="Excel file of trial IDs (nctid) to select from.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="0-based row index into --input-csv/--input-xlsx list.",
    )
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--dify-api-key", type=str, default=None, help="Override DIFY_API_KEY.")
    parser.add_argument("--conversation-id", type=str, default="")
    parser.add_argument(
        "--allowlist-csv",
        type=Path,
        default=None,
        help="Optional CSV of trial IDs (nctid) to restrict the corpus.",
    )
    parser.add_argument(
        "--relation-only",
        action="store_true",
        help="Skip lexical retrieval; only return relation-graph neighbors.",
    )
    parser.add_argument(
        "--drug-focus",
        type=str,
        default=None,
        help="Focus fields for DrugBiomarkerAgent (comma-separated).",
    )
    parser.add_argument(
        "--design-focus",
        type=str,
        default=None,
        help="Focus fields for DesignAgent (comma-separated).",
    )
    parser.add_argument(
        "--outcome-focus",
        type=str,
        default=None,
        help="Focus fields for OutcomeSummaryAgent (comma-separated).",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Directory to store trace JSON files (filename auto includes trial_id).",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trial_id:
        trial_data = _load_trial_from_corpus(args.trial_id)
    elif args.input_csv:
        ids = _read_trial_ids_from_csv(args.input_csv)
        if not ids:
            raise ValueError(f"No trial IDs found in CSV: {args.input_csv}")
        if args.row_index < 0 or args.row_index >= len(ids):
            raise IndexError(
                f"--row-index {args.row_index} out of range (0..{len(ids) - 1})"
            )
        trial_data = _load_trial_from_corpus(ids[args.row_index])
    elif args.input_xlsx:
        ids = _read_trial_ids_from_excel(args.input_xlsx)
        if not ids:
            raise ValueError(f"No trial IDs found in Excel: {args.input_xlsx}")
        if args.row_index < 0 or args.row_index >= len(ids):
            raise IndexError(
                f"--row-index {args.row_index} out of range (0..{len(ids) - 1})"
            )
        trial_data = _load_trial_from_corpus(ids[args.row_index])
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        trial_data = json.loads(args.input.read_text())
    result = run_multi_agent(
        trial_data,
        top_k=args.top_k,
        max_steps=args.max_steps,
        dify_api_key=args.dify_api_key,
        conversation_id=args.conversation_id,
        allowlist_csv=args.allowlist_csv,
        relation_only=args.relation_only,
        drug_focus=args.drug_focus,
        design_focus=args.design_focus,
        outcome_focus=args.outcome_focus,
        trace_path=args.trace_path,
    )
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
