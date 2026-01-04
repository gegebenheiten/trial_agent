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
from trial_agent.ingest.parse_ctgov import normalize_trial
from trial_agent.formatting import compact_trial_for_prompt
from trial_agent.llm import DifyClient
from trial_agent.retrieval.keyword_retrieve import (
    build_match_basis,
    extract_keywords_from_trial,
    normalize_focus_parts,
    structured_retrieve_trials,
)
from trial_agent.retrieval.relations import load_relation_graph, retrieve_related_trials
from trial_agent.retrieval.trial_store import TrialStore


@dataclass
class AgentState:
    trial: Dict[str, Any]
    trial_store: Optional[TrialStore] = None
    allowlist_path: Optional[Path] = None
    forced_focus: Optional[str] = None
    relation_context: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: Optional[List[Dict[str, Any]]] = None


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
            if not value or value.lower() in candidates:
                continue
            if value not in seen:
                ids.append(value)
                seen.add(value)
    return ids


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


def tool_keyword_retrieve(state: AgentState, args: Dict[str, Any]) -> Dict[str, Any]:
    top_k = int(args.get("top_k", settings.default_top_k))
    min_match = int(args.get("min_match", 1))
    max_keywords = int(args.get("max_keywords", 0))
    limit = int(args.get("limit", 0))
    focus = str(args.get("focus") or "condition").strip().lower()
    if state.forced_focus:
        focus = state.forced_focus.strip().lower()
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
    except ValueError as exc:
        return {"error": str(exc)}
    state.retrieved = retrieved
    match_basis = build_match_basis(state.trial, focus_parts)
    sys.stderr.write(
        "[retrieve_keyword_trials] match_basis="
        + json.dumps(match_basis, ensure_ascii=False)
        + "\n"
    )
    return {
        "retrieved_trials": _format_retrieved(retrieved),
        "keywords_used": keywords,
        "scanned": scanned,
        "match_basis": match_basis,
    }


TOOLS = {
    "retrieve_keyword_trials": {
        "description": "Retrieve trials by structured overlap on conditions/drugs/biomarkers/endpoints/phase.",
        "args": {
            "top_k": "int",
            "min_match": "int",
            "max_keywords": "int",
            "limit": "int",
            "focus": "str (comma-separated: condition,endpoint,drug,biomarker,study,full)",
        },
        "fn": tool_keyword_retrieve,
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


def parse_llm_action(response: str) -> Tuple[str, Dict[str, Any]]:
    data = _extract_json(response)
    if not data:
        return "final", {"error": "unparseable_response", "raw": response}

    action_type = data.get("type")
    if action_type == "tool":
        return "tool", {
            "tool": data.get("tool", ""),
            "args": data.get("args", {}) or {},
        }
    if action_type == "final":
        return "final", {"final": data.get("final", data)}

    # Common mistake: put tool name into "type". If it's a known tool, accept it.
    if action_type in TOOLS:
        return "tool", {"tool": action_type, "args": data.get("args", {}) or {}}

    # If "type" exists but is invalid, do not treat as final; ask again via history.
    if action_type is not None:
        return "invalid", {"error": "invalid_type", "raw": data}

    tool_name = data.get("tool") or data.get("action") or data.get("name")
    if tool_name in TOOLS:
        return "tool", {"tool": tool_name, "args": data.get("args", {}) or {}}
    # Fallback: treat any JSON without type as final payload.
    return "final", {"final": data}


def _truncate_text(text: str, limit: int = 1200) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _summarize_value(value: Any, limit: int = 400) -> Any:
    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(val, list):
                summary[key] = f"len={len(val)}"
            elif isinstance(val, dict):
                summary[key] = f"keys={list(val.keys())[:8]}"
            elif isinstance(val, str):
                summary[key] = _truncate_text(val, limit)
            else:
                summary[key] = val
        return summary
    if isinstance(value, list):
        return f"len={len(value)}"
    if isinstance(value, str):
        return _truncate_text(value, limit)
    return value


def build_prompt(state: AgentState, history: List[Dict[str, Any]]) -> str:
    tools_desc = [
        {
            "name": name,
            "description": meta["description"],
            "args": meta["args"],
        }
        for name, meta in TOOLS.items()
    ]
    known = {
        "retrieved_trials": _format_retrieved(state.retrieved) if state.retrieved else None,
        "relation_context": _format_retrieved(state.relation_context)
        if state.relation_context
        else [],
    }
    prompt = (
        "You are a clinical trial protocol ReAct agent. Decide the next tool call or finalize.\n"
        "Respond ONLY in JSON. Use one of:\n"
        '{"type":"tool","tool":"TOOL_NAME","args":{...}}\n'
        '{"type":"final","final":{...}}\n'
        'Do NOT put a tool name into the "type" field.\n'
        "Tools available:\n"
        f"{json.dumps(tools_desc, indent=2)}\n\n"
        "Focus guide (use when calling retrieve_keyword_trials):\n"
        f"{FOCUS_GUIDE}\n\n"
        "Current trial schema:\n"
        f"{json.dumps(compact_trial_for_prompt(state.trial, include_outcomes=False), ensure_ascii=False, indent=2)}\n\n"
        "Known results:\n"
        f"{json.dumps(known, ensure_ascii=False, indent=2)}\n\n"
        "History (most recent last):\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
        'If "retrieved_trials" is null, call retrieve_keyword_trials first.\n'
        "If you have enough information, output the final JSON with: retrieved evidence "
        "and an estimated success_probability (LLM heuristic) with key reasons."
    )
    return prompt


def run_react(
    trial_data: Dict[str, Any],
    top_k: int,
    max_steps: int,
    dify_api_key: Optional[str],
    conversation_id: str,
    allowlist_csv: Optional[Path],
    forced_focus: Optional[str],
    trace_path: Optional[Path],
) -> Dict[str, Any]:
    trial = normalize_trial(trial_data)
    allowlist_path = allowlist_csv if allowlist_csv else settings.trial_id_allowlist_csv
    relation_graph = None
    trial_store = None
    index_path = settings.processed_trials.with_suffix(".index.json")
    if index_path.exists():
        trial_store = TrialStore(settings.processed_trials, index_path)
    if settings.trialpanorama_relations.exists():
        try:
            relation_graph = load_relation_graph(settings.trialpanorama_relations)
        except Exception:
            relation_graph = None
    trial_id = (trial.get("trial_id") or "").strip()
    relation_context: List[Dict[str, Any]] = []
    if relation_graph and trial_store and trial_id:
        if f"study:{trial_id}" in relation_graph:
            try:
                relation_context = retrieve_related_trials(
                    trial_id=trial_id,
                    graph=relation_graph,
                    store=trial_store,
                    top_k=top_k,
                    max_hops=3,
                )
                for item in relation_context:
                    item["trial"] = normalize_trial(item["trial"])
            except Exception:
                relation_context = []

    state = AgentState(
        trial=trial,
        trial_store=trial_store,
        allowlist_path=allowlist_path,
        forced_focus=forced_focus,
        relation_context=relation_context,
    )
    history: List[Dict[str, Any]] = []

    client = DifyClient(api_key=dify_api_key)

    trace_fp = None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fp = trace_path.open("a", encoding="utf-8")

    try:
        for step in range(1, max_steps + 1):
            prompt = build_prompt(state, history)
            if trace_fp:
                trace_fp.write(
                    json.dumps(
                        {
                            "event": "prompt",
                            "step": step,
                            "ts": time.time(),
                            "prompt_preview": _truncate_text(prompt),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace_fp.flush()

            response = client.chat(prompt, conversation_id=conversation_id)
            if trace_fp:
                trace_fp.write(
                    json.dumps(
                        {
                            "event": "response",
                            "step": step,
                            "ts": time.time(),
                            "response_preview": _truncate_text(response or ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace_fp.flush()
            if not response:
                return {"error": "no_response_from_llm"}

            action_type, action_payload = parse_llm_action(response)
            if action_type == "invalid":
                history.append(
                    {"error": action_payload.get("error"), "raw": action_payload.get("raw")}
                )
                continue
            if action_type == "final":
                final_payload = action_payload.get("final", action_payload)
                if trace_fp:
                    trace_fp.write(
                        json.dumps(
                            {
                                "event": "final",
                                "step": step,
                                "ts": time.time(),
                                "final_preview": _summarize_value(final_payload),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    trace_fp.flush()
                return final_payload

            tool_name = action_payload.get("tool", "")
            tool_args = action_payload.get("args", {}) or {}
            tool_meta = TOOLS.get(tool_name)
            if not tool_meta:
                history.append(
                    {"tool": tool_name, "args": tool_args, "observation": "unknown_tool"}
                )
                continue

            if tool_name == "retrieve_keyword_trials" and "top_k" not in tool_args:
                tool_args["top_k"] = top_k

            if trace_fp:
                trace_fp.write(
                    json.dumps(
                        {
                            "event": "tool_call",
                            "step": step,
                            "ts": time.time(),
                            "tool": tool_name,
                            "args": _summarize_value(tool_args),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace_fp.flush()

            observation = tool_meta["fn"](state, tool_args)
            history.append({"tool": tool_name, "args": tool_args, "observation": observation})

            if trace_fp:
                trace_fp.write(
                    json.dumps(
                        {
                            "event": "tool_result",
                            "step": step,
                            "ts": time.time(),
                            "tool": tool_name,
                            "observation": _summarize_value(observation),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace_fp.flush()
    finally:
        if trace_fp:
            trace_fp.close()

    return {
        "error": "max_steps_reached",
        "partial_state": {
            "retrieved_trials": _format_retrieved(state.retrieved) if state.retrieved else None,
            "relation_context": _format_retrieved(state.relation_context)
            if state.relation_context
            else [],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReAct agent with Dify as the LLM.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.processed_trials.parent / "sample_input.json",
        help="Path to a JSON file containing the trial schema (used when --trial-id/--input-csv are not set).",
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
        help="CSV of trial IDs (uses --csv-index to pick one when --trial-id not set).",
    )
    parser.add_argument(
        "--csv-index",
        type=int,
        default=0,
        help="0-based index into --input-csv list.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.default_top_k,
        help="Number of similar trials to retrieve.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Max tool steps before abort.")
    parser.add_argument("--dify-api-key", type=str, default=None, help="Override DIFY_API_KEY.")
    parser.add_argument(
        "--conversation-id",
        type=str,
        default="",
        help="Optional Dify conversation_id for multi-turn memory.",
    )
    parser.add_argument(
        "--allowlist-csv",
        type=Path,
        default=None,
        help="Optional CSV of trial IDs (nctid) to restrict the corpus.",
    )
    parser.add_argument(
        "--force-focus",
        type=str,
        default=None,
        help="Force retrieve_keyword_trials focus (e.g. condition,endpoint,drug).",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional JSONL file to log prompt/response/tool I/O (summarized).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_data: Dict[str, Any]
    if args.trial_id:
        trial_data = _load_trial_from_corpus(args.trial_id)
    elif args.input_csv:
        ids = _read_trial_ids_from_csv(args.input_csv)
        if not ids:
            raise ValueError(f"No trial IDs found in CSV: {args.input_csv}")
        if args.csv_index < 0 or args.csv_index >= len(ids):
            raise IndexError(
                f"--csv-index {args.csv_index} out of range (0..{len(ids) - 1})"
            )
        trial_data = _load_trial_from_corpus(ids[args.csv_index])
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        trial_data = json.loads(args.input.read_text())
    
    result = run_react(
        trial_data,
        top_k=args.top_k,
        max_steps=args.max_steps,
        dify_api_key=args.dify_api_key,
        conversation_id=args.conversation_id,
        allowlist_csv=args.allowlist_csv,
        forced_focus=args.force_focus,
        trace_path=args.trace_path,
    )
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
