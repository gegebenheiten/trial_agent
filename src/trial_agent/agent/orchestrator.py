import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from trial_agent.agent.suggest import generate_recommendations
from trial_agent.config import settings
from trial_agent.formatting import compact_trial_for_prompt
from trial_agent.ingest.parse_ctgov import load_jsonl, normalize_trial
from trial_agent.llm import DifyClient, build_dify_prompt
from trial_agent.retrieval.index import build_in_memory_index
from trial_agent.retrieval.relations import load_relation_graph, retrieve_related_trials
from trial_agent.retrieval.search import search_trials, search_trials_simhash, search_trials_vector
from trial_agent.retrieval.simhash_index import SimHashIndex
from trial_agent.retrieval.trial_store import TrialStore
from trial_agent.retrieval.vector_store import VectorStore


def load_corpus(path: Path, allowlist_path: Optional[Path]) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Processed trial corpus not found: {path}")
    trials = [normalize_trial(t) for t in load_jsonl(path)]
    if allowlist_path and allowlist_path.exists():
        from trial_agent.ingest.allowlist import load_trial_id_allowlist

        allowlist = load_trial_id_allowlist(allowlist_path)
        if allowlist:
            trials = [t for t in trials if t.get("trial_id") in allowlist]
    return trials


def run_pipeline(
    trial_data: Dict,
    top_k: int = settings.default_top_k,
    use_dify: bool = False,
    dify_api_key: Optional[str] = None,
    allowlist_csv: Optional[Path] = None,
    relation_only: bool = False,
) -> Dict:
    trial = normalize_trial(trial_data)
    allowlist_path = allowlist_csv if allowlist_csv else settings.trial_id_allowlist_csv
    relation_graph = None
    trial_store = None
    simhash_index = None
    vector_store = None
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
    if settings.vector_index_path.exists() and settings.vector_id_map_path.exists():
        try:
            vector_store = VectorStore(
                settings.vector_index_path,
                settings.vector_id_map_path,
                model_name=settings.embedding_model_name,
                trust_remote_code=settings.embedding_trust_remote_code,
            )
        except Exception:
            vector_store = None

    trial_id = (trial.get("trial_id") or "").strip()
    if trial_store and relation_graph and trial_id and f"study:{trial_id}" in relation_graph:
        retrieved = retrieve_related_trials(
            trial_id=trial_id,
            graph=relation_graph,
            store=trial_store,
            top_k=top_k,
            max_hops=3,
        )
        for item in retrieved:
            item["trial"] = normalize_trial(item["trial"])
    elif not relation_only and trial_store and vector_store:
        allowlist = None
        if allowlist_path and allowlist_path.exists():
            from trial_agent.ingest.allowlist import load_trial_id_allowlist

            allowlist = load_trial_id_allowlist(allowlist_path)
        try:
            retrieved = search_trials_vector(
                trial,
                trial_store=trial_store,
                vector_store=vector_store,
                top_k=top_k,
                allowlist=allowlist,
            )
        except Exception:
            retrieved = []
    elif not relation_only and trial_store and simhash_index:
        allowlist = None
        if allowlist_path and allowlist_path.exists():
            from trial_agent.ingest.allowlist import load_trial_id_allowlist

            allowlist = load_trial_id_allowlist(allowlist_path)
        retrieved = search_trials_simhash(
            trial,
            trial_store=trial_store,
            simhash_index=simhash_index,
            top_k=top_k,
            allowlist=allowlist,
        )
    elif not relation_only:
        corpus = load_corpus(settings.processed_trials, allowlist_path)
        index = build_in_memory_index(corpus)
        retrieved = search_trials(trial, index, top_k=top_k, relation_graph=relation_graph)
    else:
        retrieved = []
    recommendations, open_questions = generate_recommendations(trial, retrieved)

    if not retrieved:
        retrieved = []

    def _primary_endpoint(trial: Dict) -> str:
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

    retrieved_payload = [
        {
            "trial_id": r["trial"].get("trial_id"),
            "score": round(r["score"], 3),
            "relation_hop": r.get("hop"),
            "snippet": r.get("snippet", ""),
            "phase": r["trial"].get("phase", "") or (r["trial"].get("study", {}) or {}).get("phase", ""),
            "primary_endpoint": _primary_endpoint(r["trial"]),
            "trial_compact": compact_trial_for_prompt(r["trial"]),
        }
        for r in retrieved
    ]

    dify_response: Optional[str] = None
    if use_dify:
        try:
            client = DifyClient(api_key=dify_api_key)
            prompt = build_dify_prompt(
                trial,
                retrieved_payload,
            )
            dify_response = client.chat(prompt)
        except Exception as exc:  # pragma: no cover - network path
            dify_response = f"error: {exc}"

    return {
        "recommendations": recommendations,
        "retrieved_trials": retrieved_payload,
        "open_questions_for_medical_review": open_questions,
        "dify_response": dify_response,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trial agent MVP pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.processed_trials.parent / "sample_input.json",
        help="Path to a JSON file containing the trial schema.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.default_top_k,
        help="Number of similar trials to retrieve.",
    )
    parser.add_argument(
        "--use-dify",
        action="store_true",
        help="Call Dify API with the grounded prompt to generate LLM suggestions.",
    )
    parser.add_argument(
        "--dify-api-key",
        type=str,
        default=None,
        help="Optional override for DIFY_API_KEY environment variable.",
    )
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
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    trial_data = json.loads(args.input.read_text())
    result = run_pipeline(
        trial_data,
        top_k=args.top_k,
        use_dify=args.use_dify,
        dify_api_key=args.dify_api_key,
        allowlist_csv=args.allowlist_csv,
        relation_only=args.relation_only,
    )
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
