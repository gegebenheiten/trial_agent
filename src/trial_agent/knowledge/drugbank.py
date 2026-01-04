import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trial_agent.ingest.clean_text import normalize_whitespace, short_snippet


def _norm(s: str) -> str:
    s = normalize_whitespace(s or "").lower()
    s = s.strip("\"'`")
    return s


@dataclass
class DrugBankIndex:
    """
    Minimal in-memory index built from a JSONL file.

    - `records_by_id`: drugbank_id -> record dict
    - `name_to_id`: normalized name/synonym -> drugbank_id
    """

    records_by_id: Dict[str, Dict[str, Any]]
    name_to_id: Dict[str, str]


def _add_name(name_to_id: Dict[str, str], name: str, drugbank_id: str) -> None:
    key = _norm(name)
    if not key:
        return
    # Keep the first mapping to avoid accidental overwrites.
    name_to_id.setdefault(key, drugbank_id)


def _summarize_properties(properties: Dict[str, Any], max_items: int = 6) -> Dict[str, Any]:
    if not properties or not isinstance(properties, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("calculated", "experimental"):
        items = properties.get(key) or []
        if not isinstance(items, list):
            continue
        trimmed: List[Dict[str, str]] = []
        for item in items[:max_items]:
            if not isinstance(item, dict):
                continue
            entry = {
                "kind": item.get("kind", ""),
                "value": item.get("value", ""),
            }
            if item.get("source"):
                entry["source"] = item.get("source", "")
            trimmed.append(entry)
        if trimmed:
            out[key] = trimmed
    return out


def _summarize_pharmacology(pharmacology: Dict[str, Any]) -> Dict[str, str]:
    if not pharmacology or not isinstance(pharmacology, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in pharmacology.items():
        if not value:
            continue
        out[key] = short_snippet(str(value), limit=400)
    return out


def _summarize_record(record: Dict[str, Any], max_targets: int = 8) -> Dict[str, Any]:
    targets = record.get("targets", []) or []
    out_targets = []
    genes: List[str] = []
    for t in targets:
        t_genes = [g for g in (t.get("genes", []) or []) if g]
        genes.extend(t_genes)
        out_targets.append(
            {
                "name": t.get("name", ""),
                "organism": t.get("organism", ""),
                "actions": t.get("actions", []) or [],
                "genes": t_genes[:10],
                "uniprot_ids": (t.get("uniprot_ids", []) or [])[:10],
            }
        )
        if len(out_targets) >= max_targets:
            break

    unique_genes = []
    seen = set()
    for g in genes:
        gl = g.strip()
        if not gl or gl in seen:
            continue
        seen.add(gl)
        unique_genes.append(gl)

    summary = {
        "drugbank_id": record.get("drugbank_id", ""),
        "name": record.get("name", ""),
        "groups": record.get("groups", []) or [],
        "targets": out_targets,
        "target_genes": unique_genes[:40],
        "indication": short_snippet(record.get("indication", "") or "", limit=500),
        "mechanism_of_action": short_snippet(record.get("mechanism_of_action", "") or "", limit=500),
    }
    props = _summarize_properties(record.get("properties", {}))
    if props:
        summary["properties"] = props
    pharm = _summarize_pharmacology(record.get("pharmacology", {}))
    if pharm:
        summary["pharmacology"] = pharm
    return summary


@lru_cache(maxsize=2)
def load_drugbank_index(jsonl_path: str) -> DrugBankIndex:
    """
    Load a minimal DrugBank index produced by preprocess/build_drugbank_minimal.py.
    Cached per process.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DrugBank index not found: {path}. Build it with preprocess/build_drugbank_minimal.py."
        )

    records_by_id: Dict[str, Dict[str, Any]] = {}
    name_to_id: Dict[str, str] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            drugbank_id = record.get("drugbank_id") or ""
            if not drugbank_id:
                continue
            records_by_id[drugbank_id] = record
            _add_name(name_to_id, record.get("name", ""), drugbank_id)
            for syn in (record.get("synonyms", []) or []):
                _add_name(name_to_id, syn, drugbank_id)

    return DrugBankIndex(records_by_id=records_by_id, name_to_id=name_to_id)


def lookup(
    index: DrugBankIndex,
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Return best-effort matches for a drug name/synonym.
    This is heuristic string matching meant to support LLM grounding.
    """
    q = _norm(query)
    if not q:
        return {"error": "empty_query"}

    # Exact lookup first.
    if q in index.name_to_id:
        drug_id = index.name_to_id[q]
        record = index.records_by_id.get(drug_id, {})
        return {
            "query": query,
            "match_type": "exact",
            "matches": [_summarize_record(record)],
        }

    # Substring search across known names/synonyms (bounded).
    candidates: List[Tuple[float, str]] = []
    for name_key, drug_id in index.name_to_id.items():
        if q in name_key:
            # Prefer shorter distance between lengths, and prefix matches.
            score = 0.6
            if name_key.startswith(q):
                score += 0.25
            score += max(0.0, 0.2 - abs(len(name_key) - len(q)) / 100)
            candidates.append((score, drug_id))

    if not candidates:
        return {"query": query, "match_type": "none", "matches": []}

    # Rank by score and de-duplicate by drug id.
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    matches: List[Dict[str, Any]] = []
    for score, drug_id in candidates:
        if drug_id in seen:
            continue
        seen.add(drug_id)
        record = index.records_by_id.get(drug_id)
        if not record:
            continue
        summary = _summarize_record(record)
        summary["score"] = round(score, 3)
        matches.append(summary)
        if len(matches) >= top_k:
            break

    return {"query": query, "match_type": "substring", "matches": matches}
