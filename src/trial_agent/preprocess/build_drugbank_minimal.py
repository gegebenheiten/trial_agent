"""
Build a minimal DrugBank index JSONL from the full DrugBank XML dump.

Input (default):
  trial_agent/data/drugbank/full_database.xml

Output (default):
  trial_agent/data/processed/drugbank_minimal.jsonl

The output is intentionally compact and only keeps fields useful for reasoning:
name/synonyms/groups + short indication/MoA + target genes/actions.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from trial_agent.config import settings
from trial_agent.ingest.clean_text import normalize_whitespace, short_snippet


NS_URI = "http://www.drugbank.ca"
NS = {"db": NS_URI}
DRUG_TAG = f"{{{NS_URI}}}drug"


def _text(parent: ET.Element, path: str) -> str:
    return normalize_whitespace(parent.findtext(path, default="", namespaces=NS) or "")


def _list_text(parent: ET.Element, path: str) -> List[str]:
    out: List[str] = []
    for el in parent.findall(path, NS):
        if el is None or el.text is None:
            continue
        t = normalize_whitespace(el.text)
        if t:
            out.append(t)
    return out


def _primary_drugbank_id(drug: ET.Element) -> str:
    ids = drug.findall("db:drugbank-id", NS)
    for el in ids:
        if el.get("primary") == "true" and el.text:
            return el.text.strip()
    for el in ids:
        if el.text:
            return el.text.strip()
    return ""


def _parse_targets(drug: ET.Element) -> List[Dict[str, Any]]:
    targets_out: List[Dict[str, Any]] = []
    for t in drug.findall("db:targets/db:target", NS):
        tname = _text(t, "db:name")
        organism = _text(t, "db:organism")
        actions = _list_text(t, "db:actions/db:action")

        genes: List[str] = []
        uniprots: List[str] = []
        for pol in t.findall("db:polypeptide", NS):
            gene = _text(pol, "db:gene-name")
            if gene:
                genes.append(gene)
            up = _text(pol, "db:uniprot-id")
            if up:
                uniprots.append(up)

        if not genes and not uniprots and not tname:
            continue

        targets_out.append(
            {
                "name": tname,
                "organism": organism,
                "actions": actions,
                "genes": genes,
                "uniprot_ids": uniprots,
            }
        )
    return targets_out


def _parse_properties(drug: ET.Element) -> Dict[str, List[Dict[str, str]]]:
    properties: Dict[str, List[Dict[str, str]]] = {}
    for category, key in (
        ("calculated-properties", "calculated"),
        ("experimental-properties", "experimental"),
    ):
        items: List[Dict[str, str]] = []
        for prop in drug.findall(f"db:{category}/db:property", NS):
            kind = _text(prop, "db:kind")
            value = _text(prop, "db:value")
            source = _text(prop, "db:source")
            if not kind or not value:
                continue
            entry = {"kind": kind, "value": value}
            if source:
                entry["source"] = source
            items.append(entry)
        if items:
            properties[key] = items
    return properties


def _parse_pharmacology(drug: ET.Element, max_text_chars: int) -> Dict[str, str]:
    fields = {
        "pharmacology": "db:pharmacology",
        "pharmacodynamics": "db:pharmacodynamics",
        "absorption": "db:absorption",
        "toxicity": "db:toxicity",
        "clearance": "db:clearance",
        "half_life": "db:half-life",
        "route_of_elimination": "db:route-of-elimination",
        "volume_of_distribution": "db:volume-of-distribution",
        "protein_binding": "db:protein-binding",
        "metabolism": "db:metabolism",
    }
    out: Dict[str, str] = {}
    for key, path in fields.items():
        value = short_snippet(_text(drug, path), limit=max_text_chars)
        if value:
            out[key] = value
    return out


def parse_drug(drug: ET.Element, max_text_chars: int) -> Optional[Dict[str, Any]]:
    # Only keep top-level drugs (they have attributes like type/created/updated).
    if drug.tag != DRUG_TAG or drug.get("type") is None:
        return None

    drugbank_id = _primary_drugbank_id(drug)
    name = _text(drug, "db:name")
    if not drugbank_id or not name:
        return None

    groups = _list_text(drug, "db:groups/db:group")
    synonyms = _list_text(drug, "db:synonyms/db:synonym")

    # Also include international brand names when available.
    synonyms.extend(_list_text(drug, "db:international-brands/db:international-brand/db:name"))
    # De-duplicate synonyms while preserving order.
    seen = set()
    deduped = []
    for s in synonyms:
        key = s.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    synonyms = deduped

    indication = short_snippet(_text(drug, "db:indication"), limit=max_text_chars)
    mechanism = short_snippet(_text(drug, "db:mechanism-of-action"), limit=max_text_chars)
    targets = _parse_targets(drug)
    properties = _parse_properties(drug)
    pharmacology = _parse_pharmacology(drug, max_text_chars=max_text_chars)

    record = {
        "drugbank_id": drugbank_id,
        "name": name,
        "groups": groups,
        "synonyms": synonyms,
        "indication": indication,
        "mechanism_of_action": mechanism,
        "targets": targets,
    }
    if properties:
        record["properties"] = properties
    if pharmacology:
        record["pharmacology"] = pharmacology
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal DrugBank JSONL index from full_database.xml.")
    parser.add_argument("--xml", type=Path, default=settings.drugbank_xml)
    parser.add_argument("--output", type=Path, default=settings.drugbank_minimal_index)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="If >0, stop after N drugs (debug).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"DrugBank XML not found: {args.xml}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w") as out_f:
        for _, elem in ET.iterparse(args.xml, events=("end",)):
            if elem.tag != DRUG_TAG:
                continue
            record = parse_drug(elem, max_text_chars=args.max_text_chars)
            if record is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if args.limit and count >= args.limit:
                    break
            # Free memory for both top-level and nested <drug> elements.
            elem.clear()

    print(f"Wrote {count} DrugBank records to {args.output}")


if __name__ == "__main__":
    main()
