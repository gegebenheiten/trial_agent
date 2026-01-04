"""
Build an addition.parquet file with extra TrialPanorama features:
 - criteria (from ClinicalTrials.gov XML)
 - smiles (from DrugBank XML, keyed by drugbank_id)
 - ICD-10-CM codes (via clinicaltables API from condition_name)
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.append("src")

from trial_agent.ingest.clean_text import normalize_whitespace


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../Trialbench/trial_agent
REPO_ROOT = PROJECT_ROOT.parent  # .../Trialbench

RAW_DIR = PROJECT_ROOT / "data" / "trialpanorama" / "raw"
DEFAULT_XML_ROOT = REPO_ROOT / "data_curation" / "raw_data"
DEFAULT_DRUGBANK_XML = PROJECT_ROOT / "data" / "drugbank" / "full_database.xml"
DEFAULT_OUTPUT = RAW_DIR / "addition.parquet"
DEFAULT_ICD_CACHE = PROJECT_ROOT / "data" / "processed" / "icd10_cache.json"

NS_URI = "http://www.drugbank.ca"
NS = {"db": NS_URI}
DRUG_TAG = f"{{{NS_URI}}}drug"


def load_studies() -> pd.DataFrame:
    paths = [RAW_DIR / "studies.parquet", RAW_DIR / "studies_part_2.parquet"]
    frames = [pd.read_parquet(p, columns=["study_id", "study_source"]) for p in paths]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_drugs() -> pd.DataFrame:
    paths = [RAW_DIR / "drugs.parquet", RAW_DIR / "drugs_part_2.parquet"]
    frames = [pd.read_parquet(p, columns=["study_id", "drugbank_id"]) for p in paths]
    return pd.concat(frames, ignore_index=True)


def load_conditions() -> pd.DataFrame:
    paths = [RAW_DIR / "conditions.parquet", RAW_DIR / "conditions_part_2.parquet"]
    frames = [pd.read_parquet(p, columns=["study_id", "condition_name"]) for p in paths]
    return pd.concat(frames, ignore_index=True)


def xml_path_for_id(nctid: str, xml_root: Path) -> Path:
    prefix = nctid[:7] + "xxxx"
    return xml_root / prefix / f"{nctid}.xml"


def split_criteria(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    lower = text.lower()
    inc_idx = lower.find("inclusion")
    exc_idx = lower.find("exclusion")
    if inc_idx != -1 and exc_idx != -1:
        if inc_idx < exc_idx:
            inclusion = text[inc_idx:exc_idx]
            exclusion = text[exc_idx:]
        else:
            exclusion = text[exc_idx:inc_idx]
            inclusion = text[inc_idx:]
    elif inc_idx != -1:
        inclusion, exclusion = text[inc_idx:], ""
    elif exc_idx != -1:
        inclusion, exclusion = "", text[exc_idx:]
    else:
        inclusion, exclusion = text, ""
    return normalize_whitespace(inclusion), normalize_whitespace(exclusion)


def parse_criteria(nctid: str, xml_root: Path) -> Tuple[str, str, str]:
    xml_path = xml_path_for_id(nctid, xml_root)
    if not xml_path.exists():
        return "", "", "missing"
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        textblock = root.findtext("eligibility/criteria/textblock") or ""
        inclusion_text, exclusion_text = split_criteria(textblock)
        return inclusion_text, exclusion_text, ""
    except Exception:
        return "", "", "error"


def _primary_drugbank_id(drug: ET.Element) -> str:
    ids = drug.findall("db:drugbank-id", NS)
    for el in ids:
        if el.get("primary") == "true" and el.text:
            return el.text.strip()
    for el in ids:
        if el.text:
            return el.text.strip()
    return ""


def _extract_smiles(drug: ET.Element) -> str:
    for prop in drug.findall("db:calculated-properties/db:property", NS):
        kind = (prop.findtext("db:kind", default="", namespaces=NS) or "").strip()
        if kind != "SMILES":
            continue
        value = (prop.findtext("db:value", default="", namespaces=NS) or "").strip()
        if value:
            return value
    return ""


def build_drugbank_smiles(xml_path: Path) -> Dict[str, str]:
    smiles_by_id: Dict[str, str] = {}
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != DRUG_TAG:
            continue
        drugbank_id = _primary_drugbank_id(elem)
        smiles = _extract_smiles(elem)
        if drugbank_id and smiles:
            smiles_by_id[drugbank_id] = smiles
        elem.clear()
    return smiles_by_id


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for val in values:
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def build_smiles_by_study(drugs: pd.DataFrame, smiles_by_id: Dict[str, str]) -> Dict[str, List[str]]:
    df = drugs.copy()
    df = df[df["drugbank_id"].notna()]
    df["drugbank_id"] = df["drugbank_id"].astype(str).str.strip()
    df = df[df["drugbank_id"].str.startswith("DB")]
    df["smiles"] = df["drugbank_id"].map(smiles_by_id)
    df = df[df["smiles"].notna()]
    grouped = df.groupby("study_id")["smiles"].agg(_dedupe)
    return grouped.to_dict()


def load_icd_cache(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_icd_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_icd_codes(session: requests.Session, term: str, max_list: int) -> List[str]:
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {"terms": term, "sf": "name", "maxList": max_list}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) < 4 or not data[3]:
        return []
    codes: List[str] = []
    for row in data[3]:
        if not row or not row[0]:
            continue
        code = row[0]
        if code not in codes:
            codes.append(code)
    return codes


def ensure_icd_cache(
    condition_names: Iterable[str],
    cache: Dict[str, List[str]],
    max_list: int,
    sleep_s: float,
    cache_path: Path,
) -> Dict[str, List[str]]:
    session = requests.Session()
    updated = 0
    for idx, term in enumerate(condition_names, 1):
        if term in cache:
            continue
        codes: List[str] = []
        for attempt in range(3):
            try:
                codes = fetch_icd_codes(session, term, max_list=max_list)
                break
            except Exception:
                time.sleep(1.0 + attempt)
        cache[term] = codes
        updated += 1
        if updated % 200 == 0:
            save_icd_cache(cache_path, cache)
        if sleep_s:
            time.sleep(sleep_s)
        if idx % 500 == 0:
            print(f"[ICD] processed {idx} terms (new {updated})")
    save_icd_cache(cache_path, cache)
    return cache


def _merge_code_lists(series: Iterable[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for codes in series:
        if not codes:
            continue
        for code in codes:
            if code in seen:
                continue
            seen.add(code)
            out.append(code)
    return out


def build_icd_by_study(conditions: pd.DataFrame, icd_cache: Dict[str, List[str]]) -> Dict[str, List[str]]:
    df = conditions.copy()
    df = df[df["condition_name"].notna()]
    df["condition_name"] = df["condition_name"].astype(str).str.strip()
    df = df[df["condition_name"] != ""]
    df["icd_codes"] = df["condition_name"].map(icd_cache)
    df = df[df["icd_codes"].map(bool)]
    grouped = df.groupby("study_id")["icd_codes"].agg(_merge_code_lists)
    return grouped.to_dict()


def write_addition(
    studies: pd.DataFrame,
    xml_root: Path,
    smiles_by_study: Dict[str, List[str]],
    icd_by_study: Dict[str, List[str]],
    output: Path,
    chunk_size: int,
    limit: int,
) -> None:
    schema = pa.schema(
        [
            ("study_id", pa.string()),
            ("study_source", pa.string()),
            ("criteria_inclusion_text", pa.string()),
            ("criteria_exclusion_text", pa.string()),
            ("smiles", pa.list_(pa.string())),
            ("icd_codes", pa.list_(pa.string())),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    writer = pq.ParquetWriter(output, schema=schema)
    rows: List[Dict[str, object]] = []
    missing_xml = 0
    parse_errors = 0

    for idx, (study_id, study_source) in enumerate(
        studies[["study_id", "study_source"]].itertuples(index=False, name=None), 1
    ):
        if limit and idx > limit:
            break
        inclusion = ""
        exclusion = ""
        if study_source == "ClinicalTrials.gov" and isinstance(study_id, str):
            inclusion, exclusion, status = parse_criteria(study_id, xml_root)
            if status == "missing":
                missing_xml += 1
            elif status == "error":
                parse_errors += 1

        smiles = smiles_by_study.get(study_id) or None
        icd_codes = icd_by_study.get(study_id) or None

        rows.append(
            {
                "study_id": study_id,
                "study_source": study_source,
                "criteria_inclusion_text": inclusion or None,
                "criteria_exclusion_text": exclusion or None,
                "smiles": smiles,
                "icd_codes": icd_codes,
            }
        )

        if len(rows) >= chunk_size:
            table = pa.Table.from_pylist(rows, schema=schema)
            writer.write_table(table)
            rows = []

        if idx % 10000 == 0:
            print(
                f"[ADD] processed {idx} studies | missing_xml={missing_xml} "
                f"| parse_errors={parse_errors}"
            )

    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)

    writer.close()
    print(
        f"Wrote {output} | missing_xml={missing_xml} | parse_errors={parse_errors}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TrialPanorama addition.parquet with criteria/smiles/ICD codes.")
    parser.add_argument("--xml-root", type=Path, default=DEFAULT_XML_ROOT)
    parser.add_argument("--drugbank-xml", type=Path, default=DEFAULT_DRUGBANK_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--icd-cache", type=Path, default=DEFAULT_ICD_CACHE)
    parser.add_argument("--icd-max", type=int, default=50)
    parser.add_argument("--icd-sleep", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml_root.exists():
        raise FileNotFoundError(f"CT.gov XML root not found: {args.xml_root}")
    if not args.drugbank_xml.exists():
        raise FileNotFoundError(f"DrugBank XML not found: {args.drugbank_xml}")

    studies = load_studies()
    print(f"Loaded {len(studies)} studies")

    print("Parsing DrugBank SMILES...")
    smiles_by_id = build_drugbank_smiles(args.drugbank_xml)
    print(f"Loaded {len(smiles_by_id)} DrugBank SMILES entries")

    print("Building smiles-by-study map...")
    drugs = load_drugs()
    if args.limit:
        study_subset = set(studies["study_id"].head(args.limit))
        drugs = drugs[drugs["study_id"].isin(study_subset)]
    smiles_by_study = build_smiles_by_study(drugs, smiles_by_id)
    print(f"Built smiles for {len(smiles_by_study)} studies")

    print("Loading conditions...")
    conditions = load_conditions()
    if args.limit:
        study_subset = set(studies["study_id"].head(args.limit))
        conditions = conditions[conditions["study_id"].isin(study_subset)]
    condition_names = sorted(
        {name for name in conditions["condition_name"].dropna().astype(str).str.strip().unique() if name}
    )
    print(f"Found {len(condition_names)} unique condition names")

    icd_cache = load_icd_cache(args.icd_cache)
    icd_cache = ensure_icd_cache(
        condition_names,
        icd_cache,
        args.icd_max,
        args.icd_sleep,
        args.icd_cache,
    )

    print("Building ICD-by-study map...")
    icd_by_study = build_icd_by_study(conditions, icd_cache)
    print(f"Built ICD codes for {len(icd_by_study)} studies")

    print("Writing addition.parquet...")
    write_addition(
        studies=studies,
        xml_root=args.xml_root,
        smiles_by_study=smiles_by_study,
        icd_by_study=icd_by_study,
        output=args.output,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
