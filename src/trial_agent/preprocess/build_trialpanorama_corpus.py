"""
Build a retrieval-friendly JSONL corpus from TrialPanorama parquet tables.

This script expects TrialPanorama parquet files under:
  data/trialpanorama/raw/

It produces:
  data/processed/trialpanorama_trials.jsonl

Notes:
- Uses Polars for parquet scanning/joining.
- Column names are inferred via a heuristic mapping; adjust if the schema changes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trial_agent.config import settings
from trial_agent.ingest.allowlist import load_trial_id_allowlist
from trial_agent.ingest.clean_text import normalize_whitespace

MAX_LIST_ITEMS = 20


def _pick_col(schema: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in schema:
            return col
    return None


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value]


def _normalize_list(values: List[str], max_items: int = MAX_LIST_ITEMS) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        text = normalize_whitespace(str(value))
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _normalize_struct_list(items: List[Dict], max_items: int = MAX_LIST_ITEMS) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cleaned: Dict[str, str] = {}
        for key, value in item.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = normalize_whitespace(value)
            cleaned[key] = value
        if cleaned:
            out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TrialPanorama JSONL for retrieval.")
    parser.add_argument("--raw-dir", type=Path, default=settings.trialpanorama_raw_dir)
    parser.add_argument("--output", type=Path, default=settings.trialpanorama_processed)
    parser.add_argument(
        "--allowlist-csv",
        type=Path,
        default=None,
        help="Optional CSV of trial IDs (nctid) to restrict output.",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, stop after N records (debug).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "polars is required to parse parquet. Install with `pip install polars`."
        ) from exc

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"TrialPanorama raw directory not found: {args.raw_dir}")

    def load_table(prefix: str) -> Tuple["pl.LazyFrame", Dict[str, str]]:
        files = sorted(args.raw_dir.glob(f"{prefix}*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found for {prefix} in {args.raw_dir}")
        lf = pl.scan_parquet(files)
        return lf, lf.schema

    def cast_key(lf: "pl.LazyFrame", col_name: str) -> "pl.LazyFrame":
        return lf.with_columns(pl.col(col_name).cast(pl.Utf8).alias(col_name))

    # Base studies table.
    studies_lf, studies_schema = load_table("studies")
    key_col = _pick_col(studies_schema, ["study_id", "nct_id", "nctid", "trial_id"])
    if not key_col:
        raise RuntimeError("Could not find study_id/nct_id column in studies table.")

    study_fields = {
        "study_source": ["study_source"],
        "trial_type": ["trial_type"],
        "title": ["title", "brief_title", "official_title"],
        "abstract": ["abstract", "brief_summary", "summary", "description"],
        "sponsor": ["sponsor"],
        "start_year": ["start_year"],
        "recruitment_status": ["recruitment_status", "overall_status"],
        "phase": ["phase", "study_phase", "trial_phase"],
        "actual_accrual": ["actual_accrual"],
        "target_accrual": ["target_accrual"],
        "min_age": ["min_age"],
        "max_age": ["max_age"],
        "min_age_unit": ["min_age_unit"],
        "max_age_unit": ["max_age_unit"],
        "sex": ["sex"],
        "healthy_volunteers": ["healthy_volunteers"],
    }
    study_cols = {
        out_key: _pick_col(studies_schema, candidates)
        for out_key, candidates in study_fields.items()
    }

    base_cols = [key_col] + [col for col in study_cols.values() if col]

    base = studies_lf.select(base_cols).unique(subset=[key_col])
    base = cast_key(base, key_col)
    if args.allowlist_csv:
        allowlist = load_trial_id_allowlist(args.allowlist_csv)
        if allowlist:
            base = base.filter(pl.col(key_col).is_in(list(allowlist)))

    def agg_list(prefix: str, value_candidates: List[str], alias: str):
        lf, schema = load_table(prefix)
        table_key = _pick_col(schema, [key_col, "study_id", "nct_id", "nctid", "trial_id"])
        if not table_key:
            return None
        value_col = _pick_col(schema, value_candidates)
        if not value_col:
            return None
        lf = cast_key(lf, table_key)
        return lf.group_by(table_key).agg(
            pl.col(value_col).drop_nulls().unique().alias(alias)
        )

    def agg_struct(prefix: str, field_candidates: Dict[str, List[str]], alias: str):
        lf, schema = load_table(prefix)
        table_key = _pick_col(schema, [key_col, "study_id", "nct_id", "nctid", "trial_id"])
        if not table_key:
            return None
        selected = {}
        for out_key, candidates in field_candidates.items():
            col = _pick_col(schema, candidates)
            if col:
                selected[out_key] = col
        if not selected:
            return None
        lf = cast_key(lf, table_key)
        lf = lf.select([table_key] + list(selected.values()))
        lf = lf.rename({col: out_key for out_key, col in selected.items()})
        return lf.group_by(table_key).agg(
            pl.struct(list(selected.keys())).drop_nulls().unique().alias(alias)
        )

    conditions = agg_struct(
        "conditions",
        {
            "study_source": ["study_source"],
            "condition_name": ["condition_name"],
            "condition_mesh_id": ["condition_mesh_id"],
            "condition_mesh_type": ["condition_mesh_type"],
        },
        "conditions",
    )
    drugs = agg_struct(
        "drugs",
        {
            "study_source": ["study_source"],
            "drug_moa_id": ["drug_moa_id"],
            "drug_name": ["drug_name"],
            "rx_normalized_name": ["rx_normalized_name"],
            "rx_source": ["rx_source"],
            "rx_mapping_id": ["rx_mapping_id"],
            "drugbank_name": ["drugbank_name"],
            "drugbank_id": ["drugbank_id"],
            "drugbank_mesh_id": ["drugbank_mesh_id"],
            "drug_source": ["drug_source"],
            "fda_approved": ["fda_approved"],
            "ema_approved": ["ema_approved"],
            "pmda_approved": ["pmda_approved"],
            "drug_description": ["drug_description"],
        },
        "drugs",
    )
    biomarkers = agg_struct(
        "biomarkers",
        {
            "study_source": ["study_source"],
            "biomarker_type": ["biomarker_type"],
            "biomarker_name": ["biomarker_name"],
            "biomarker_genes": ["biomarker_genes"],
        },
        "biomarkers",
    )
    endpoints = agg_struct(
        "endpoints",
        {
            "study_source": ["study_source"],
            "primary_endpoint": ["primary_endpoint"],
            "primary_endpoint_domain": ["primary_endpoint_domain"],
            "primary_endpoint_subdomain": ["primary_endpoint_subdomain"],
        },
        "endpoints",
    )
    outcomes = agg_struct(
        "outcomes",
        {
            "study_source": ["study_source"],
            "overall_status": ["overall_status"],
            "outcome_type": ["outcome_type"],
            "why_terminated": ["why_terminated"],
        },
        "outcomes",
    )

    results = agg_struct(
        "results",
        {
            "study_source": ["study_source"],
            "population": ["population"],
            "interventions": ["interventions", "intervention"],
            "outcomes": ["outcomes", "outcome"],
            "group_type": ["group_type", "arm_type"],
        },
        "results",
    )
    disposition = agg_struct(
        "disposition",
        {
            "study_source": ["study_source"],
            "intervention_type": ["intervention_type"],
            "intervention_name": ["intervention_name", "intervention"],
            "group_type": ["group_type"],
            "intervention_description": ["intervention_description"],
            "number_of_subjects": ["number_of_subjects", "subjects"],
        },
        "disposition",
    )
    adverse_events = agg_struct(
        "adverse_events",
        {
            "study_source": ["study_source"],
            "adverse_event_name": ["adverse_event_name", "event_name", "term"],
            "adverse_event_description": ["adverse_event_description", "event_description"],
            "is_serious": ["is_serious", "serious"],
            "meddra_id": ["meddra_id"],
            "meddra_id_type": ["meddra_id_type"],
        },
        "adverse_events",
    )

    # Join drugs with drug_moa to surface targets/genes per study.
    drug_moa_agg = None
    try:
        drugs_lf, drugs_schema = load_table("drugs")
        drug_moa_lf, drug_moa_schema = load_table("drug_moa")
        drug_key = _pick_col(drugs_schema, [key_col, "study_id", "nct_id", "nctid", "trial_id"])
        drug_moa_id = _pick_col(drugs_schema, ["drug_moa_id", "moa_id"])
        drug_moa_key = _pick_col(drug_moa_schema, ["drug_moa_id", "moa_id"])
        if drug_key and drug_moa_id and drug_moa_key:
            drugs_lf = cast_key(drugs_lf, drug_key)
            drugs_lf = drugs_lf.rename({drug_moa_id: "drug_moa_id", drug_key: "study_key"})
            drug_moa_lf = drug_moa_lf.rename({drug_moa_key: "drug_moa_id"})
            drug_moa_lf = cast_key(drug_moa_lf, "drug_moa_id")

            merged = drugs_lf.join(drug_moa_lf, on="drug_moa_id", how="left")
            merged = merged.rename({"study_key": key_col})

            fields = {}
            for out_key, candidates in {
                "drug_moa_id": ["drug_moa_id", "moa_id"],
                "drug_name": ["drug_name", "drug", "name", "intervention"],
                "drugbank_name": ["drugbank_name"],
                "drugbank_id": ["drugbank_id"],
                "rx_normalized_name": ["rx_normalized_name"],
                "target_name": ["target_name", "target"],
                "target_class": ["target_class"],
                "accession": ["accession"],
                "gene": ["gene", "gene_name"],
                "swissprot": ["swissprot"],
                "act_value": ["act_value"],
                "act_unit": ["act_unit"],
                "act_type": ["act_type", "action_type"],
                "act_comment": ["act_comment"],
                "act_source": ["act_source"],
                "relation": ["relation"],
                "moa": ["moa", "mechanism_of_action"],
                "moa_source": ["moa_source"],
                "act_source_url": ["act_source_url"],
                "moa_source_url": ["moa_source_url"],
                "action_type": ["action_type"],
                "tdl": ["tdl"],
                "organism": ["organism"],
            }.items():
                col = _pick_col(merged.schema, candidates)
                if col:
                    fields[out_key] = col

            if fields:
                merged = merged.select([key_col] + list(fields.values()))
                merged = merged.rename({col: out_key for out_key, col in fields.items()})
                drug_moa_agg = merged.group_by(key_col).agg(
                    pl.struct(list(fields.keys())).drop_nulls().unique().alias("drug_moa")
                )
    except FileNotFoundError:
        drug_moa_agg = None

    combined = base
    for agg in [
        conditions,
        drugs,
        biomarkers,
        endpoints,
        outcomes,
        results,
        disposition,
        adverse_events,
        drug_moa_agg,
    ]:
        if agg is not None:
            if key_col in agg.schema:
                combined = combined.join(agg, on=key_col, how="left")
            elif "study_key" in agg.schema:
                combined = combined.join(agg, left_on=key_col, right_on="study_key", how="left")
                combined = combined.drop("study_key")

    df = combined.collect(streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        for idx, row in enumerate(df.iter_rows(named=True)):
            trial_id = str(row.get(key_col))
            study: Dict[str, str] = {"study_id": trial_id}
            for out_key, col in study_cols.items():
                if not col:
                    continue
                value = row.get(col)
                if value is None:
                    continue
                if isinstance(value, str):
                    value = normalize_whitespace(value)
                study[out_key] = value

            conditions_list = _normalize_struct_list(_ensure_list(row.get("conditions")))
            drugs_list = _normalize_struct_list(_ensure_list(row.get("drugs")))
            biomarkers_list = _normalize_struct_list(_ensure_list(row.get("biomarkers")))
            endpoints_list = _normalize_struct_list(_ensure_list(row.get("endpoints")))
            outcomes_list = _normalize_struct_list(_ensure_list(row.get("outcomes")))
            results_list = _normalize_struct_list(_ensure_list(row.get("results")))
            disposition_list = _normalize_struct_list(_ensure_list(row.get("disposition")))
            adverse_events_list = _normalize_struct_list(_ensure_list(row.get("adverse_events")))
            drug_moa_list = _normalize_struct_list(_ensure_list(row.get("drug_moa")))

            record = {
                "trial_id": trial_id,
                "source": "trialpanorama",
                "study": study,
                "conditions": conditions_list,
                "drugs": drugs_list,
                "drug_moa": drug_moa_list,
                "biomarkers": biomarkers_list,
                "endpoints": endpoints_list,
                "outcomes": outcomes_list,
                "results": results_list,
                "disposition": disposition_list,
                "adverse_events": adverse_events_list,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.limit and (idx + 1) >= args.limit:
                break

    print(f"Wrote {min(len(df), args.limit) if args.limit else len(df)} records to {args.output}")


if __name__ == "__main__":
    main()
