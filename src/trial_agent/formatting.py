from typing import Any, Dict, List


def _truncate(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_list(
    items: List[Dict[str, Any]],
    fields: List[str],
    max_items: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        trimmed: Dict[str, Any] = {}
        for field in fields:
            value = item.get(field)
            if value is None or value == "":
                continue
            if isinstance(value, str):
                value = _truncate(value, text_limit)
            trimmed[field] = value
        if trimmed:
            out.append(trimmed)
        if len(out) >= max_items:
            break
    return out


def compact_trial_for_prompt(
    trial: Dict[str, Any],
    max_list_items: int = 5,
    text_limit: int = 240,
    include_outcomes: bool = True,
) -> Dict[str, Any]:
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial
    if is_trialpanorama:
        study = trial.get("study", {}) or {}
        study_fields = [
            "study_source",
            "trial_type",
            "title",
            "abstract",
            "sponsor",
            "start_year",
            "recruitment_status",
            "phase",
            "actual_accrual",
            "target_accrual",
            "min_age",
            "max_age",
            "min_age_unit",
            "max_age_unit",
            "sex",
            "healthy_volunteers",
        ]
        compact_study: Dict[str, Any] = {}
        for field in study_fields:
            value = study.get(field)
            if value is None or value == "":
                continue
            if isinstance(value, str):
                value = _truncate(value, text_limit)
            compact_study[field] = value

        payload = {
            "trial_id": trial.get("trial_id"),
            "study": compact_study,
            "conditions": _compact_list(
                trial.get("conditions", []) or [],
                ["condition_name", "condition_mesh_id", "condition_mesh_type"],
                max_list_items,
                text_limit,
            ),
            "drugs": _compact_list(
                trial.get("drugs", []) or [],
                [
                    "drug_name",
                    "rx_normalized_name",
                    "drugbank_id",
                    "drug_moa_id",
                    "fda_approved",
                    "ema_approved",
                    "pmda_approved",
                ],
                max_list_items,
                text_limit,
            ),
            "biomarkers": _compact_list(
                trial.get("biomarkers", []) or [],
                ["biomarker_name", "biomarker_genes", "biomarker_type"],
                max_list_items,
                text_limit,
            ),
            "endpoints": _compact_list(
                trial.get("endpoints", []) or [],
                [
                    "primary_endpoint",
                    "primary_endpoint_domain",
                    "primary_endpoint_subdomain",
                ],
                max_list_items,
                text_limit,
            ),
            "results": _compact_list(
                trial.get("results", []) or [],
                ["population", "interventions", "outcomes", "group_type"],
                max_list_items,
                text_limit,
            ),
            "disposition": _compact_list(
                trial.get("disposition", []) or [],
                [
                    "intervention_type",
                    "intervention_name",
                    "group_type",
                    "number_of_subjects",
                ],
                max_list_items,
                text_limit,
            ),
            "adverse_events": _compact_list(
                trial.get("adverse_events", []) or [],
                ["adverse_event_name", "is_serious", "meddra_id"],
                max_list_items,
                text_limit,
            ),
            "drug_moa": _compact_list(
                trial.get("drug_moa", []) or [],
                ["drug_name", "target_name", "gene", "moa", "act_type"],
                max_list_items,
                text_limit,
            ),
        }
        if include_outcomes:
            payload["outcomes"] = _compact_list(
                trial.get("outcomes", []) or [],
                ["overall_status", "outcome_type", "why_terminated"],
                max_list_items,
                text_limit,
            )
        return payload

    criteria = trial.get("criteria", {}) or {}
    endpoints = trial.get("endpoints", {}) or {}
    design = trial.get("design", {}) or {}

    return {
        "trial_id": trial.get("trial_id"),
        "condition": (trial.get("condition", []) or [])[:max_list_items],
        "phase": trial.get("phase", ""),
        "interventions": (trial.get("interventions", []) or [])[:max_list_items],
        "design": {
            "allocation": design.get("allocation", ""),
            "intervention_model": design.get("intervention_model", ""),
            "masking": design.get("masking", ""),
            "primary_purpose": design.get("primary_purpose", ""),
            "arms": [
                {"name": arm.get("name", "")}
                for arm in (design.get("arms", []) or [])[:max_list_items]
            ],
            "dose": design.get("dose", ""),
        },
        "criteria": {
            "inclusion_text": _truncate(criteria.get("inclusion_text", ""), text_limit),
            "exclusion_text": _truncate(criteria.get("exclusion_text", ""), text_limit),
            "parsed": criteria.get("parsed", {}),
        },
        "endpoints": {
            "primary": [
                {
                    "name": ep.get("name", ""),
                    "time_frame": ep.get("time_frame", ""),
                }
                for ep in (endpoints.get("primary", []) or [])[:max_list_items]
            ],
            "secondary": [
                {"name": ep.get("name", "")}
                for ep in (endpoints.get("secondary", []) or [])[:max_list_items]
            ],
            "parsed": endpoints.get("parsed", {}),
        },
    }
