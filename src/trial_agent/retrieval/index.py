from typing import Dict, List

from trial_agent.retrieval.embed import vectorize


def _extract_conditions(trial: Dict) -> List[str]:
    if "condition" in trial:
        return trial.get("condition", []) or []
    conditions = trial.get("conditions", []) or []
    if conditions and isinstance(conditions[0], dict):
        return [c.get("condition_name", "") for c in conditions if c.get("condition_name")]
    return conditions


def _extract_drugs_ctgov(trial: Dict) -> List[str]:
    names: List[str] = []
    for intervention in trial.get("interventions", []) or []:
        if not isinstance(intervention, dict):
            continue
        names.append(intervention.get("name", ""))
        names.append(intervention.get("type", ""))
    return names


def _extract_drugs_panorama(trial: Dict) -> List[str]:
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


def _extract_drug_targets(trial: Dict) -> List[str]:
    out: List[str] = []
    drug_moa = trial.get("drug_moa", {}) or {}
    if isinstance(drug_moa, dict):
        out.extend(drug_moa.get("target_names", []) or [])
        out.extend(drug_moa.get("target_genes", []) or [])
        for detail in drug_moa.get("details", []) or []:
            out.extend(
                [
                    detail.get("drug_name", ""),
                    detail.get("target_name", ""),
                    detail.get("gene", ""),
                    detail.get("moa", ""),
                ]
            )
    elif isinstance(drug_moa, list):
        for detail in drug_moa:
            if not isinstance(detail, dict):
                continue
            out.extend(
                [
                    detail.get("drug_name", ""),
                    detail.get("target_name", ""),
                    detail.get("gene", ""),
                    detail.get("moa", ""),
                ]
            )
    return out


def _extract_outcomes(trial: Dict, prefer_summary: bool = True) -> List[str]:
    out: List[str] = []
    if prefer_summary:
        outcomes = trial.get("outcomes_summary", {}) or {}
        if isinstance(outcomes, dict) and outcomes:
            out.extend(outcomes.get("overall_status", []) or [])
            out.extend(outcomes.get("outcome_type", []) or [])
            out.extend(outcomes.get("why_terminated", []) or [])
            return out
    for outcome in trial.get("outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        out.extend(
            [
                outcome.get("overall_status", ""),
                outcome.get("outcome_type", ""),
                outcome.get("why_terminated", ""),
            ]
        )
    return out


def _extract_results(trial: Dict, prefer_summary: bool = True) -> List[str]:
    out: List[str] = []
    if prefer_summary:
        for result in trial.get("results_summary", []) or []:
            if not isinstance(result, dict):
                continue
            out.extend(
                [
                    result.get("population", ""),
                    result.get("interventions", ""),
                    result.get("outcomes", ""),
                ]
            )
    for result in trial.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        out.extend(
            [
                result.get("population", ""),
                result.get("interventions", ""),
                result.get("outcomes", ""),
            ]
        )
    return out


def _extract_disposition(trial: Dict, prefer_summary: bool = True) -> List[str]:
    out: List[str] = []
    if prefer_summary:
        for disp in trial.get("disposition_summary", []) or []:
            if not isinstance(disp, dict):
                continue
            out.extend([disp.get("intervention_name", ""), disp.get("group_type", "")])
    for disp in trial.get("disposition", []) or []:
        if not isinstance(disp, dict):
            continue
        out.extend([disp.get("intervention_name", ""), disp.get("group_type", "")])
    return out


def _extract_adverse_events(trial: Dict, prefer_summary: bool = True) -> List[str]:
    out: List[str] = []
    if prefer_summary:
        for ae in trial.get("adverse_events_summary", []) or []:
            if not isinstance(ae, dict):
                continue
            out.append(ae.get("adverse_event_name", ""))
    for ae in trial.get("adverse_events", []) or []:
        if not isinstance(ae, dict):
            continue
        out.append(ae.get("adverse_event_name", ""))
    return out


def _extract_design_ctgov(trial: Dict) -> List[str]:
    out: List[str] = []
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


def _extract_study_panorama(trial: Dict) -> List[str]:
    out: List[str] = []
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


def trial_to_field_text(trial: Dict, focus: str) -> str:
    parts: List[str] = []
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial

    def add(value) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)

    def extend(values: List) -> None:
        for value in values:
            add(value)

    focus_key = (focus or "full").strip().lower()
    if focus_key in {"full", "all", "default"}:
        extend(_extract_conditions(trial))
        study = trial.get("study", {}) or {}
        add(trial.get("phase", "") or study.get("phase", ""))
        if is_trialpanorama:
            add(study.get("trial_type", ""))
            add(study.get("recruitment_status", ""))
            add(study.get("sex", ""))
            extend(_extract_drugs_panorama(trial))
        else:
            extend(_extract_drugs_ctgov(trial))
        extend(_extract_endpoints(trial))
        extend(_extract_biomarkers(trial))
        extend(_extract_drug_targets(trial))
        if is_trialpanorama:
            extend(_extract_outcomes(trial, prefer_summary=False))
            extend(_extract_results(trial, prefer_summary=False))
            extend(_extract_disposition(trial, prefer_summary=False))
            extend(_extract_adverse_events(trial, prefer_summary=False))
        else:
            extend(_extract_outcomes(trial, prefer_summary=True))
            extend(_extract_results(trial, prefer_summary=True))
            extend(_extract_disposition(trial, prefer_summary=True))
            extend(_extract_adverse_events(trial, prefer_summary=True))
            criteria = trial.get("criteria", {}) or {}
            add(criteria.get("inclusion_text", ""))
            add(criteria.get("exclusion_text", ""))
        return " ".join([p for p in parts if p])

    if focus_key in {"condition", "conditions"}:
        extend(_extract_conditions(trial))
    elif focus_key in {"drug", "drugs", "intervention", "interventions"}:
        if is_trialpanorama:
            extend(_extract_drugs_panorama(trial))
        else:
            extend(_extract_drugs_ctgov(trial))
        extend(_extract_drug_targets(trial))
    elif focus_key in {"biomarker", "biomarkers"}:
        extend(_extract_biomarkers(trial))
    elif focus_key in {"endpoint", "endpoints"}:
        extend(_extract_endpoints(trial))
    elif focus_key in {"study", "studies", "trial"}:
        if is_trialpanorama:
            extend(_extract_study_panorama(trial))
        else:
            extend(_extract_design_ctgov(trial))
    elif focus_key in {"design"}:
        if is_trialpanorama:
            extend(_extract_study_panorama(trial))
        else:
            extend(_extract_design_ctgov(trial))
    elif focus_key in {"outcome", "outcomes", "results"}:
        extend(_extract_outcomes(trial, prefer_summary=not is_trialpanorama))
        extend(_extract_results(trial, prefer_summary=not is_trialpanorama))
    else:
        return trial_to_field_text(trial, "full")

    return " ".join([p for p in parts if p])


def trial_to_corpus_text(trial: Dict) -> str:
    """
    Combine salient fields into a single bag-of-words string.
    This keeps retrieval simple while using multiple trial facets.
    """
    return trial_to_field_text(trial, "full")


def build_in_memory_index(trials: List[Dict]) -> List[Dict]:
    """
    Create an in-memory index where each item contains the trial data and a sparse vector.
    """
    indexed: List[Dict] = []
    for trial in trials:
        text = trial_to_corpus_text(trial)
        indexed.append({"trial": trial, "vector": vectorize([text])})
    return indexed


def trial_to_full_chunks(trial: Dict, max_chars: int = 0) -> List[str]:
    chunks: List[str] = []
    is_trialpanorama = trial.get("source") == "trialpanorama" or "study" in trial

    def _normalize(values: List) -> str:
        parts: List[str] = []
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if value:
                parts.append(value)
        text = " ".join(parts)
        if max_chars and max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        return text.strip()

    def add_chunk(values: List) -> None:
        text = _normalize(values)
        if text:
            chunks.append(text)

    add_chunk(_extract_conditions(trial))

    study_values: List[str] = []
    study = trial.get("study", {}) or {}
    study_values.append(trial.get("phase", "") or study.get("phase", ""))
    if is_trialpanorama and isinstance(study, dict):
        study_values.extend(
            [
                study.get("trial_type", ""),
                study.get("recruitment_status", ""),
                study.get("sex", ""),
            ]
        )
    add_chunk(study_values)

    if is_trialpanorama:
        add_chunk(_extract_drugs_panorama(trial))
    else:
        add_chunk(_extract_drugs_ctgov(trial))

    add_chunk(_extract_endpoints(trial))
    add_chunk(_extract_biomarkers(trial))
    add_chunk(_extract_drug_targets(trial))

    if is_trialpanorama:
        add_chunk(_extract_outcomes(trial, prefer_summary=False))
        add_chunk(_extract_results(trial, prefer_summary=False))
        add_chunk(_extract_disposition(trial, prefer_summary=False))
        add_chunk(_extract_adverse_events(trial, prefer_summary=False))
    else:
        add_chunk(_extract_outcomes(trial, prefer_summary=True))
        add_chunk(_extract_results(trial, prefer_summary=True))
        add_chunk(_extract_disposition(trial, prefer_summary=True))
        add_chunk(_extract_adverse_events(trial, prefer_summary=True))
        criteria = trial.get("criteria", {}) or {}
        add_chunk(
            [
                criteria.get("inclusion_text", ""),
                criteria.get("exclusion_text", ""),
            ]
        )

    return chunks


def trial_to_field_chunks(trial: Dict, focus: str, max_chars: int = 0) -> List[str]:
    focus_key = (focus or "full").strip().lower()
    if focus_key in {"full", "all", "default"}:
        return trial_to_full_chunks(trial, max_chars=max_chars)
    text = trial_to_field_text(trial, focus_key)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return [text] if text.strip() else []
