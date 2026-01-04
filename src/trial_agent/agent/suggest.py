from collections import Counter
from typing import Dict, List, Tuple

from trial_agent.features.featurize import featurize_trial
from trial_agent.features.strictness import strictness_label


def _most_common_endpoint(retrieved: List[Dict]) -> str:
    counter: Counter = Counter()
    for item in retrieved:
        ep = (item["trial"].get("endpoints", {}).get("parsed", {}) or {}).get("primary_type", "")
        if ep:
            counter.update([ep.lower()])
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _pick_evidence(retrieved: List[Dict], predicate) -> Dict:
    for item in retrieved:
        if predicate(item["trial"]):
            return {"trial_id": item["trial"].get("trial_id", ""), "snippet": item.get("snippet", "")}
    if retrieved:
        top = retrieved[0]
        return {"trial_id": top["trial"].get("trial_id", ""), "snippet": top.get("snippet", "")}
    return {"trial_id": "", "snippet": ""}


def generate_recommendations(current_trial: Dict, retrieved: List[Dict]) -> Tuple[Dict, List[str]]:
    recs = {"design": [], "criteria": [], "endpoints": []}
    open_questions: List[str] = []

    current_feats = featurize_trial(current_trial)
    retrieved_feats = [featurize_trial(item["trial"]) for item in retrieved]
    avg_strictness = sum(f["criteria_strictness"] for f in retrieved_feats) / max(1, len(retrieved_feats))
    common_endpoint = _most_common_endpoint(retrieved)

    # Design suggestions
    if not current_feats["is_randomized"]:
        evidence = _pick_evidence(retrieved, lambda t: "random" in (t.get("design", {}).get("allocation", "") or "").lower())
        recs["design"].append(
            {
                "action": "Consider adding randomization or justify single-arm design",
                "rationale": "Most similar trials used randomization to strengthen interpretability.",
                "evidence_trials": [evidence],
                "risk_note": "May impact feasibility and requires statistical review.",
                "change_impact": "medium",
            }
        )
        open_questions.append("Is randomization feasible given recruitment and control availability?")

    if not current_feats["has_control"]:
        evidence = _pick_evidence(retrieved, lambda t: "control" in str(t.get("design", {}).get("arms", "")).lower())
        recs["design"].append(
            {
                "action": "Add an active comparator or provide external control rationale",
                "rationale": "Comparator arms in similar trials improved credibility of outcomes.",
                "evidence_trials": [evidence],
                "risk_note": "Need clinical agreement on comparator and safety monitoring.",
                "change_impact": "large",
            }
        )

    # Criteria suggestions
    if current_feats["criteria_strictness"] > avg_strictness + 0.1:
        evidence = _pick_evidence(retrieved, lambda t: True)
        recs["criteria"].append(
            {
                "action": "Loosen inclusion/exclusion thresholds where safe (e.g., ECOG, prior lines)",
                "rationale": "Criteria appear stricter than peers and may slow enrollment.",
                "evidence_trials": [evidence],
                "risk_note": "Relaxing safety-related exclusions requires medical oversight.",
                "change_impact": "medium",
            }
        )
        open_questions.append("Which criteria can be relaxed without compromising safety?")
    elif current_feats["criteria_strictness"] < avg_strictness - 0.1:
        evidence = _pick_evidence(retrieved, lambda t: True)
        recs["criteria"].append(
            {
                "action": "Review exclusion list to ensure high-risk populations are addressed",
                "rationale": "Criteria are more permissive than similar trials; ensure safety coverage.",
                "evidence_trials": [evidence],
                "risk_note": "Focus on hepatic/renal function and CNS metastases considerations.",
                "change_impact": "small",
            }
        )

    # Endpoint suggestions
    current_primary = (current_trial.get("endpoints", {}).get("parsed", {}) or {}).get("primary_type", "")
    if common_endpoint and current_primary.lower() != common_endpoint.lower():
        evidence = _pick_evidence(retrieved, lambda t: (t.get("endpoints", {}).get("parsed", {}) or {}).get("primary_type", "").lower() == common_endpoint.lower())
        recs["endpoints"].append(
            {
                "action": f"Align primary endpoint with common choice: {common_endpoint.upper()}",
                "rationale": "Using the prevalent endpoint in similar phase/indication improves comparability.",
                "evidence_trials": [evidence],
                "risk_note": "Confirm statistical powering and assessment schedule.",
                "change_impact": "medium",
            }
        )

    if not current_trial.get("endpoints", {}).get("primary"):
        evidence = _pick_evidence(retrieved, lambda t: True)
        recs["endpoints"].append(
            {
                "action": "Define primary endpoint with time frame and assessment criteria",
                "rationale": "Current protocol lacks a clear primary endpoint; similar trials specify RECIST and timing.",
                "evidence_trials": [evidence],
                "risk_note": "Requires statistical design review.",
                "change_impact": "large",
            }
        )

    if strictness_label(current_feats["criteria_strictness"]) == "very_strict":
        open_questions.append("Is a rescue recruitment strategy in place if enrollment lags?")

    return recs, open_questions

