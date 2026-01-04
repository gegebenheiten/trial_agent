from typing import Dict, Tuple

from trial_agent.features.strictness import criteria_strictness

PHASE_ORDER = {
    "phase 1": 1.0,
    "phase 1/2": 1.5,
    "phase 2": 2.0,
    "phase 2/3": 2.5,
    "phase 3": 3.0,
    "phase 4": 4.0,
}

ENDPOINT_RISK = {
    "os": 0.9,
    "pfs": 0.7,
    "orr": 0.5,
    "dcr": 0.45,
    "dor": 0.55,
    "safety": 0.4,
    "qol": 0.35,
}


def phase_value(phase: str) -> float:
    return PHASE_ORDER.get(phase.lower(), 2.0) if phase else 2.0


def endpoint_value(primary_type: str) -> float:
    return ENDPOINT_RISK.get(primary_type.lower(), 0.5) if primary_type else 0.5


def detect_control(design: Dict) -> Tuple[bool, str]:
    arms = design.get("arms", []) or []
    control_terms = ["placebo", "standard of care", "soc", "control"]
    for arm in arms:
        name = (arm.get("name") or "").lower()
        desc = (arm.get("description") or "").lower()
        if any(term in name or term in desc for term in control_terms):
            return True, arm.get("name", "control")
    return False, ""


def _primary_endpoint_type(endpoints: Dict) -> str:
    if isinstance(endpoints, dict):
        parsed = endpoints.get("parsed", {}) or {}
        return str(parsed.get("primary_type") or parsed.get("primary_type_label") or "")
    if isinstance(endpoints, list):
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            for key in (
                "primary_endpoint_domain",
                "primary_endpoint_subdomain",
                "primary_endpoint",
            ):
                value = ep.get(key)
                if value:
                    return str(value)
    return ""


def featurize_trial(trial: Dict) -> Dict:
    design = trial.get("design", {}) or {}
    endpoints = trial.get("endpoints", {}) or {}
    criteria = trial.get("criteria", {}) or {}

    strictness_score, strictness_details = criteria_strictness(criteria)
    has_control, control_name = detect_control(design)
    allocation = (design.get("allocation", "") or "").lower()
    masking = (design.get("masking", "") or "").lower()
    is_randomized = ("randomized" in allocation) and ("non" not in allocation)
    is_blinded = not ("none" in masking or "open" in masking)
    primary_type = _primary_endpoint_type(endpoints)

    features = {
        "phase_value": phase_value(trial.get("phase", "")),
        "is_randomized": is_randomized,
        "has_control": has_control,
        "control_name": control_name,
        "is_blinded": is_blinded,
        "primary_endpoint_type": primary_type,
        "primary_endpoint_value": endpoint_value(primary_type),
        "criteria_strictness": strictness_score,
        "criteria_details": strictness_details,
        "arm_count": len(design.get("arms", []) or []),
    }
    return features
