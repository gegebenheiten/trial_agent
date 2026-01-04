import math
import re
from typing import Dict, List, Tuple

STRICT_KEYWORDS = [
    "uncontrolled",
    "unstable",
    "severe",
    "history of",
    "prior malignancy",
    "hiv",
    "hepatitis",
    "contraindicated",
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9\-\+]+", text or ""))


def criteria_strictness(criteria: Dict) -> Tuple[float, Dict]:
    """
    Rough strictness proxy using text length, keyword hits,
    and numeric limits (if provided).
    """
    inclusion = criteria.get("inclusion_text", "") or ""
    exclusion = criteria.get("exclusion_text", "") or ""
    parsed = criteria.get("parsed", {}) or {}

    inc_words = _word_count(inclusion)
    exc_words = _word_count(exclusion)
    keyword_hits = sum(1 for kw in STRICT_KEYWORDS if kw in exclusion.lower())

    # Numeric gates: higher ECog max -> less strict; more prior lines allowed -> less strict
    ecog = parsed.get("ecog_max")
    prior = parsed.get("prior_lines_max")
    ecog_penalty = 0.0 if ecog is None else max(0.0, (1.5 - float(ecog)) / 1.5)
    prior_penalty = 0.0 if prior is None else max(0.0, (1.0 - float(prior) / 3))

    length_score = min(1.0, (inc_words + exc_words) / 800)
    keyword_score = min(1.0, keyword_hits / 5)
    numeric_score = max(ecog_penalty, prior_penalty)

    raw_score = (0.5 * length_score) + (0.3 * keyword_score) + (0.2 * numeric_score)
    score = float(round(min(1.0, raw_score), 3))

    details = {
        "inc_words": inc_words,
        "exc_words": exc_words,
        "keyword_hits": keyword_hits,
        "ecog_penalty": ecog_penalty,
        "prior_penalty": prior_penalty,
    }
    return score, details


def strictness_label(score: float) -> str:
    if score >= 0.65:
        return "very_strict"
    if score >= 0.45:
        return "moderate"
    return "permissive"

