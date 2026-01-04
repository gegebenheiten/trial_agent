import re
from typing import List


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace and trim.
    Keeps the text human-readable for prompts and scoring.
    """
    if not text:
        return ""
    cleaned = WHITESPACE_RE.sub(" ", text)
    return cleaned.strip()


def split_into_sentences(text: str, max_sentences: int = 4) -> List[str]:
    """
    Heuristic sentence splitter that is good enough for demo snippets.
    """
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [normalize_whitespace(s) for s in sentences if s.strip()]
    return sentences[:max_sentences]


def short_snippet(text: str, limit: int = 400) -> str:
    """
    Return a truncated snippet with ellipsis to keep prompts compact.
    """
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

