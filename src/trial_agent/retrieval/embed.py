import re
from collections import Counter
from typing import Iterable, List

TOKEN_RE = re.compile(r"[A-Za-z0-9\-\+]+")


def tokenize(text: str) -> List[str]:
    """Basic tokenization over alphanumerics and plus/minus signs."""
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def vectorize(texts: Iterable[str]) -> Counter:
    """
    Extremely small-footprint embedding: a sparse token counter.
    Good enough for lexical similarity and avoids external dependencies.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter


def jaccard_similarity(a: Counter, b: Counter) -> float:
    """Compute Jaccard similarity on token sets."""
    if not a or not b:
        return 0.0
    set_a = set(a.keys())
    set_b = set(b.keys())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union

