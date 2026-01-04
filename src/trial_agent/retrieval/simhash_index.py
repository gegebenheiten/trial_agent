import hashlib
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from trial_agent.retrieval.embed import tokenize

DEFAULT_BITS = 64
DEFAULT_BANDS = 4
DEFAULT_BAND_SIZE = 16


def _hash_token(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def simhash(tokens: Iterable[str], bits: int = DEFAULT_BITS) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    weights = Counter(tokens)
    vector = [0] * bits
    for token, weight in weights.items():
        hashed = _hash_token(token)
        for idx in range(bits):
            if (hashed >> idx) & 1:
                vector[idx] += weight
            else:
                vector[idx] -= weight
    fingerprint = 0
    for idx, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << idx
    return fingerprint


def simhash_from_text(text: str, bits: int = DEFAULT_BITS) -> int:
    return simhash(tokenize(text), bits=bits)


def buckets(
    hash_value: int,
    bands: int = DEFAULT_BANDS,
    band_size: int = DEFAULT_BAND_SIZE,
) -> List[int]:
    if bands <= 0 or band_size <= 0:
        raise ValueError("bands and band_size must be positive")
    mask = (1 << band_size) - 1
    bucket_ids: List[int] = []
    for band in range(bands):
        segment = (hash_value >> (band * band_size)) & mask
        bucket_ids.append((band << band_size) | segment)
    return bucket_ids


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimHashIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SimHash index not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))

    def close(self) -> None:
        self.conn.close()

    def search(
        self,
        text: str,
        top_k: int = 5,
        exclude_id: Optional[str] = None,
        bands: int = DEFAULT_BANDS,
        band_size: int = DEFAULT_BAND_SIZE,
    ) -> List[Tuple[str, float, int]]:
        tokens = tokenize(text)
        if not tokens:
            return []
        signature = simhash(tokens)
        bucket_ids = buckets(signature, bands=bands, band_size=band_size)
        placeholders = ",".join("?" for _ in bucket_ids)
        query = f"SELECT trial_id, hash FROM buckets WHERE bucket IN ({placeholders})"
        rows = self.conn.execute(query, bucket_ids).fetchall()
        best: dict = {}
        for trial_id, hash_value in rows:
            if exclude_id and trial_id == exclude_id:
                continue
            distance = hamming_distance(signature, int(hash_value))
            current = best.get(trial_id)
            if current is None or distance < current:
                best[trial_id] = distance
        ranked = sorted(best.items(), key=lambda x: x[1])
        results: List[Tuple[str, float, int]] = []
        for trial_id, distance in ranked[:top_k]:
            score = 1.0 / (1.0 + distance)
            results.append((trial_id, score, distance))
        return results
