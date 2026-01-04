from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from trial_agent.config import settings


def resolve_vector_paths(
    focus: str,
    base_index: Optional[Path] = None,
    base_ids: Optional[Path] = None,
) -> Tuple[Path, Path]:
    focus_key = (focus or "full").strip().lower()
    index_path = Path(base_index or settings.vector_index_path)
    ids_path = Path(base_ids or settings.vector_id_map_path)
    if focus_key in {"full", "all", "default"}:
        return index_path, ids_path

    index_name = f"{index_path.stem}.{focus_key}{index_path.suffix}"
    index_path = index_path.with_name(index_name)

    ids_name = ids_path.name
    suffix = ".vector_ids.txt"
    if ids_name.endswith(suffix):
        base = ids_name[: -len(suffix)]
    else:
        base = ids_path.stem
    ids_path = ids_path.with_name(f"{base}.{focus_key}{suffix}")
    return index_path, ids_path


def vector_index_available(
    base_index: Optional[Path] = None,
    base_ids: Optional[Path] = None,
) -> bool:
    index_path = Path(base_index or settings.vector_index_path)
    ids_path = Path(base_ids or settings.vector_id_map_path)
    if index_path.exists() and ids_path.exists():
        return True
    suffix = ".vector_ids.txt"
    ids_name = ids_path.name
    if ids_name.endswith(suffix):
        base = ids_name[: -len(suffix)]
    else:
        base = ids_path.stem
    ids_candidates = list(ids_path.parent.glob(f"{base}.*{suffix}"))
    index_candidates = list(index_path.parent.glob(f"{index_path.stem}.*{index_path.suffix}"))
    return bool(ids_candidates and index_candidates)


class VectorStore:
    def __init__(
        self,
        index_path: Path,
        ids_path: Path,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        trust_remote_code: Optional[bool] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.model_name = model_name or settings.embedding_model_name
        self.device = device
        self.normalize = settings.embedding_normalize if normalize is None else normalize
        self.trust_remote_code = (
            settings.embedding_trust_remote_code
            if trust_remote_code is None
            else trust_remote_code
        )
        self._index = None
        self._ids: Optional[List[str]] = None
        self._model = None
        self._metric_type: Optional[int] = None

    def _load(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"Vector id map not found: {self.ids_path}")
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss is required for vector retrieval.") from exc
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for vector retrieval.") from exc
        self._index = faiss.read_index(str(self.index_path))
        self._metric_type = getattr(self._index, "metric_type", None)
        self._ids = [line.strip() for line in self.ids_path.read_text().splitlines() if line.strip()]
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )

    def _embed(self, texts: Iterable[str]):
        if self._model is None:
            self._load()
        assert self._model is not None
        return self._model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def search(
        self,
        text: str,
        top_k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if not text.strip():
            return []
        self._load()
        assert self._index is not None
        assert self._ids is not None
        if self._index.ntotal <= 0:
            return []
        vectors = self._embed([text])
        try:
            import numpy as np  # type: ignore

            if not np.isfinite(vectors).all():
                raise ValueError("Query embedding contains NaN/Inf values.")
        except ImportError:
            pass
        search_k = min(max(top_k * 10, top_k), len(self._ids), self._index.ntotal)
        if search_k <= 0:
            return []
        scores, indices = self._index.search(vectors, search_k)
        best_scores: dict = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            trial_id = self._ids[idx]
            if exclude_id and trial_id == exclude_id:
                continue
            score_value = float(score)
            if self._metric_type is not None:
                try:
                    import faiss  # type: ignore

                    if self._metric_type == faiss.METRIC_L2:
                        score_value = -score_value
                except ImportError:
                    pass
            if trial_id not in best_scores or score_value > best_scores[trial_id]:
                best_scores[trial_id] = score_value
            if len(best_scores) >= top_k and idx > search_k // 2:
                continue
        results = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
        return results[:top_k]
