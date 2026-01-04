import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

def _resolve_embedding_model_name() -> str:
    explicit = os.getenv("EMBEDDING_MODEL_NAME")
    if explicit:
        return explicit
    for env_var in ("SENTENCE_TRANSFORMERS_HOME", "HF_HOME"):
        root = os.getenv(env_var)
        if not root:
            continue
        candidate = Path(root) / "bge-m3"
        if candidate.exists():
            return str(candidate)
    return "BAAI/bge-m3"


@dataclass
class Settings:
    """Centralized configuration for paths and lightweight defaults."""

    data_dir: Path = DATA_DIR
    processed_trials: Path = PROCESSED_DIR / "trialpanorama_trials.jsonl" # PROCESSED_DIR / "trials_ctgov_phase2_oncology.jsonl"
    labels_path: Path = PROCESSED_DIR / "labels.csv"
    bm25_index_dir: Path = INDEX_DIR / "bm25"
    vector_index_dir: Path = INDEX_DIR / "vectors"
    drugbank_xml: Path = DATA_DIR / "drugbank" / "full_database.xml"
    drugbank_minimal_index: Path = PROCESSED_DIR / "drugbank_minimal.jsonl"
    trialpanorama_raw_dir: Path = DATA_DIR / "trialpanorama" / "raw"
    trialpanorama_processed: Path = PROCESSED_DIR / "trialpanorama_trials.jsonl"
    trialpanorama_relations: Path = PROCESSED_DIR / "trialpanorama_relations.json"
    simhash_index_path: Path = PROCESSED_DIR / "trialpanorama_trials.simhash.sqlite"
    keyword_index_path: Path = INDEX_DIR / "keyword_trials.sqlite"
    vector_index_path: Path = INDEX_DIR / "vectors" / "trialpanorama_trials.faiss"
    vector_id_map_path: Path = INDEX_DIR / "vectors" / "trialpanorama_trials.vector_ids.txt"
    embedding_model_name: str = _resolve_embedding_model_name()
    embedding_normalize: bool = True
    embedding_trust_remote_code: bool = True
    trial_id_allowlist_csv: Optional[Path] = None
    default_top_k: int = 5


settings = Settings()
