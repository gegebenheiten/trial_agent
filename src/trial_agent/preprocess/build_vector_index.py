import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))  

from trial_agent.config import settings
from trial_agent.retrieval.index import trial_to_field_chunks
from trial_agent.retrieval.vector_store import resolve_vector_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS vector index for trial retrieval.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.processed_trials,
        help="JSONL corpus to index.",
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=settings.vector_index_path,
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--ids-out",
        type=Path,
        default=settings.vector_id_map_path,
        help="Output trial_id list (line-delimited).",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="full",
        help="Comma-separated focus fields to build (e.g. full,condition,drug,biomarker,study,design,endpoint).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.embedding_model_name,
        help="SentenceTransformers model name/path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embedding (e.g. cpu, cuda:0). Defaults to SentenceTransformer auto.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated devices for multi-GPU encoding (e.g. cuda:0,cuda:1).",
    )
    parser.add_argument(
        "--stable-gpu",
        action="store_true",
        help="Disable TF32/reduced-precision matmul to improve GPU numerical stability.",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force float32 model weights/compute (avoid BF16/FP16).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (tokens) for embedding.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Max character length per text before embedding (0 disables).",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=settings.embedding_normalize,
        help="Normalize embeddings (recommended for cosine similarity with IndexFlatIP).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable embedding normalization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=settings.embedding_trust_remote_code,
        help="Allow remote code when loading the embedding model.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable remote code when loading the embedding model.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "hnsw"],
        help="FAISS index type.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW parameter M (only for hnsw).",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction (only for hnsw).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after N records (debug).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing index/ids files (skips already indexed chunks).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write FAISS index every N chunks (0 disables).",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path, limit: int) -> Iterable[Tuple[int, dict]]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def _count_lines(path: Path) -> int:
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _validate_model_path(model_name: str) -> None:
    model_path = Path(model_name)
    if not model_path.exists():
        return
    if (model_path / "modules.json").exists() or (model_path / "config.json").exists():
        return
    raise FileNotFoundError(
        f"Local model path is missing modules.json/config.json: {model_path}"
    )


def _validate_embeddings(vectors, context: str, dim: int) -> None:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for embedding validation.") from exc
    if vectors is None or not hasattr(vectors, "shape"):
        raise RuntimeError(f"{context}: embeddings missing or invalid.")
    if vectors.ndim != 2 or vectors.shape[1] != dim:
        raise RuntimeError(
            f"{context}: embedding shape mismatch (got {vectors.shape}, expected dim={dim})."
        )
    if not np.isfinite(vectors).all():
        breakpoint()
        raise RuntimeError(f"{context}: embeddings contain NaN/Inf.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError(f"{context}: embeddings have invalid norms.")


def _safe_normalize(vectors):
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for embedding normalization.") from exc
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")
    _validate_model_path(args.model)
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss is required to build the vector index.") from exc
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required to build the vector index.") from exc

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)

    torch = None
    if (args.stable_gpu and (args.device or args.devices)) or args.force_fp32:
        try:
            import torch as torch_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError("torch is required for --stable-gpu/--force-fp32.") from exc
        torch = torch_module
    if args.stable_gpu and (args.device or args.devices):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        except (AttributeError, AssertionError):
            pass
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        except (AttributeError, AssertionError):
            pass
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")

    model_kwargs = None
    if args.force_fp32:
        model_kwargs = {"torch_dtype": torch.float32}
    model = SentenceTransformer(
        args.model,
        device=args.device if not args.devices else None,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
        )
    if args.force_fp32:
        model = model.to(dtype=torch.float32)
    if args.max_length and args.max_length > 0:
        model.max_seq_length = args.max_length
    devices: List[str] = [d.strip() for d in args.devices.split(",") if d.strip()]
    pool = None
    if devices:
        pool = model.start_multi_process_pool(target_devices=devices)

    def encode_texts(texts: List[str]):
        if pool:
            vectors = model.encode_multi_process(
                texts,
                pool,
                batch_size=args.batch_size,
                normalize_embeddings=False,
            )
            return _safe_normalize(vectors) if args.normalize else vectors
        vectors = model.encode(
            texts,
            normalize_embeddings=False,
            show_progress_bar=False,
            batch_size=args.batch_size,
        )
        return _safe_normalize(vectors) if args.normalize else vectors
    dim = model.get_sentence_embedding_dimension()
    _validate_embeddings(encode_texts(["sanity check"]), "sanity check", dim)

    fields = [f.strip().lower() for f in args.fields.split(",") if f.strip()]
    if not fields:
        fields = ["full"]

    for field in fields:
        index_out, ids_out = resolve_vector_paths(field, args.index_out, args.ids_out)
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive.")
        existing_index = index_out.exists()
        existing_ids = ids_out.exists()

        def make_index():
            if args.index_type == "hnsw":
                metric = faiss.METRIC_INNER_PRODUCT if args.normalize else faiss.METRIC_L2
                index_local = faiss.IndexHNSWFlat(dim, args.hnsw_m, metric)
                index_local.hnsw.efConstruction = args.ef_construction
                return index_local
            return faiss.IndexFlatIP(dim)

        skip_remaining = 0
        ids_out.parent.mkdir(parents=True, exist_ok=True)
        index_out.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if existing_index and existing_ids:
                index = faiss.read_index(str(index_out))
                already_indexed = _count_lines(ids_out)
                index_total = int(index.ntotal)
                if index_total != already_indexed:
                    print(
                        f"[{field}] Warning: index.ntotal ({index_total}) != ids ({already_indexed}). "
                        "Resuming using ids count."
                    )
                skip_remaining = already_indexed
                total = already_indexed
                ids_file = ids_out.open("a")
                print(f"[{field}] Resuming from {already_indexed} chunks.")
            elif not existing_index and not existing_ids:
                index = make_index()
                total = 0
                ids_file = ids_out.open("w")
            else:
                raise FileExistsError(
                    f"Cannot resume '{field}': index/ids mismatch. "
                    f"Index exists: {existing_index}, ids exists: {existing_ids}."
                )
        else:
            if existing_index or existing_ids:
                if not args.overwrite:
                    raise FileExistsError(
                        f"Vector index exists for '{field}'. Use --overwrite to rebuild."
                    )
                if existing_index:
                    index_out.unlink()
                if existing_ids:
                    ids_out.unlink()
            index = make_index()
            total = 0
            ids_file = ids_out.open("w")
        batch_texts = []
        batch_ids = []

        def checkpoint() -> None:
            tmp_path = index_out.with_suffix(index_out.suffix + ".tmp")
            faiss.write_index(index, str(tmp_path))
            tmp_path.replace(index_out)
            ids_file.flush()
            os.fsync(ids_file.fileno())

        for _, trial in _iter_jsonl(args.jsonl, args.limit):
            trial_id = str(trial.get("trial_id") or "").strip()
            if not trial_id:
                continue
            texts = trial_to_field_chunks(trial, field, max_chars=args.max_chars)
            if skip_remaining:
                if skip_remaining >= len(texts):
                    skip_remaining -= len(texts)
                    continue
                texts = texts[skip_remaining:]
                skip_remaining = 0
            for text in texts:
                if not text.strip():
                    continue
                batch_texts.append(text)
                batch_ids.append(trial_id)

                if len(batch_texts) >= args.batch_size:
                    vectors = encode_texts(batch_texts)
                    breakpoint()
                    _validate_embeddings(vectors, f"{field} batch", dim)
                    index.add(vectors)
                    for tid in batch_ids:
                        ids_file.write(tid + "\n")
                    total += len(batch_ids)
                    if args.checkpoint_every and total % args.checkpoint_every == 0:
                        checkpoint()
                        print(f"[{field}] Checkpointed {total} chunks to {index_out}")
                    batch_texts.clear()
                    batch_ids.clear()
                    if total and total % 50000 == 0:
                        print(f"[{field}] Indexed {total} chunks...")

        if batch_texts:
            vectors = encode_texts(batch_texts)
            _validate_embeddings(vectors, f"{field} tail batch", dim)
            index.add(vectors)
            for tid in batch_ids:
                ids_file.write(tid + "\n")
            total += len(batch_ids)
            if args.checkpoint_every and total % args.checkpoint_every == 0:
                checkpoint()
                print(f"[{field}] Checkpointed {total} chunks to {index_out}")

        ids_file.close()
        if skip_remaining:
            print(
                f"[{field}] Warning: resume skipped {skip_remaining} chunks beyond corpus length."
            )
        faiss.write_index(index, str(index_out))
        print(f"Wrote FAISS index for '{field}' ({total} chunks) to {index_out}")
        print(f"Wrote trial_id map to {ids_out}")
    if pool:
        model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    main()
