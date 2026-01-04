import argparse
import csv
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError, HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/ctgov_provided_documents.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/ctgov_documents"
DEFAULT_OUT_CSV = PROJECT_ROOT / "data/processed/ctgov_document_paths.csv"
DEFAULT_LOG_CSV = PROJECT_ROOT / "data/logs/ctgov_pdf_downloads.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download clinicaltrials.gov provided documents and record local paths."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with document URLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to store downloaded PDFs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output CSV with local paths per trial.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=DEFAULT_LOG_CSV,
        help="Download log CSV (one row per document).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max concurrent downloads.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout (seconds) per request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count per URL on failure.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between retries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit total downloads (0 means no limit).",
    )
    return parser.parse_args()


def _safe_filename(url: str, fallback: str) -> str:
    name = unquote(Path(urlparse(url).path).name)
    if not name:
        name = fallback
    base, ext = os.path.splitext(name)
    if not ext:
        name = f"{name}.pdf"
    return name


def _dedupe_filename(name: str, used: set, url: str) -> str:
    if name not in used:
        return name
    base, ext = os.path.splitext(name)
    short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}__{short_hash}{ext or '.pdf'}"
    suffix = 1
    while candidate in used:
        candidate = f"{base}__{short_hash}_{suffix}{ext or '.pdf'}"
        suffix += 1
    return candidate


def _iter_rows(path: Path) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")
        return reader.fieldnames, list(reader)


def _download_one(
    url: str,
    local_path: Path,
    timeout: int,
    retries: int,
    sleep_s: float,
) -> Tuple[str, int, str]:
    if local_path.exists():
        size = local_path.stat().st_size
        if size > 0:
            return "exists", size, ""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    last_error = ""
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                with tmp_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            os.replace(tmp_path, local_path)
            size = local_path.stat().st_size
            return "ok", size, ""
        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(sleep_s)

    return "fail", 0, last_error


def main() -> None:
    args = _parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = _iter_rows(args.input_csv)
    output_fields = list(fieldnames)
    if "document_local_paths" not in output_fields:
        output_fields.append("document_local_paths")

    tasks = []
    used_names: Dict[str, set] = {}

    with args.output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            nct_id = (row.get("nct_id") or "").strip()
            urls = [u for u in (row.get("document_urls") or "").split("|") if u]
            types = [t for t in (row.get("document_types") or "").split("|") if t]

            local_paths = []
            used = used_names.setdefault(nct_id, set())

            for idx, url in enumerate(urls):
                doc_type = types[idx] if idx < len(types) else ""
                fallback = f"document_{idx + 1}.pdf"
                name = _safe_filename(url, fallback)
                name = _dedupe_filename(name, used, url)
                used.add(name)

                local_path = args.output_dir / nct_id / name
                local_paths.append(str(local_path))
                tasks.append(
                    {
                        "nct_id": nct_id,
                        "url": url,
                        "doc_type": doc_type,
                        "local_path": local_path,
                    }
                )

            row["document_local_paths"] = "|".join(local_paths)
            writer.writerow(row)

    if args.limit > 0:
        tasks = tasks[: args.limit]

    log_exists = args.log_csv.exists()
    with args.log_csv.open("a", newline="", encoding="utf-8") as f_log:
        log_writer = csv.DictWriter(
            f_log,
            fieldnames=[
                "nct_id",
                "document_url",
                "document_type",
                "local_path",
                "status",
                "bytes",
                "error",
            ],
        )
        if not log_exists:
            log_writer.writeheader()

        stats = {"ok": 0, "exists": 0, "fail": 0}
        completed = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_map = {
                ex.submit(
                    _download_one,
                    task["url"],
                    task["local_path"],
                    args.timeout,
                    args.retries,
                    args.sleep,
                ): task
                for task in tasks
            }

            for fut in as_completed(future_map):
                task = future_map[fut]
                status, size, error = fut.result()
                stats[status] = stats.get(status, 0) + 1
                completed += 1
                log_writer.writerow(
                    {
                        "nct_id": task["nct_id"],
                        "document_url": task["url"],
                        "document_type": task["doc_type"],
                        "local_path": str(task["local_path"]),
                        "status": status,
                        "bytes": size,
                        "error": error,
                    }
                )

                if completed % 500 == 0:
                    print(
                        f"Downloaded {completed}/{len(tasks)} "
                        f"(ok={stats['ok']}, exists={stats['exists']}, fail={stats['fail']})"
                    )

    print(
        "Done. "
        f"Total={len(tasks)} ok={stats['ok']} "
        f"exists={stats['exists']} fail={stats['fail']}"
    )
    print(f"Output paths CSV: {args.output_csv}")
    print(f"Download log: {args.log_csv}")


if __name__ == "__main__":
    main()
