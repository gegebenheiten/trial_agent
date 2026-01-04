import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append("src")

from trial_agent.retrieval.index import trial_to_field_text, trial_to_full_chunks


@dataclass
class RunningStats:
    sample_size: int
    count: int = 0
    total: int = 0
    min_val: Optional[int] = None
    max_val: Optional[int] = None
    sample: List[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value
        if self.sample_size <= 0:
            return
        if len(self.sample) < self.sample_size:
            self.sample.append(value)
            return
        idx = random.randint(1, self.count)
        if idx <= self.sample_size:
            self.sample[idx - 1] = value

    def mean(self) -> float:
        if not self.count:
            return 0.0
        return self.total / self.count

    def quantiles(self, qs: List[float]) -> Dict[str, float]:
        if not self.sample:
            return {f"p{int(q * 100)}": 0.0 for q in qs}
        sample_sorted = sorted(self.sample)
        out: Dict[str, float] = {}
        n = len(sample_sorted)
        for q in qs:
            idx = int(round(q * (n - 1)))
            out[f"p{int(q * 100)}"] = float(sample_sorted[idx])
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute length stats for trial fields by focus.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/processed/trialpanorama_trials.jsonl"),
        help="Path to trialpanorama JSONL.",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="full,condition,drug,biomarker,endpoint,study,outcome",
        help="Comma-separated focus keys.",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, stop after N trials.")
    parser.add_argument("--sample-size", type=int, default=200000, help="Reservoir sample size.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")

    focus_list = [f.strip().lower() for f in args.focus.split(",") if f.strip()]
    stats: Dict[str, RunningStats] = {
        focus: RunningStats(sample_size=args.sample_size) for focus in focus_list
    }
    empty_counts = {focus: 0 for focus in focus_list}

    full_total_stats = RunningStats(sample_size=args.sample_size) if "full" in stats else None
    total_trials = 0
    total_full_chunks = 0

    with args.jsonl.open() as f:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                trial = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_trials += 1
            for focus in focus_list:
                if focus == "full":
                    chunks = trial_to_full_chunks(trial, max_chars=0)
                    total_full_chunks += len(chunks)
                    total_len = 0
                    for chunk in chunks:
                        length = len(chunk)
                        total_len += length
                        stats[focus].add(length)
                    if full_total_stats is not None and total_len:
                        full_total_stats.add(total_len)
                else:
                    text = trial_to_field_text(trial, focus)
                    length = len(text)
                    if length <= 0:
                        empty_counts[focus] += 1
                        continue
                    stats[focus].add(length)

    output = {
        "total_trials": total_trials,
        "focus_stats": {},
    }
    for focus in focus_list:
        s = stats[focus]
        q = s.quantiles([0.5, 0.9, 0.95, 0.99])
        output["focus_stats"][focus] = {
            "count": s.count,
            "empty_trials": empty_counts.get(focus, 0),
            "min": s.min_val or 0,
            "mean": round(s.mean(), 2),
            "max": s.max_val or 0,
            **{k: int(v) for k, v in q.items()},
        }
        if focus == "full":
            output["focus_stats"][focus]["avg_chunks_per_trial"] = round(
                total_full_chunks / max(1, total_trials), 3
            )
            if full_total_stats is not None:
                full_q = full_total_stats.quantiles([0.5, 0.9, 0.95, 0.99])
                output["focus_stats"][focus]["total_len_per_trial"] = {
                    "count": full_total_stats.count,
                    "min": full_total_stats.min_val or 0,
                    "mean": round(full_total_stats.mean(), 2),
                    "max": full_total_stats.max_val or 0,
                    **{k: int(v) for k, v in full_q.items()},
                }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
