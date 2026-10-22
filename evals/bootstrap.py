"""Bootstrap confidence intervals for per-mode Precision@5 and bm25 - vector.

Resamples answerable queries with replacement (rejection queries have no
Precision@5). The same resampled query set is used for every mode in an
iteration, so the bm25 - vector difference is a paired bootstrap. Intervals
are percentile intervals; the seed is fixed so results are reproducible.

Usage:
    python -m evals.bootstrap evals/results/<run>.rescored.json [...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

N_RESAMPLES = 10_000
SEED = 20261022
CONFIDENCE = 0.95
METRIC = "precision@5"


def _nearest_rank(ordered: Sequence[float], p: float) -> float:
    """p-th percentile by nearest rank: the ceil(p * N)-th smallest value (1-based).

    p * N is rounded first so float noise (0.05 * 100 = 4.999...) cannot shift
    the rank by one.
    """
    rank = max(1, math.ceil(round(p * len(ordered), 9)))
    return ordered[rank - 1]


def percentile_interval(samples: Sequence[float], confidence: float) -> tuple[float, float]:
    """Central `confidence` interval of `samples` by nearest-rank percentiles."""
    ordered = sorted(samples)
    tail = (1 - confidence) / 2
    return _nearest_rank(ordered, tail), _nearest_rank(ordered, 1 - tail)


def paired_bootstrap(
    scores: dict[str, list[float]],
    diff: tuple[str, str],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, dict[str, float]]:
    """CIs of each mode's mean and of mean(diff[0]) - mean(diff[1]).

    `scores[mode][i]` must refer to the same query i for every mode.
    """
    n = len(next(iter(scores.values())))
    if any(len(values) != n for values in scores.values()):
        raise ValueError("every mode needs one score per query, in the same order")

    rng = random.Random(seed)
    means: dict[str, list[float]] = {mode: [] for mode in scores}
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.choices(range(n), k=n)
        for mode, values in scores.items():
            means[mode].append(sum(values[i] for i in idx) / n)
        diffs.append(means[diff[0]][-1] - means[diff[1]][-1])

    out: dict[str, dict[str, float]] = {}
    for mode, values in scores.items():
        lo, hi = percentile_interval(means[mode], confidence)
        out[mode] = {"mean": sum(values) / n, "ci_low": lo, "ci_high": hi}
    lo, hi = percentile_interval(diffs, confidence)
    observed = out[diff[0]]["mean"] - out[diff[1]]["mean"]
    out[f"{diff[0]} - {diff[1]}"] = {"mean": observed, "ci_low": lo, "ci_high": hi}
    return out


def answerable_scores(run: dict[str, Any], metric: str = METRIC) -> dict[str, list[float]]:
    """Per-mode metric values over answerable queries, aligned by query_id."""
    by_mode = {
        mode: {q["query_id"]: q[metric] for q in r["per_query"] if q[metric] is not None}
        for mode, r in run["results"].items()
    }
    query_ids = sorted(next(iter(by_mode.values())))
    return {mode: [values[qid] for qid in query_ids] for mode, values in by_mode.items()}


def markdown(label: str, n_queries: int, result: dict[str, dict[str, float]]) -> str:
    lines = [
        f"| {label} (n={n_queries}) | mean | 95% CI |",
        "|---|---|---|",
    ]
    for name, r in result.items():
        lines.append(f"| {name} | {r['mean']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for Precision@5")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    print(f"{METRIC}, {N_RESAMPLES} resamples, seed {SEED}, percentile intervals\n")
    for path in args.runs:
        scores = answerable_scores(json.loads(path.read_text(encoding="utf-8")))
        result = paired_bootstrap(scores, diff=("bm25", "vector"))
        n_queries = len(next(iter(scores.values())))
        print(markdown(path.name.split("_")[0], n_queries, result))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
