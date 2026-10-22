"""Diagnostics over a saved result file: distractor intrusions and RRF traces.

Intrusion: a chunk from the *rival* draft of a query's relevant draft (see
RIVALS) appearing in the top 5. Only answerable queries whose relevant draft
has a rival are counted.

RRF trace: rebuilds the fused scores of one query from its stored vector and
bm25 rankings with the same formula as triage_agent.rag.hybrid.rrf_merge,
score = sum over lists of 1 / (rrf_k + rank). Stored lists hold only the top
10 of the fetch_k candidates fused in production, so a chunk ranked in both
stored lists has its exact score, while a chunk missing from one list shows a
lower bound. "Exact" assumes the hybrid run's internal vector ranking equals
the separate vector run's (BM25 is deterministic; the query embedding is a
separate API call); compare the reconstructed order with the hybrid rank
column to check.

Usage:
    python -m evals.analysis evals/results/<run>.json --trace q23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Pairs of an original draft and its near-topic distractor (README, "Corpus rounds").
RIVALS = {
    "peft": "llm_quant",
    "llm_quant": "peft",
    "diffusion": "gan",
    "gan": "diffusion",
    "rl_trading": "marl",
    "marl": "rl_trading",
}
BOUNDARY_IDS = {f"q{i}" for i in range(21, 33)}
TOP = 5


def rival_intrusions(per_query: list[dict[str, Any]], query_ids: set[str]) -> dict[str, int]:
    stats = {"queries": 0, "slots": 0, "rival_chunks": 0, "queries_hit": 0, "rival_at_rank1": 0}
    for q in per_query:
        relevant = q["relevant_draft_ids"]
        if q["query_id"] not in query_ids or len(relevant) != 1 or relevant[0] not in RIVALS:
            continue
        rival = RIVALS[relevant[0]]
        top_ids = [d["local_id"] for d in q["retrieved"][:TOP]]
        n_rival = top_ids.count(rival)
        stats["queries"] += 1
        stats["slots"] += TOP
        stats["rival_chunks"] += n_rival
        stats["queries_hit"] += int(n_rival > 0)
        stats["rival_at_rank1"] += int(bool(top_ids) and top_ids[0] == rival)
    return stats


def intrusion_table(run: dict[str, Any], groups: dict[str, set[str]]) -> str:
    lines = [
        "| mode | query group | queries | rival chunks in top 5 / slots "
        "| queries with >=1 rival in top 5 | rival at rank 1 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, result in run["results"].items():
        for label, ids in groups.items():
            s = rival_intrusions(result["per_query"], ids)
            if not s["queries"]:
                continue
            lines.append(
                f"| {mode} | {label} | {s['queries']} | {s['rival_chunks']} / {s['slots']} "
                f"| {s['queries_hit']} | {s['rival_at_rank1']} |"
            )
    return "\n".join(lines)


def _chunk_key(d: dict[str, Any]) -> tuple[str, str, str]:
    return (d["local_id"], d["section"], d["preview"])


def rrf_trace(run: dict[str, Any], query_id: str, top: int = TOP) -> str:
    rrf_k = run["config"]["retrieval"]["rrf_k"]
    ranks: dict[str, dict[tuple[str, str, str], int]] = {}
    for mode in ("vector", "bm25", "hybrid"):
        q = next(q for q in run["results"][mode]["per_query"] if q["query_id"] == query_id)
        ranks[mode] = {_chunk_key(d): d["rank"] for d in q["retrieved"]}

    rows = []
    for key in ranks["vector"].keys() | ranks["bm25"].keys():
        v, b = ranks["vector"].get(key), ranks["bm25"].get(key)
        score = sum(1 / (rrf_k + r) for r in (v, b) if r is not None)
        rows.append((score, key, v, b, ranks["hybrid"].get(key)))
    rows.sort(key=lambda row: (-row[0], row[4] or 99))

    lines = [
        f"RRF trace for {query_id} (rrf_k = {rrf_k}); '-' = outside that list's stored top 10.",
        "",
        "| chunk (draft / section) | vector rank | bm25 rank | RRF score | exact? "
        "| hybrid rank |",
        "|---|---|---|---|---|---|",
    ]
    for score, (draft, section, _), v, b, h in rows[:top]:
        exact = "yes" if v is not None and b is not None else "lower bound"
        lines.append(
            f"| {draft} / {section} | {v or '-'} | {b or '-'} | {score:.6f} | {exact} "
            f"| {h or '-'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intrusion stats and RRF traces for a run")
    parser.add_argument("run", type=Path)
    parser.add_argument("--trace", action="append", default=[], metavar="QUERY_ID")
    args = parser.parse_args(argv)

    run = json.loads(args.run.read_text(encoding="utf-8"))
    all_ids = {q["query_id"] for q in next(iter(run["results"].values()))["per_query"]}
    groups = {
        "boundary (q21-q32)": BOUNDARY_IDS & all_ids,
        "original with a rival (q01-q20)": all_ids - BOUNDARY_IDS,
    }
    print(intrusion_table(run, groups))
    for query_id in args.trace:
        print()
        print(rrf_trace(run, query_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
