"""Side-by-side Markdown comparison of two run_eval result files.

Rows for each mode:
  old            the old run, all its queries
  new            the new run, all its queries
  new (old set)  the new run scored only on the old run's query ids,
                 so corpus changes can be separated from query-set changes

Usage:
    python -m evals.compare evals/results/<old>.json evals/results/<new>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.run_eval import RESULTS_DIR, _fmt, summarize


def _corpus(config: dict[str, Any]) -> str:
    n_drafts = len(config["inputs"]["corpus_sha256"])
    return f"{n_drafts} drafts / {config['chunking']['n_chunks']} chunks"


def _row(mode: str, label: str, corpus: str, summary: dict[str, Any]) -> str:
    n_scored = f"{summary['n_answerable']}+{summary['n_rejection']}"
    return (
        f"| {mode} | {label} | {corpus} | {n_scored} | {_fmt(summary['recall@5'])} "
        f"| {_fmt(summary['mrr@10'])} | {_fmt(summary['rejection_rate'])} "
        f"| {_fmt(summary['latency_ms_mean'], 1)} |"
    )


def compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    old_corpus, new_corpus = _corpus(old["config"]), _corpus(new["config"])
    lines = [
        "| mode | round | corpus | queries (answerable+rejection) | Recall@5 | MRR@10 "
        "| rejection rate | mean latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in new["results"]:
        if mode not in old["results"]:
            continue
        old_queries = old["results"][mode]["per_query"]
        old_ids = {q["query_id"] for q in old_queries}
        new_queries = new["results"][mode]["per_query"]
        new_on_old_set = [q for q in new_queries if q["query_id"] in old_ids]

        lines.append(_row(mode, "old", old_corpus, summarize(old_queries)))
        lines.append(_row(mode, "new", new_corpus, summarize(new_queries)))
        lines.append(_row(mode, "new (old set)", new_corpus, summarize(new_on_old_set)))
    lines.append("")
    lines.append(
        f"old = {old['config']['timestamp_utc']} @ {old['config']['git_commit'][:7]}; "
        f"new = {new['config']['timestamp_utc']} @ {new['config']['git_commit'][:7]}."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two run_eval result files")
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)

    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    table = compare(old, new)

    out = RESULTS_DIR / f"compare_{args.old.stem.split('_')[0]}_vs_{args.new.stem.split('_')[0]}.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
