"""Run the retrieval eval over evals/golden.jsonl and save results.

Modes:
  vector  existing LocalRAGRetriever path with RAG_HYBRID=0 (MMR vector search)
  bm25    Bm25Index built directly over the same indexed chunks
  hybrid  existing default LocalRAGRetriever path (MMR + BM25 fused by RRF)

Usage:
    python -m evals.run_eval                      # all three modes
    python -m evals.run_eval --mode vector bm25   # a subset
    python -m evals.run_eval --reranker none

Requires OPENAI_API_KEY (environment or repo .env) for embeddings; there is
no fallback. Raw results + a config snapshot go to
evals/results/<timestamp>_<modes>_<reranker>.json, and the Markdown table is
printed and saved next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.metrics import correctly_rejected, mean_defined, recall_at_k, reciprocal_rank_at_k
from evals.rerankers import RERANKER_CHOICES, Reranker, get_reranker
from triage_agent.rag.build_index import _DEFAULT_CHUNK, _DEFAULT_OVERLAP, build_index
from triage_agent.rag.hybrid import Bm25Index, rrf_merge
from triage_agent.rag.retriever import (
    _DEFAULT_FETCH_K,
    _DEFAULT_MMR_LAMBDA,
    LocalRAGRetriever,
    _all_chunks_from_store,
)

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
MANIFEST_PATH = EVALS_DIR / "eval_manifest.json"
CORPUS_DIR = EVALS_DIR / "corpus"
INDEX_DIR = EVALS_DIR / ".index"
RESULTS_DIR = EVALS_DIR / "results"

MODES = ("vector", "bm25", "hybrid")
RETRIEVE_K = 10  # chunks fetched per query; enough for MRR@10
RECALL_K = 5
MRR_K = 10

# Excluded from scoring and timing: loads Chroma and fills the BM25 cache.
WARMUP_QUERY = "Title: warm-up\n\nAbstract:\nwarm-up query, not scored"


class _WarningCollector(logging.Handler):
    """Collects WARNING+ records from triage_agent.rag.

    LocalRAGRetriever.retrieve catches every exception, logs a warning and
    returns []. An empty result would be scored as a correct rejection, so
    the runner treats any such warning as a hard failure instead.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")

    def raise_if_any(self, context: str) -> None:
        if self.messages:
            raise RuntimeError(f"{context}: {'; '.join(self.messages)}")


def require_api_key() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        sys.exit("OPENAI_API_KEY is not set (env or .env); embeddings are required. Aborting.")


def preflight_embedding(model: str) -> None:
    """One real embedding call, so a bad key fails here and not mid-eval."""
    from langchain_openai import OpenAIEmbeddings

    OpenAIEmbeddings(model=model).embed_query("preflight")


def build_eval_index() -> int:
    """Rebuild the eval-only Chroma index and return its chunk count."""
    # HARD REQUIREMENT: always pass persist_directory explicitly, pointing
    # inside evals/. build_index() deletes its persist directory before
    # writing (shutil.rmtree); with the default it would wipe the real index
    # at local_kb/chroma_index (or RAG_CHROMA_PATH).
    if not INDEX_DIR.resolve().is_relative_to(EVALS_DIR):
        raise RuntimeError(f"eval index dir must live under {EVALS_DIR}, got {INDEX_DIR}")
    return build_index(
        manifest_path=MANIFEST_PATH,
        persist_directory=INDEX_DIR,
        project_root_path=REPO_ROOT,
    )


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        missing = {"query_id", "query", "relevant_draft_ids"} - row.keys()
        if missing:
            raise ValueError(f"golden row {row.get('query_id')} missing {sorted(missing)}")
    return rows


def make_searcher(mode: str, retriever: LocalRAGRetriever) -> Callable[[str], list[Any]]:
    if mode == "bm25":
        chunks = _all_chunks_from_store(retriever.vectorstore)
        if not chunks:
            raise RuntimeError("no chunks in the eval index; cannot build BM25")
        bm25 = Bm25Index(chunks)
        return lambda query: bm25.search(query, k=RETRIEVE_K)
    # vector / hybrid share the production retriever; RAG_HYBRID selects the path
    # and is read on every retrieve() call.
    os.environ["RAG_HYBRID"] = "0" if mode == "vector" else "1"
    return retriever.retrieve


def run_mode(
    mode: str,
    golden: list[dict[str, Any]],
    retriever: LocalRAGRetriever,
    reranker: Reranker,
    warnings: _WarningCollector,
) -> list[dict[str, Any]]:
    search = make_searcher(mode, retriever)
    search(WARMUP_QUERY)
    warnings.raise_if_any(f"[{mode}] warm-up failed")

    per_query: list[dict[str, Any]] = []
    for row in golden:
        query, relevant = row["query"], row["relevant_draft_ids"]

        start = time.perf_counter()
        docs = reranker.rerank(query, search(query))
        latency_ms = (time.perf_counter() - start) * 1000
        warnings.raise_if_any(f"[{mode}] retrieval failed for {row['query_id']}")

        ranked_ids = [str(doc.metadata.get("local_id", "")) for doc in docs]
        per_query.append(
            {
                "query_id": row["query_id"],
                "relevant_draft_ids": relevant,
                "recall@5": recall_at_k(ranked_ids, relevant, RECALL_K),
                "rr@10": reciprocal_rank_at_k(ranked_ids, relevant, MRR_K),
                "correctly_rejected": correctly_rejected(ranked_ids, relevant),
                "latency_ms": round(latency_ms, 2),
                "retrieved": [
                    {
                        "rank": rank,
                        "local_id": draft_id,
                        "section": doc.metadata.get("section", ""),
                        "preview": doc.page_content[:80],
                    }
                    for rank, (draft_id, doc) in enumerate(zip(ranked_ids, docs), start=1)
                ],
            }
        )
    return per_query


def summarize(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [q["latency_ms"] for q in per_query]
    return {
        "recall@5": mean_defined(q["recall@5"] for q in per_query),
        "mrr@10": mean_defined(q["rr@10"] for q in per_query),
        "rejection_rate": mean_defined(q["correctly_rejected"] for q in per_query),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_max": max(latencies),
        "n_answerable": sum(1 for q in per_query if q["relevant_draft_ids"]),
        "n_rejection": sum(1 for q in per_query if not q["relevant_draft_ids"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def config_snapshot(
    modes: list[str],
    reranker_name: str,
    n_chunks: int,
    embedding_model: str,
    golden: list[dict[str, Any]],
) -> dict[str, Any]:
    # Mirrors how build_index() and LocalRAGRetriever.retrieve() resolve these values.
    fetch_k = int(os.getenv("RAG_FETCH_K", str(max(RETRIEVE_K * 2, _DEFAULT_FETCH_K))))
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "modes": modes,
        "reranker": reranker_name,
        "chunking": {
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", str(_DEFAULT_CHUNK))),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", str(_DEFAULT_OVERLAP))),
            "latex_section_split": True,
            "n_chunks": n_chunks,
        },
        "retrieval": {
            "retrieve_k": RETRIEVE_K,
            "recall_k": RECALL_K,
            "mrr_k": MRR_K,
            "fetch_k": fetch_k,
            "mmr_lambda": float(os.getenv("RAG_MMR_LAMBDA", str(_DEFAULT_MMR_LAMBDA))),
            "rrf_k": signature(rrf_merge).parameters["k"].default,
            "embedding_model": embedding_model,
            "latency_note": "per query, warm-up excluded; includes OpenAI query-embedding "
            "round trip for vector/hybrid",
        },
        "inputs": {
            "golden_sha256": _sha256(GOLDEN_PATH),
            "n_queries": len(golden),
            "manifest_sha256": _sha256(MANIFEST_PATH),
            "corpus_sha256": {p.name: _sha256(p) for p in sorted(CORPUS_DIR.glob("*.tex"))},
        },
        "python": platform.python_version(),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown_table(summaries: dict[str, dict[str, Any]], reranker_name: str) -> str:
    lines = [
        "| mode | reranker | Recall@5 | MRR@10 | rejection rate | mean latency (ms) "
        "| p50 latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, s in summaries.items():
        lines.append(
            f"| {mode} | {reranker_name} | {_fmt(s['recall@5'])} | {_fmt(s['mrr@10'])} "
            f"| {_fmt(s['rejection_rate'])} | {_fmt(s['latency_ms_mean'], 1)} "
            f"| {_fmt(s['latency_ms_p50'], 1)} |"
        )
    any_summary = next(iter(summaries.values()))
    lines.append("")
    lines.append(
        f"Recall@5 / MRR@10 are means over {any_summary['n_answerable']} answerable queries; "
        f"rejection rate is over {any_summary['n_rejection']} rejection queries "
        "(correct iff nothing is returned)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval over evals/golden.jsonl")
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--reranker", choices=RERANKER_CHOICES, default="none")
    args = parser.parse_args(argv)

    require_api_key()
    reranker = get_reranker(args.reranker)
    # RAG_TOP_K would override the explicit k passed to LocalRAGRetriever below.
    os.environ.pop("RAG_TOP_K", None)
    embedding_model = os.getenv("RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    warnings = _WarningCollector()
    logging.getLogger("triage_agent.rag").addHandler(warnings)

    preflight_embedding(embedding_model)
    n_chunks = build_eval_index()
    warnings.raise_if_any("index build reported problems")
    print(f"Built eval index: {n_chunks} chunks in {INDEX_DIR}")

    golden = load_golden(GOLDEN_PATH)
    retriever = LocalRAGRetriever(k=RETRIEVE_K, persist_directory=INDEX_DIR)

    results: dict[str, Any] = {}
    for mode in args.mode:
        per_query = run_mode(mode, golden, retriever, reranker, warnings)
        results[mode] = {"summary": summarize(per_query), "per_query": per_query}
        print(f"[{mode}] done")

    summaries = {mode: r["summary"] for mode, r in results.items()}
    table = markdown_table(summaries, args.reranker)
    snapshot = config_snapshot(args.mode, args.reranker, n_chunks, embedding_model, golden)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{'-'.join(args.mode)}_{args.reranker}"
    json_path = RESULTS_DIR / f"{stem}.json"
    json_path.write_text(
        json.dumps({"config": snapshot, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stem}.md").write_text(table + "\n", encoding="utf-8")

    print()
    print(table)
    print(f"\nSaved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
