"""Recompute all metrics offline from saved run_eval result files.

No retrieval or embedding calls are made: rankings come from each query's
stored `retrieved` list (the top RETRIEVE_K chunks). nDCG also needs the
number of indexed chunks per draft. Runs made before that was recorded get it
recomputed with the same chunking pipeline as build_index(), after checking
the corpus files against the snapshot's sha256 and the total against its
n_chunks. Stored Recall@5 / RR@10 must be reproduced exactly, or the script
stops.

Usage:
    python -m evals.rescore evals/results/<run>.json [<run>.json ...]
Writes <run>.rescored.json next to each input; inputs are left unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import score_query
from evals.run_eval import CORPUS_DIR, MANIFEST_PATH, REPO_ROOT, _git, _sha256, summarize
from triage_agent.local_kb import LocalManifest
from triage_agent.rag.build_index import _expand_latex_docs, load_documents_from_manifest


def recompute_chunks_per_draft(config: dict[str, Any]) -> dict[str, int]:
    """Re-chunk the run's corpus exactly as build_index() did; no embeddings."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    corpus = config["inputs"]["corpus_sha256"]
    for name, expected in corpus.items():
        if _sha256(CORPUS_DIR / name) != expected:
            raise RuntimeError(f"{name} changed since the run; cannot recompute chunk counts")

    manifest = LocalManifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    manifest.sources = [s for s in manifest.sources if Path(s.path).name in corpus]
    if len(manifest.sources) != len(corpus):
        raise RuntimeError("eval manifest no longer lists every corpus file of the run")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    docs = _expand_latex_docs(load_documents_from_manifest(manifest, REPO_ROOT))
    counts = Counter(str(c.metadata["local_id"]) for c in splitter.split_documents(docs))

    if sum(counts.values()) != config["chunking"]["n_chunks"]:
        raise RuntimeError(
            f"re-chunking gives {sum(counts.values())} chunks, "
            f"run recorded {config['chunking']['n_chunks']}"
        )
    return dict(sorted(counts.items()))


def rescore_run(run: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(run)
    recorded = run["config"]["chunking"].get("chunks_per_draft")
    chunks_per_draft = recorded or recompute_chunks_per_draft(run["config"])

    for mode, mode_result in out["results"].items():
        for q in mode_result["per_query"]:
            ranked_ids = [d["local_id"] for d in q["retrieved"]]
            scores = score_query(ranked_ids, q["relevant_draft_ids"], chunks_per_draft)
            for key in ("recall@5", "rr@10", "correctly_rejected"):
                if scores[key] != q[key]:
                    raise RuntimeError(
                        f"[{mode}] {q['query_id']}: recomputed {key}={scores[key]} "
                        f"differs from stored {q[key]}"
                    )
            q.update(scores)
        mode_result["summary"] = summarize(mode_result["per_query"])

    out["rescore"] = {
        "rescored_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "chunks_per_draft": chunks_per_draft,
        "chunks_per_draft_source": "config" if recorded else "recomputed offline (verified)",
        "note": "metrics recomputed from stored top-10 rankings; no retrieval re-run",
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rescore saved run_eval results offline")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    for path in args.runs:
        rescored = rescore_run(json.loads(path.read_text(encoding="utf-8")))
        out = path.with_name(f"{path.stem}.rescored.json")
        out.write_text(json.dumps(rescored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.name} -> {out.name} ({rescored['rescore']['chunks_per_draft_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
