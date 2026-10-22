"""Tests for evals/compare.py using small hand-built run payloads."""

from evals.compare import compare


def _query(query_id: str, relevant: list[str], rr: float | None) -> dict:
    return {
        "query_id": query_id,
        "relevant_draft_ids": relevant,
        "recall@5": None if rr is None else float(rr > 0),
        "rr@10": rr,
        "precision@5": None if rr is None else 0.4,
        "ndcg@5": None if rr is None else 0.5,
        "correctly_rejected": False if not relevant else None,
        "latency_ms": 10.0,
    }


def _run(n_drafts: int, n_chunks: int, per_query: list[dict]) -> dict:
    return {
        "config": {
            "timestamp_utc": "t",
            "git_commit": "abcdef0123",
            "chunking": {"n_chunks": n_chunks},
            "inputs": {"corpus_sha256": {f"d{i}.tex": "x" for i in range(n_drafts)}},
        },
        "results": {"vector": {"per_query": per_query}},
    }


def test_old_set_row_scores_only_old_query_ids() -> None:
    old = _run(5, 50, [_query("q01", ["a"], 1.0), _query("q02", [], None)])
    new = _run(
        8,
        90,
        [_query("q01", ["a"], 1.0), _query("q02", [], None), _query("q03", ["b"], 0.5)],
    )
    rows = [line for line in compare(old, new).splitlines() if line.startswith("| vector")]

    assert len(rows) == 3
    old_row, new_row, subset_row = rows
    assert "5 drafts / 50 chunks | 1+1 | 1.000 | 1.000 | 0.400 | 0.500 | 0.000" in old_row
    assert "8 drafts / 90 chunks | 2+1 | 1.000 | 0.750 | 0.400 | 0.500 | 0.000" in new_row
    assert "8 drafts / 90 chunks | 1+1 | 1.000 | 1.000 | 0.400 | 0.500 | 0.000" in subset_row
