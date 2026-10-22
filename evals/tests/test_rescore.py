"""Tests for evals/rescore.py (runs that already record chunks_per_draft)."""

import pytest

from evals.rescore import rescore_run


def _run(stored_rr: float) -> dict:
    retrieved = [
        {"rank": i, "local_id": d, "section": "", "preview": ""}
        for i, d in enumerate(["llm_quant", "peft", "peft", "peft", "peft"], start=1)
    ]
    query = {
        "query_id": "q21",
        "relevant_draft_ids": ["peft"],
        "recall@5": 1.0,
        "rr@10": stored_rr,
        "correctly_rejected": None,
        "latency_ms": 5.0,
        "retrieved": retrieved,
    }
    return {
        "config": {"chunking": {"chunks_per_draft": {"peft": 12, "llm_quant": 13}}},
        "results": {"bm25": {"summary": {}, "per_query": [query]}},
    }


def test_rescore_adds_new_metrics_and_summary() -> None:
    run = _run(stored_rr=0.5)
    out = rescore_run(run)
    q = out["results"]["bm25"]["per_query"][0]
    assert q["precision@5"] == pytest.approx(4 / 5)
    assert 0 < q["ndcg@5"] < 1
    assert out["results"]["bm25"]["summary"]["precision@5"] == pytest.approx(4 / 5)
    assert out["rescore"]["chunks_per_draft_source"] == "config"
    assert "precision@5" not in run["results"]["bm25"]["per_query"][0]  # input untouched


def test_rescore_stops_when_stored_metric_is_not_reproduced() -> None:
    with pytest.raises(RuntimeError, match="rr@10"):
        rescore_run(_run(stored_rr=1.0))
