"""Tests for evals/analysis.py using small hand-built run payloads."""

from evals.analysis import rival_intrusions, rrf_trace


def _chunk(rank: int, draft: str, section: str = "S") -> dict:
    return {"rank": rank, "local_id": draft, "section": section, "preview": f"{draft}-{section}"}


def _query(query_id: str, relevant: list[str], drafts: list[str]) -> dict:
    retrieved = [_chunk(i, d, f"s{i}") for i, d in enumerate(drafts, start=1)]
    return {"query_id": query_id, "relevant_draft_ids": relevant, "retrieved": retrieved}


def test_rival_intrusions_counts_only_rival_chunks_in_top5() -> None:
    per_query = [
        # Rival at rank 1 and 4; a gnn chunk is not a rival of peft; rank 6 ignored.
        _query("q21", ["peft"], ["llm_quant", "peft", "gnn", "llm_quant", "peft", "llm_quant"]),
        _query("q22", ["gan"], ["gan"] * 5),
        _query("q07", ["gnn"], ["peft"] * 5),  # gnn has no rival: skipped
        _query("q16", [], ["peft"] * 5),  # rejection query: skipped
    ]
    stats = rival_intrusions(per_query, {"q21", "q22", "q07", "q16"})
    assert stats == {
        "queries": 2,
        "slots": 10,
        "rival_chunks": 2,
        "queries_hit": 1,
        "rival_at_rank1": 1,
    }


def test_rival_intrusions_respects_query_id_filter() -> None:
    per_query = [_query("q21", ["peft"], ["llm_quant"] * 5)]
    assert rival_intrusions(per_query, {"q99"})["queries"] == 0


def test_rrf_trace_reports_exact_tie() -> None:
    a, b = _chunk(1, "llm_quant", "RW"), _chunk(4, "peft", "Quant")
    vector = [a, _chunk(2, "x", "1"), _chunk(3, "x", "2"), b]
    bm25 = [{**b, "rank": 1}, _chunk(2, "y", "1"), _chunk(3, "y", "2"), {**a, "rank": 4}]
    hybrid = [{**a, "rank": 1}, {**b, "rank": 2}]
    run = {
        "config": {"retrieval": {"rrf_k": 60}},
        "results": {
            mode: {"per_query": [{"query_id": "q23", "retrieved": lst}]}
            for mode, lst in (("vector", vector), ("bm25", bm25), ("hybrid", hybrid))
        },
    }
    lines = rrf_trace(run, "q23", top=2).splitlines()
    data_rows = lines[lines.index("|---|---|---|---|---|---|") + 1 :]
    score = f"{1 / 61 + 1 / 64:.6f}"
    assert data_rows == [
        f"| llm_quant / RW | 1 | 4 | {score} | yes | 1 |",
        f"| peft / Quant | 4 | 1 | {score} | yes | 2 |",
    ]
