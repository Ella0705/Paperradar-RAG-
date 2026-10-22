"""Unit tests for evals/metrics.py, including the rejection-query rules."""

import math

import pytest

from evals.metrics import (
    correctly_rejected,
    mean_defined,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    score_query,
)

# --- Recall@k ---------------------------------------------------------------


def test_recall_hit_in_top_k() -> None:
    assert recall_at_k(["gnn", "peft", "peft"], ["peft"], k=5) == 1.0


def test_recall_miss_when_relevant_only_after_k() -> None:
    ranked = ["gnn"] * 5 + ["peft"]
    assert recall_at_k(ranked, ["peft"], k=5) == 0.0


def test_recall_counts_repeated_chunks_of_a_draft_once() -> None:
    # Two relevant drafts, but all top-k chunks come from one of them.
    assert recall_at_k(["peft", "peft", "peft"], ["peft", "gnn"], k=5) == 0.5


def test_recall_partial_and_full_with_multiple_relevant() -> None:
    ranked = ["peft", "diffusion", "gnn"]
    assert recall_at_k(ranked, ["peft", "gnn"], k=2) == 0.5
    assert recall_at_k(ranked, ["peft", "gnn"], k=3) == 1.0


def test_recall_k_larger_than_result_list() -> None:
    assert recall_at_k(["peft"], ["peft"], k=10) == 1.0


def test_recall_empty_result_for_answerable_query_is_zero() -> None:
    assert recall_at_k([], ["peft"], k=5) == 0.0


def test_recall_undefined_for_rejection_query() -> None:
    assert recall_at_k(["peft"], [], k=5) is None
    assert recall_at_k([], [], k=5) is None


# --- Reciprocal rank@k ------------------------------------------------------


def test_rr_uses_first_relevant_rank() -> None:
    assert reciprocal_rank_at_k(["gnn", "peft", "peft"], ["peft"], k=10) == 0.5


def test_rr_rank_one() -> None:
    assert reciprocal_rank_at_k(["peft", "gnn"], ["peft"], k=10) == 1.0


def test_rr_zero_when_relevant_only_after_k() -> None:
    ranked = ["gnn"] * 10 + ["peft"]
    assert reciprocal_rank_at_k(ranked, ["peft"], k=10) == 0.0


def test_rr_relevant_exactly_at_k_counts() -> None:
    ranked = ["gnn"] * 9 + ["peft"]
    assert reciprocal_rank_at_k(ranked, ["peft"], k=10) == pytest.approx(0.1)


def test_rr_empty_result_for_answerable_query_is_zero() -> None:
    assert reciprocal_rank_at_k([], ["peft"], k=10) == 0.0


def test_rr_undefined_for_rejection_query() -> None:
    assert reciprocal_rank_at_k(["peft"], [], k=10) is None


# --- Precision@k ------------------------------------------------------------


def test_precision_counts_every_relevant_chunk() -> None:
    ranked = ["peft", "llm_quant", "peft", "peft", "llm_quant", "peft"]
    assert precision_at_k(ranked, ["peft"], k=5) == pytest.approx(3 / 5)


def test_precision_all_relevant_is_one() -> None:
    assert precision_at_k(["gan"] * 5, ["gan"], k=5) == 1.0


def test_precision_divides_by_k_even_for_short_results() -> None:
    assert precision_at_k(["peft", "peft"], ["peft"], k=5) == pytest.approx(2 / 5)


def test_precision_empty_result_for_answerable_query_is_zero() -> None:
    assert precision_at_k([], ["peft"], k=5) == 0.0


def test_precision_undefined_for_rejection_query() -> None:
    assert precision_at_k(["peft"], [], k=5) is None


# --- nDCG@k -----------------------------------------------------------------


def _dcg(relevant_ranks: list[int]) -> float:
    return sum(1 / math.log2(r + 1) for r in relevant_ranks)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["gan"] * 5, ["gan"], k=5, n_relevant_available=12) == pytest.approx(1.0)


def test_ndcg_matches_hand_computed_value() -> None:
    # Relevant at ranks 1, 3, 4; ideal is ranks 1..5.
    ranked = ["peft", "llm_quant", "peft", "peft", "llm_quant"]
    expected = _dcg([1, 3, 4]) / _dcg([1, 2, 3, 4, 5])
    assert ndcg_at_k(ranked, ["peft"], k=5, n_relevant_available=12) == pytest.approx(expected)


def test_ndcg_rewards_earlier_relevant_chunks() -> None:
    early = ["peft", "gan", "gan", "gan", "gan"]
    late = ["gan", "gan", "gan", "gan", "peft"]
    assert ndcg_at_k(early, ["peft"], 5, 12) > ndcg_at_k(late, ["peft"], 5, 12)


def test_ndcg_ideal_capped_by_available_relevant_chunks() -> None:
    # Only 2 relevant chunks exist, both retrieved at the top: perfect score.
    ranked = ["peft", "peft", "gan", "gan", "gan"]
    assert ndcg_at_k(ranked, ["peft"], k=5, n_relevant_available=2) == pytest.approx(1.0)


def test_ndcg_ignores_ranks_after_k() -> None:
    ranked = ["gan"] * 5 + ["peft"]
    assert ndcg_at_k(ranked, ["peft"], k=5, n_relevant_available=12) == 0.0


def test_ndcg_zero_when_nothing_relevant_is_indexed() -> None:
    assert ndcg_at_k(["gan"], ["peft"], k=5, n_relevant_available=0) == 0.0


def test_ndcg_undefined_for_rejection_query() -> None:
    assert ndcg_at_k(["peft"], [], k=5, n_relevant_available=0) is None


# --- score_query ------------------------------------------------------------


def test_score_query_answerable() -> None:
    ranked = ["llm_quant", "peft", "peft", "peft", "peft", "peft"]
    scores = score_query(ranked, ["peft"], {"peft": 12, "llm_quant": 13})
    assert scores["recall@5"] == 1.0
    assert scores["rr@10"] == 0.5
    assert scores["precision@5"] == pytest.approx(4 / 5)
    assert scores["ndcg@5"] == pytest.approx(_dcg([2, 3, 4, 5]) / _dcg([1, 2, 3, 4, 5]))
    assert scores["correctly_rejected"] is None


def test_score_query_rejection() -> None:
    scores = score_query(["gnn"], [], {"gnn": 11})
    assert {k: v for k, v in scores.items() if k != "correctly_rejected"} == {
        "recall@5": None,
        "rr@10": None,
        "precision@5": None,
        "ndcg@5": None,
    }
    assert scores["correctly_rejected"] is False


# --- Rejection --------------------------------------------------------------


def test_rejection_query_with_empty_result_is_correct() -> None:
    assert correctly_rejected([], []) is True


def test_rejection_query_with_any_result_is_wrong() -> None:
    assert correctly_rejected(["gnn"], []) is False


def test_rejection_not_applicable_to_answerable_query() -> None:
    assert correctly_rejected([], ["peft"]) is None


# --- Means ------------------------------------------------------------------


def test_mean_skips_undefined_values() -> None:
    # Rejection queries contribute None and must not drag the mean down.
    assert mean_defined([1.0, None, 0.5, None]) == 0.75


def test_mean_counts_bools_as_zero_one() -> None:
    assert mean_defined([True, False, None, False]) == pytest.approx(1 / 3)


def test_mean_all_undefined_is_none() -> None:
    assert mean_defined([None, None]) is None
    assert mean_defined([]) is None
