"""Unit tests for evals/metrics.py, including the rejection-query rules."""

import pytest

from evals.metrics import correctly_rejected, mean_defined, recall_at_k, reciprocal_rank_at_k

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
