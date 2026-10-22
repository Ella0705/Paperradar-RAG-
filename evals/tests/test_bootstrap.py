"""Tests for evals/bootstrap.py."""

import pytest

from evals.bootstrap import answerable_scores, paired_bootstrap, percentile_interval


def test_percentile_interval_on_known_sequence() -> None:
    # 0..99: 5th percentile = 5th smallest = 4.0; 95th = 95th smallest = 94.0.
    samples = [float(i) for i in range(100)]
    assert percentile_interval(samples, 0.90) == (4.0, 94.0)
    assert percentile_interval(samples, 0.95) == (2.0, 97.0)  # ranks 3 and 98


def test_constant_scores_give_degenerate_intervals() -> None:
    out = paired_bootstrap({"bm25": [0.8] * 5, "vector": [1.0] * 5}, ("bm25", "vector"), 200)
    assert out["bm25"] == pytest.approx({"mean": 0.8, "ci_low": 0.8, "ci_high": 0.8})
    assert out["bm25 - vector"] == pytest.approx({"mean": -0.2, "ci_low": -0.2, "ci_high": -0.2})


def test_identical_modes_have_zero_paired_difference() -> None:
    values = [1.0, 0.6, 0.8, 0.4, 1.0, 0.2]
    out = paired_bootstrap({"bm25": values, "vector": list(values)}, ("bm25", "vector"), 500)
    assert out["bm25 - vector"]["ci_low"] == 0.0
    assert out["bm25 - vector"]["ci_high"] == 0.0
    assert out["bm25"]["ci_low"] < out["bm25"]["mean"] < out["bm25"]["ci_high"]


def test_same_seed_is_reproducible() -> None:
    scores = {"bm25": [1.0, 0.6, 0.8, 0.4], "vector": [1.0, 1.0, 0.8, 0.6]}
    assert paired_bootstrap(scores, ("bm25", "vector"), 300) == paired_bootstrap(
        scores, ("bm25", "vector"), 300
    )


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap({"bm25": [1.0], "vector": [1.0, 0.0]}, ("bm25", "vector"), 10)


def test_answerable_scores_aligns_by_query_id_and_drops_rejection() -> None:
    run = {
        "results": {
            "vector": {
                "per_query": [
                    {"query_id": "q02", "precision@5": 0.8},
                    {"query_id": "q01", "precision@5": 1.0},
                    {"query_id": "q16", "precision@5": None},
                ]
            },
            "bm25": {
                "per_query": [
                    {"query_id": "q01", "precision@5": 0.6},
                    {"query_id": "q16", "precision@5": None},
                    {"query_id": "q02", "precision@5": 0.4},
                ]
            },
        }
    }
    assert answerable_scores(run) == {"vector": [1.0, 0.8], "bm25": [0.6, 0.4]}
