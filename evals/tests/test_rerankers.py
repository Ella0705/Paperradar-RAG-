"""Tests for the reranker hook."""

import pytest

from evals.rerankers import get_reranker


def test_none_reranker_keeps_order_and_copies() -> None:
    docs = ["a", "b", "c"]
    out = get_reranker("none").rerank("q", docs)
    assert out == docs
    assert out is not docs


def test_jev_is_reserved_but_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        get_reranker("jev")


def test_unknown_reranker_rejected() -> None:
    with pytest.raises(ValueError):
        get_reranker("cohere")
