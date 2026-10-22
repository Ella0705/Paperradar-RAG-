"""Pluggable reranker hook applied to retrieved chunks in the eval runner.

Only "none" (identity) is implemented. "jev" is a registered placeholder for a
later experiment; selecting it fails loudly instead of silently falling back.
No external reranker API is called from here.
"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[Any]) -> list[Any]:
        """Return `docs` reordered (and possibly truncated) for `query`."""
        ...


class NoneReranker:
    """Identity reranker: keeps the retriever's order."""

    name = "none"

    def rerank(self, query: str, docs: list[Any]) -> list[Any]:
        return list(docs)


RERANKER_CHOICES = ("none", "jev")


def get_reranker(name: str) -> Reranker:
    if name == "none":
        return NoneReranker()
    if name == "jev":
        raise NotImplementedError("reranker 'jev' is reserved but not implemented yet")
    raise ValueError(f"unknown reranker {name!r}; choose from {RERANKER_CHOICES}")
