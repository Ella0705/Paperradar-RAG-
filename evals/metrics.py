"""Retrieval metrics for the eval: Recall@k, reciprocal rank@k, rejection.

Unit of ranking: the retriever returns *chunks*. Each chunk is mapped to the
draft it came from (its `local_id`), giving a ranked list of draft ids that
may contain repeats, e.g. ["gnn", "gnn", "peft", ...]. Relevance labels are
per draft: a chunk is relevant if its draft id is in `relevant_draft_ids`.

Ranking chunks rather than deduplicated drafts matters here: the corpus has
only 5 drafts, so a deduplicated top-5 would contain every draft and
draft-level Recall@5 would be 1.0 for almost any retriever.

Scoring rules:
- Recall@k = |distinct relevant drafts among the top-k chunks| / |relevant drafts|.
  Several chunks of the same relevant draft count once.
- RR@k = 1 / rank of the first relevant chunk within the top k (ranks start
  at 1); 0.0 if no relevant chunk is in the top k. MRR@k is its mean.
- Rejection queries (empty `relevant_draft_ids`): Recall and RR are undefined
  and returned as None, so they are excluded from the Recall/MRR means.
  They are scored separately: a rejection query is "correctly rejected" iff
  the retriever returned no chunks at all. Rejection rate = share of
  rejection queries that were correctly rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Recall@k over draft ids; None for a rejection query (no relevant drafts)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    hits = relevant_set & set(ranked_ids[:k])
    return len(hits) / len(relevant_set)


def reciprocal_rank_at_k(
    ranked_ids: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float | None:
    """1/rank of the first relevant id in the top k; None for a rejection query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    for rank, draft_id in enumerate(ranked_ids[:k], start=1):
        if draft_id in relevant_set:
            return 1.0 / rank
    return 0.0


def correctly_rejected(ranked_ids: Sequence[str], relevant: Iterable[str]) -> bool | None:
    """True iff a rejection query got an empty result; None for answerable queries."""
    if set(relevant):
        return None
    return len(ranked_ids) == 0


def mean_defined(values: Iterable[float | bool | None]) -> float | None:
    """Mean of the non-None values (bools count as 0/1); None if there are none."""
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
