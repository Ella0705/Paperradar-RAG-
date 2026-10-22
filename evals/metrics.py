"""Retrieval metrics for the eval: Recall@k, RR@k, Precision@k, nDCG@k, rejection.

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
- Precision@k = relevant chunks in the top k / k. The denominator is always
  k, so a result list shorter than k is penalised for the missing slots.
- nDCG@k uses binary gain per chunk (1 if relevant) and the log2(rank + 1)
  discount. The ideal ranking puts min(k, available) relevant chunks first,
  where `available` is the number of relevant chunks in the index; it must
  be passed in, because the ranked list alone cannot tell.
- Rejection queries (empty `relevant_draft_ids`): Recall, RR, Precision and
  nDCG are undefined and returned as None, so they are excluded from means.
  They are scored separately: a rejection query is "correctly rejected" iff
  the retriever returned no chunks at all. Rejection rate = share of
  rejection queries that were correctly rejected.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Cutoffs reported by the eval (score_query and the result-file keys use these).
RECALL_K = 5
MRR_K = 10
PRECISION_K = 5
NDCG_K = 5


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


def precision_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Share of the k slots filled by relevant chunks; None for a rejection query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    hits = sum(1 for draft_id in ranked_ids[:k] if draft_id in relevant_set)
    return hits / k


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant: Iterable[str],
    k: int,
    n_relevant_available: int,
) -> float | None:
    """Binary-gain nDCG@k; None for a rejection query.

    `n_relevant_available` is how many relevant chunks exist in the index.
    Returns 0.0 if it is 0 (nothing relevant could be retrieved).
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, draft_id in enumerate(ranked_ids[:k], start=1)
        if draft_id in relevant_set
    )
    n_ideal = min(k, n_relevant_available)
    if n_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg


def score_query(
    ranked_ids: Sequence[str],
    relevant: Sequence[str],
    chunks_per_draft: Mapping[str, int],
) -> dict[str, Any]:
    """All per-query metrics used by the eval, keyed as stored in result files."""
    available = sum(chunks_per_draft.get(draft_id, 0) for draft_id in set(relevant))
    return {
        f"recall@{RECALL_K}": recall_at_k(ranked_ids, relevant, RECALL_K),
        f"rr@{MRR_K}": reciprocal_rank_at_k(ranked_ids, relevant, MRR_K),
        f"precision@{PRECISION_K}": precision_at_k(ranked_ids, relevant, PRECISION_K),
        f"ndcg@{NDCG_K}": ndcg_at_k(ranked_ids, relevant, NDCG_K, available),
        "correctly_rejected": correctly_rejected(ranked_ids, relevant),
    }


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
