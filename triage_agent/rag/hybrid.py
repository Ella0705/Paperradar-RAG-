"""BM25 关键词检索 + RRF 融合,与向量检索组成混合检索。"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    """小写化 + 按字母数字切词。英文与术语缩写足够;中文语料是已知局限。"""
    return re.findall(r"[a-z0-9]+", text.lower())


class Bm25Index:
    """在内存里为一批文档建 BM25 索引。"""

    def __init__(self, docs: list) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        corpus = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[:k] if scores[i] > 0]


def _doc_key(doc: object) -> str:
    meta = getattr(doc, "metadata", None) or {}
    path = str(meta.get("path", ""))
    section = str(meta.get("section", ""))
    body = (getattr(doc, "page_content", None) or "")[:200]
    return f"{path}|{section}|{body}"


def rrf_merge(ranked_lists: list[list], k: int = 60, top_n: int = 5) -> list:
    """Reciprocal rank fusion: score += 1 / (k + rank) per list."""
    scores: dict[str, float] = {}
    by_key: dict[str, object] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ordered[:top_n]]