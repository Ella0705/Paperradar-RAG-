"""Vector-store retrieval: hybrid MMR + BM25/RRF, return top-k chunks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from triage_agent.rag.hybrid import Bm25Index, rrf_merge
from triage_agent.rag.paths import default_chroma_persist_path

logger = logging.getLogger(__name__)

_DEFAULT_K = 5
_DEFAULT_FETCH_K = 20
_DEFAULT_MMR_LAMBDA = 0.5


def _load_chroma(
    persist_directory: Path,
) -> Any:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=os.getenv("RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def chroma_index_ready(persist_directory: Path | None = None) -> bool:
    """True if a persisted Chroma index directory exists and is non-empty."""
    p = persist_directory or default_chroma_persist_path()
    if not p.is_dir():
        return False
    if not any(p.iterdir()):
        return False
    return True


def should_use_rag() -> bool:
    """RAG is used when the user did not opt out, the index exists, and OpenAI key is set."""
    if os.getenv("TRIAGE_USE_RAG", "").strip().lower() in {"0", "false", "no"}:
        return False
    if not bool(os.getenv("OPENAI_API_KEY", "").strip()):
        return False
    return chroma_index_ready()


def _all_chunks_from_store(vectorstore: Any) -> list[Any]:
    """Load every indexed chunk from Chroma for BM25 (local KB sizes only)."""
    from langchain_core.documents import Document

    try:
        collection = vectorstore._collection  # noqa: SLF001 — Chroma API
        payload = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("Could not load full chunk list for BM25: %s", exc)
        return []

    texts = payload.get("documents") or []
    metas = payload.get("metadatas") or []
    docs: list[Any] = []
    for text, meta in zip(texts, metas, strict=False):
        if not text:
            continue
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def _hybrid_enabled() -> bool:
    flag = os.getenv("RAG_HYBRID", "1").strip().lower()
    return flag not in {"0", "false", "no"}


def format_retrieved_context(docs: list[Any]) -> str:
    """Turn LangChain documents into a block for the overlap prompt."""
    if not docs:
        return ""
    parts: list[str] = []
    for doc in docs:
        meta: dict[str, Any] = getattr(doc, "metadata", {}) or {}
        local_id = str(meta.get("local_id", "") or meta.get("path", "") or "unknown")
        source = str(meta.get("source", "") or meta.get("type", "local"))
        section = str(meta.get("section", "") or "").strip()
        section_bit = f" | section: {section}" if section else ""
        text = (getattr(doc, "page_content", None) or "").strip()
        if not text:
            continue
        parts.append(
            f"[local_id: {local_id} | source: {source}{section_bit}]\n{text}",
        )
    return "\n\n---\n\n".join(parts)


class LocalRAGRetriever:
    """Retrieve top-k chunks: MMR vector search, optionally fused with BM25 via RRF."""

    def __init__(
        self,
        k: int = _DEFAULT_K,
        persist_directory: Path | None = None,
    ) -> None:
        self.k = k
        self.persist_directory = persist_directory or default_chroma_persist_path()
        self._vectorstore: Any = None
        self._bm25: Bm25Index | None = None
        self._bm25_doc_count: int = 0

    @property
    def vectorstore(self) -> Any:
        if self._vectorstore is None:
            self._vectorstore = _load_chroma(self.persist_directory)
        return self._vectorstore

    def _mmr_retrieve(self, query: str, fetch_k: int) -> list[Any]:
        lambda_mult = float(os.getenv("RAG_MMR_LAMBDA", str(_DEFAULT_MMR_LAMBDA)))
        pool = max(fetch_k * 2, _DEFAULT_FETCH_K)
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": fetch_k,
                "fetch_k": pool,
                "lambda_mult": lambda_mult,
            },
        )
        return list(retriever.invoke(query))

    def _bm25_index(self, corpus: list[Any]) -> Bm25Index | None:
        if not corpus:
            return None
        if self._bm25 is None or self._bm25_doc_count != len(corpus):
            self._bm25 = Bm25Index(corpus)
            self._bm25_doc_count = len(corpus)
        return self._bm25

    def retrieve(self, query: str) -> list[Any]:
        """Synchronous search; run from `asyncio.to_thread` in async code."""
        if not query.strip():
            return []

        k = int(os.getenv("RAG_TOP_K", str(self.k)))
        fetch_k = int(os.getenv("RAG_FETCH_K", str(max(k * 2, _DEFAULT_FETCH_K))))

        try:
            vector_docs = self._mmr_retrieve(query, fetch_k=fetch_k)

            if not _hybrid_enabled():
                return vector_docs[:k]

            all_docs = _all_chunks_from_store(self.vectorstore)
            if not all_docs:
                return vector_docs[:k]

            bm25 = self._bm25_index(all_docs)
            if bm25 is None:
                return vector_docs[:k]

            bm25_docs = bm25.search(query, k=fetch_k)
            return rrf_merge([vector_docs, bm25_docs], top_n=k)
        except Exception as exc:
            logger.warning("RAG retrieve failed: %s", exc)
            return []


def retrieve_for_paper_query(title: str, abstract: str) -> str:
    """Build query from paper fields and return formatted context (or empty)."""
    if not should_use_rag():
        return ""
    q = f"Title: {title}\n\nAbstract:\n{abstract}"
    retriever = LocalRAGRetriever()
    docs = retriever.retrieve(q)
    return format_retrieved_context(docs)
