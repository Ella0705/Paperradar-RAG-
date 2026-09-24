"""RAG: chunk local drafts, embed with OpenAI, retrieve from Chroma for LocalOverlap."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from triage_agent.rag.paths import default_chroma_persist_path, manifest_dir, project_root
from triage_agent.rag.retriever import (
    LocalRAGRetriever,
    chroma_index_ready,
    format_retrieved_context,
    retrieve_for_paper_query,
    should_use_rag,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    build_index: Callable[..., int]
    load_documents_from_manifest: Any


def __getattr__(name: str) -> Any:
    if name in ("build_index", "load_documents_from_manifest"):
        mod = importlib.import_module("triage_agent.rag.build_index")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocalRAGRetriever",
    "build_index",
    "chroma_index_ready",
    "default_chroma_persist_path",
    "format_retrieved_context",
    "load_documents_from_manifest",
    "manifest_dir",
    "project_root",
    "retrieve_for_paper_query",
    "should_use_rag",
]
