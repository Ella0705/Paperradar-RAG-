"""Tests for academic RAG chunking and hybrid fusion."""

from langchain_core.documents import Document

from triage_agent.rag.hybrid import rrf_merge
from triage_agent.rag.latex_splitter import split_latex_by_section


def test_split_latex_by_section_keeps_section_body_together() -> None:
    tex = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
We study LoRA for PEFT.
\section{Related Work}
BitFit and adapters are prior art.
\end{document}
"""
    sections = split_latex_by_section(tex)
    titles = [t for t, _ in sections]
    assert "Introduction" in titles
    assert "Related Work" in titles
    related = next(body for t, body in sections if t == "Related Work")
    assert "BitFit" in related
    assert r"\section" not in related


def test_rrf_merge_prefers_chunks_ranked_in_both_lists() -> None:
    a = Document(page_content="LoRA on GPUs", metadata={"path": "a.tex", "section": "Intro"})
    b = Document(page_content="BitFit baseline", metadata={"path": "a.tex", "section": "RW"})
    c = Document(page_content="Unrelated note", metadata={"path": "b.md"})

    vector_ranked = [a, c]
    bm25_ranked = [a, b]
    merged = rrf_merge([vector_ranked, bm25_ranked], top_n=2)

    assert merged[0].page_content == "LoRA on GPUs"
    assert len(merged) == 2
    # Ranked in both lists → highest RRF score; second slot may tie BM25-only vs vector-only hits.
    assert "LoRA on GPUs" in {d.page_content for d in merged}
