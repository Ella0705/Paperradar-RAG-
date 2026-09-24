"""Local Overlap Agent — compares a paper against the user's local drafts.

This agent looks at the target paper's abstract and the local manifest
(`local_kb/local_manifest.json`) and summarizes where the target paper
overlaps with or is relevant to the user's own work.

For now we only use the JSON manifest (no local_profile.md), to keep the
implementation simple and focused.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from triage_agent.agents.base import BaseAgent
from triage_agent.local_kb import LocalManifest, LocalPaper, load_local_manifest
from triage_agent.models.memo import LocalOverlapMatch, LocalOverlapReport
from triage_agent.models.paper import PaperCard
from triage_agent.utils.llm import call_llm_json

logger = logging.getLogger(__name__)

_RAG_MOD_ERROR: str | None = None
try:  # optional LangChain + Chroma; overlap falls back to manifest-only
    from triage_agent.rag.retriever import (
        LocalRAGRetriever,
        format_retrieved_context,
        should_use_rag,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    _RAG_MOD_ERROR = str(exc)
    should_use_rag = None  # type: ignore[assignment]
    LocalRAGRetriever = None  # type: ignore[assignment]
    format_retrieved_context = None  # type: ignore[assignment]


LOCAL_SYSTEM_PROMPT = """\
You are a research assistant helping a researcher decide which new papers
are relevant to their own ongoing work.

You are given:
- A TARGET PAPER (title + abstract)
- A list of LOCAL DRAFTS (each with id, title, abstract) representing the
  researcher's current projects, notes, or ideas.

Your job is to:
1. Identify which local drafts the target paper is most related to.
2. For each related local draft, briefly summarize how the target paper
   overlaps with or is relevant to that draft (methods, goals, setting, etc.).
3. Assign a relevance score from 0.0 (not related) to 1.0 (highly relevant)
   for each local draft.
4. Assign a normalized relationship label for each related local draft. Use one
   of: extends_your_work, competes_with_your_idea, method_transfer,
   citation_candidate, background_context, same_problem_different_method, related.
5. Provide an overall relevance score from 0.0 to 1.0 for how important this
   target paper is to the researcher's current work.
"""


LOCAL_USER_PROMPT = """\
TARGET PAPER:
Title: {title}
Abstract: {abstract}

LOCAL DRAFTS:
{local_list}

Analyze how the TARGET PAPER relates to the LOCAL DRAFTS above.

Return a JSON object with:
- "matches": a list of objects, each with:
    - "local_id": string (from the input)
    - "local_title": string
    - "relevance": float between 0.0 and 1.0
    - "relationship_type": one of: extends_your_work, competes_with_your_idea,
      method_transfer, citation_candidate, background_context,
      same_problem_different_method, related
    - "overlap_summary": short string (1-3 sentences) describing the overlap
- "overall_relevance": float between 0.0 and 1.0
"""

RAG_SYSTEM_PROMPT = LOCAL_SYSTEM_PROMPT + """

You are also given RETRIEVED RESEARCH CONTEXT from a vector store over the user's
drafts, notes, and files (the most relevant text chunks to this paper by embedding
similarity). Use this as primary evidence; use the LOCAL DRAFT INDEX to align local_id
and titles with manifest entries when possible. If a match only appears in
retrieved text, set local_id from the [local_id: ...] headers in the context.
"""

RAG_USER_PROMPT = """\
TARGET PAPER:
Title: {title}
Abstract: {abstract}

RETRIEVED RESEARCH CONTEXT (from vector RAG, top similar chunks to this paper's title+abstract):
{retrieved_context}

LOCAL DRAFT INDEX (id + title from manifest, if any; prefer these when appropriate):
{local_index}

Return a JSON object with the same structure as the non-RAG case:
- "matches" (list with local_id, local_title, relevance, relationship_type, overlap_summary)
- "overall_relevance" (float 0-1)
"""


def _format_local_index(manifest: LocalManifest) -> str:
    if not manifest.papers:
        return (
            "(No `papers` in manifest; use `local_id` from retrieved context headers only.)"
        )
    return "\n".join(f"- id={p.id} | {p.title}" for p in manifest.papers)


def _rag_retrieve_text(paper: PaperCard) -> str:
    """Synchronous: query vector store. Called via asyncio.to_thread from async run."""
    if not should_use_rag or not LocalRAGRetriever or not format_retrieved_context:
        return ""
    if not should_use_rag():
        return ""
    r = LocalRAGRetriever()
    q = f"Title: {paper.title}\n\nAbstract:\n{paper.abstract}"
    docs = r.retrieve(q)
    return format_retrieved_context(docs)


class LocalOverlapAgent(BaseAgent):
    """Assesses overlap between the target paper and local drafts."""

    @property
    def name(self) -> str:
        return "Local Overlap"

    async def run(self, paper: PaperCard) -> LocalOverlapReport:
        """Compare the target paper against the user's local drafts.

        If no local manifest is found, returns an empty LocalOverlapReport
        with overall_relevance = 0.0.
        """
        manifest = load_local_manifest()
        if manifest is None:
            return LocalOverlapReport(matches=[], overall_relevance=0.0)
        if not manifest.papers and not manifest.sources:
            return LocalOverlapReport(matches=[], overall_relevance=0.0)

        use_rag = bool(
            _RAG_MOD_ERROR is None
            and should_use_rag
            and should_use_rag()  # type: ignore[misc]
        )
        retrieved = ""
        if use_rag:
            try:
                retrieved = await asyncio.to_thread(_rag_retrieve_text, paper)
            except Exception as exc:  # pragma: no cover
                logger.warning("RAG retrieve failed: %s", exc)
                retrieved = ""

        if use_rag and retrieved.strip():
            user_prompt = RAG_USER_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract,
                retrieved_context=retrieved,
                local_index=_format_local_index(manifest),
            )
            system_prompt = RAG_SYSTEM_PROMPT
        elif manifest.papers:
            # Legacy: full manifest abstract list (RAG off, empty, or not installed)
            user_prompt = LOCAL_USER_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract,
                local_list=_format_local_list(manifest),
            )
            system_prompt = LOCAL_SYSTEM_PROMPT
        else:
            return LocalOverlapReport(matches=[], overall_relevance=0.0)

        try:
            raw: dict[str, Any] = await call_llm_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # pragma: no cover - network/pathological errors
            logger.warning(
                "Local overlap LLM call failed for '%s': %s",
                paper.title,
                exc,
            )
            return LocalOverlapReport(matches=[], overall_relevance=0.0)

        return _parse_local_overlap_response(raw, manifest)


def _format_local_list(manifest: LocalManifest) -> str:
    """Format local drafts as a bullet list for the prompt."""
    if not manifest.papers:
        return "(No local drafts provided)"
    lines: list[str] = []
    for p in manifest.papers:
        lines.append(f"- id={p.id} | {p.title}")
        lines.append(f"  abstract: {p.abstract}")
    return "\n".join(lines)


def _parse_local_overlap_response(
    raw: dict[str, Any],
    manifest: LocalManifest,
) -> LocalOverlapReport:
    """Convert the LLM JSON response into a LocalOverlapReport."""
    matches_raw = raw.get("matches") or []
    overall_raw = raw.get("overall_relevance", 0.0)

    try:
        overall = float(overall_raw)
    except (TypeError, ValueError):
        overall = 0.0
    overall = max(0.0, min(1.0, overall))

    # Build a quick lookup from local_id to LocalPaper for nicer fallbacks.
    by_id: dict[str, LocalPaper] = {p.id: p for p in manifest.papers}

    matches: list[LocalOverlapMatch] = []
    if isinstance(matches_raw, list):
        for item in matches_raw:
            if not isinstance(item, dict):
                continue
            local_id = str(item.get("local_id", "")).strip()
            if not local_id:
                continue

            lp = by_id.get(local_id)
            local_title = str(item.get("local_title") or (lp.title if lp else "")).strip()

            try:
                relevance = float(item.get("relevance", 0.0))
            except (TypeError, ValueError):
                relevance = 0.0
            relevance = max(0.0, min(1.0, relevance))

            relationship_type = _normalize_relationship_type(item.get("relationship_type"))
            overlap_summary = str(item.get("overlap_summary", "")).strip()

            matches.append(
                LocalOverlapMatch(
                    local_id=local_id,
                    local_title=local_title or (lp.title if lp else local_id),
                    relevance=relevance,
                    relationship_type=relationship_type,
                    overlap_summary=overlap_summary,
                )
            )

    return LocalOverlapReport(matches=matches, overall_relevance=overall)


def _normalize_relationship_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    allowed = {
        "extends_your_work",
        "competes_with_your_idea",
        "method_transfer",
        "citation_candidate",
        "background_context",
        "same_problem_different_method",
        "related",
    }
    if value in allowed:
        return value
    return "related"
