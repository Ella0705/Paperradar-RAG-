"""Tests for local-overlap parsing and relationship labels."""

from triage_agent.agents.local_overlap import _parse_local_overlap_response
from triage_agent.local_kb import LocalFileSource, LocalManifest, LocalPaper


def test_parse_local_overlap_response_preserves_relationship_type() -> None:
    manifest = LocalManifest(
        papers=[
            LocalPaper(
                id="draft-1",
                title="Informal Bids",
                abstract="A draft on bidder learning.",
            )
        ]
    )

    report = _parse_local_overlap_response(
        {
            "matches": [
                {
                    "local_id": "draft-1",
                    "local_title": "Informal Bids",
                    "relevance": 0.91,
                    "relationship_type": "competes_with_your_idea",
                    "overlap_summary": "Targets the same market-design question.",
                }
            ],
            "overall_relevance": 0.91,
        },
        manifest,
    )

    assert report.matches[0].relationship_type == "competes_with_your_idea"


def test_parse_local_overlap_response_normalizes_unknown_relationship_type() -> None:
    manifest = LocalManifest(
        papers=[
            LocalPaper(
                id="draft-1",
                title="Informal Bids",
                abstract="A draft on bidder learning.",
            )
        ]
    )

    report = _parse_local_overlap_response(
        {
            "matches": [
                {
                    "local_id": "draft-1",
                    "relevance": 0.77,
                    "relationship_type": "mystery_bucket",
                    "overlap_summary": "Still relevant, but the label is invalid.",
                }
            ],
            "overall_relevance": 0.77,
        },
        manifest,
    )

    assert report.matches[0].relationship_type == "related"


def test_local_manifest_accepts_rag_file_sources() -> None:
    m = LocalManifest(
        papers=[],
        sources=[LocalFileSource(path="notes/draft.md", id="d1", source="notes", type="local")],
    )
    assert m.sources[0].path == "notes/draft.md"
    assert m.sources[0].id == "d1"
