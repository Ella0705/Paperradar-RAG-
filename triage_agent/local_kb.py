"""Local knowledge base loading utilities.

This module loads locally-authored paper drafts / notes from a JSON manifest,
so that downstream agents can compare incoming papers against the user's
current work and interests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class LocalPaper(BaseModel):
    """A locally-authored or tracked paper/draft."""

    id: str = Field(description="Local identifier for the paper/draft")
    title: str = Field(description="Title of the local paper/draft")
    abstract: str = Field(description="Short description or abstract")


class LocalFileSource(BaseModel):
    """On-disk file to index for RAG (Overleaf mirror, README, notes, etc.)."""

    path: str = Field(description="Absolute or project-root-relative path")
    id: str = Field(
        default="",
        description="Stable id for overlap matches; defaults to a slug from the path",
    )
    source: str = Field(default="", description="Human-readable label (e.g. repo name)")
    type: str = Field(default="local", description="local | overleaf | github | ...")


class LocalManifest(BaseModel):
    """Manifest of local papers/drafts."""

    papers: list[LocalPaper] = Field(
        default_factory=list,
        description="List of local papers/drafts to compare against",
    )
    sources: list[LocalFileSource] = Field(
        default_factory=list,
        description="Optional file paths to chunk+embed for vector RAG (see triage_agent.rag)",
    )


def _default_manifest_path() -> Path:
    """Compute the default path to the local manifest JSON file.

    Resolution order:
    1. LOCAL_MANIFEST_PATH env var, if set
    2. LOCAL_KB_DIR/local_manifest.json, if LOCAL_KB_DIR is set
    3. TRIAGE_PROJECT_ROOT/local_kb/local_manifest.json, if TRIAGE_PROJECT_ROOT is set
    4. ./local_kb/local_manifest.json (relative to CWD)
    5. <repo_root>/local_kb/local_manifest.json (best-effort fallback)
    """
    env_path = os.getenv("LOCAL_MANIFEST_PATH")
    if env_path:
        return Path(env_path)

    kb_dir = os.getenv("LOCAL_KB_DIR")
    if kb_dir:
        return Path(kb_dir) / "local_manifest.json"

    project_root = os.getenv("TRIAGE_PROJECT_ROOT")
    if project_root:
        project_candidate = Path(project_root) / "local_kb" / "local_manifest.json"
        if project_candidate.exists():
            return project_candidate

    cwd_candidate = Path("./local_kb/local_manifest.json")
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[1] / "local_kb" / "local_manifest.json"
    return repo_candidate


def load_local_manifest(path: str | Path | None = None) -> LocalManifest | None:
    """Load the local manifest JSON describing user's own drafts/papers.

    Args:
        path: Optional explicit path to the manifest JSON file. If not
            provided, a default is resolved via environment variables.

    Returns:
        A LocalManifest instance if the file exists and is valid, otherwise
        None (in which case callers should gracefully fall back to
        "no local knowledge available").
    """
    manifest_path = Path(path) if path is not None else _default_manifest_path()

    if not manifest_path.exists():
        return None

    try:
        raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # If the manifest is invalid JSON, treat as absent to avoid
        # breaking the main triage flow.
        return None

    try:
        return LocalManifest.model_validate(raw)
    except ValidationError:
        # Invalid schema; ignore rather than raising in the main path.
        return None


def write_local_manifest(
    manifest_data: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Write manifest data to a JSON file and return the resolved path."""
    manifest_path = Path(path) if path is not None else _default_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2),
        encoding="utf-8",
    )
    return manifest_path
