"""Default paths for the local Chroma RAG index."""

from __future__ import annotations

import os
from pathlib import Path

from triage_agent.local_kb import _default_manifest_path


def manifest_dir() -> Path:
    return _default_manifest_path().parent


def default_chroma_persist_path() -> Path:
    """Directory where Chroma stores the on-disk index."""
    override = os.getenv("RAG_CHROMA_PATH", "").strip()
    if override:
        return Path(override)
    return manifest_dir() / "chroma_index"


def project_root() -> Path:
    """Best-effort repo root (parent of `local_kb/`)."""
    return manifest_dir().parent
