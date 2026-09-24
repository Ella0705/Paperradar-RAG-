#!/usr/bin/env python3
"""Entry point for running triage from the OpenClaw skill.

Usage (from skill context):
    python3 scripts/run_triage.py <arxiv-id-or-url> [--format json|markdown] [--output path]

This script bridges OpenClaw skill execution to either:
1) local Python package imports (triage_agent), or
2) installed `triage` CLI fallback if imports are unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _discover_project_root() -> Path | None:
    """Best-effort discovery of repo root containing `triage_agent/`."""
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "triage_agent").is_dir():
            return candidate
    return None


PROJECT_ROOT = _discover_project_root()
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer direct provider when key is present (so Docker/OpenClaw can use real LLM).
if os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("LLM_BACKEND", "openai")
elif os.getenv("ANTHROPIC_API_KEY"):
    os.environ.setdefault("LLM_BACKEND", "anthropic")
elif shutil.which("openclaw"):
    os.environ.setdefault("LLM_BACKEND", "openclaw")
    os.environ.setdefault("OPENCLAW_LLM_AGENT", "main")

if PROJECT_ROOT:
    os.environ.setdefault("TRIAGE_PROJECT_ROOT", str(PROJECT_ROOT))
    default_manifest = PROJECT_ROOT / "local_kb" / "local_manifest.json"
    if default_manifest.exists():
        os.environ.setdefault("LOCAL_MANIFEST_PATH", str(default_manifest))

try:
    from triage_agent.sources.sync import sync_all_sources as _sync_all_sources
except ModuleNotFoundError:
    sync_all_sources: Callable[[], object] | None = None
else:
    sync_all_sources = _sync_all_sources


SKILL_DIR = Path(__file__).resolve().parent.parent
MEMORY_DIR = SKILL_DIR / "memory"
SEEN_FILE = MEMORY_DIR / "seen.json"


def load_seen() -> dict[str, str]:
    """Load the set of previously triaged paper IDs."""
    if SEEN_FILE.exists():
        payload = json.loads(SEEN_FILE.read_text())
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
    return {}


def save_seen(seen: dict[str, str]) -> None:
    """Persist the seen papers dict."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(seen, indent=2))


def _extract_arxiv_id_fallback(value: str) -> str:
    pattern = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|^)(\d{4}\.\d{4,5}(?:v\d+)?)")
    match = pattern.search(value.strip())
    if not match:
        raise ValueError(f"Could not extract arXiv ID from: {value}")
    return match.group(1)


def _sync_sources_if_available() -> None:
    if sync_all_sources is None:
        return

    try:
        sync_all_sources()
    except Exception as exc:
        logger.warning("Source sync failed (continuing with existing manifest): %s", exc)


def _run_via_triage_cli(arxiv_input: str, output_format: str, output_path: str | None) -> None:
    triage_bin = shutil.which("triage")
    if not triage_bin:
        raise RuntimeError(
            "triage_agent package is unavailable and `triage` CLI was not found on PATH."
        )

    cmd = [triage_bin, arxiv_input, "--format", output_format]
    if output_path:
        cmd.extend(["--output", output_path])

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)


async def _run_via_python(arxiv_input: str, output_format: str, output_path: str | None) -> None:
    from triage_agent.api.arxiv import extract_arxiv_id
    from triage_agent.formatters import render_json, render_markdown
    from triage_agent.orchestrator import run_triage

    # Extract ID and check if already seen
    try:
        arxiv_id = extract_arxiv_id(arxiv_input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    seen = load_seen()
    if arxiv_id in seen:
        print(f"Paper {arxiv_id} was already triaged on {seen[arxiv_id]}.")
        print("Re-running triage anyway...")

    memo = await run_triage(arxiv_input)

    seen[arxiv_id] = datetime.now(UTC).isoformat()
    save_seen(seen)

    output = render_json(memo) if output_format == "json" else render_markdown(memo)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        print(f"Memo written to: {path}")
    else:
        print(output)


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run_triage.py <arxiv-id-or-url> [--format json|markdown]")
        sys.exit(1)

    arxiv_input = sys.argv[1]
    output_format = "markdown"
    output_path: str | None = None

    # Simple arg parsing
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--format" and i + 1 < len(args):
            output_format = args[i + 1]
        elif arg == "--output" and i + 1 < len(args):
            output_path = args[i + 1]

    _sync_sources_if_available()

    # Try Python-package path first; fallback to triage CLI.
    try:
        await _run_via_python(arxiv_input, output_format, output_path)
        return
    except ModuleNotFoundError:
        pass

    # CLI fallback path uses a local regex extractor for seen tracking.
    try:
        arxiv_id = _extract_arxiv_id_fallback(arxiv_input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    seen = load_seen()
    if arxiv_id in seen:
        print(f"Paper {arxiv_id} was already triaged on {seen[arxiv_id]}.")
        print("Re-running triage anyway...")

    try:
        _run_via_triage_cli(arxiv_input, output_format, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    seen[arxiv_id] = datetime.now(UTC).isoformat()
    save_seen(seen)


if __name__ == "__main__":
    asyncio.run(main())
