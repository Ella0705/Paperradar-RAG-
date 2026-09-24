"""Build (or rebuild) the local Chroma index from `local_manifest.json`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from triage_agent.local_kb import LocalManifest, _default_manifest_path
from triage_agent.rag.latex_splitter import split_latex_by_section
from triage_agent.rag.paths import default_chroma_persist_path, project_root

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 800
_DEFAULT_OVERLAP = 120


def _resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (project_root / p).resolve()

def _expand_latex_docs(base_docs: list) -> list:
    """把 .tex 文档按 section 预切,并写入章节 metadata;其他文档原样通过。"""
    from langchain_core.documents import Document

    out: list = []
    for doc in base_docs:
        path = str(doc.metadata.get("path", ""))
        if Path(path).suffix.lower() != ".tex":
            out.append(doc)          # 非 LaTeX:走老路,一切照旧
            continue
        for title, body in split_latex_by_section(doc.page_content):
            out.append(
                Document(
                    page_content=f"[{title}] {body}",              # 章节名写进正文开头(参与 embedding)
                    metadata={**doc.metadata, "section": title},   # 章节名也写进行李牌(用于归因)
                )
            )
    return out

def load_documents_from_manifest(
    manifest: LocalManifest,
    root: Path,
) -> list:
    from langchain_core.documents import Document

    docs: list[Document] = []
    for fs in manifest.sources:
        path = _resolve_path(fs.path, root)
        if not path.is_file():
            logger.warning("Skipping missing file: %s", path)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        local_id = (fs.id or path.stem or path.name).strip()
        label = (fs.source or path.name).strip()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": label,
                    "path": str(path),
                    "type": (fs.type or "local").strip() or "local",
                    "local_id": local_id,
                },
            )
        )
    for p in manifest.papers:
        body = f"{p.title}\n\n{p.abstract}".strip()
        if not body:
            continue
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": "manifest",
                    "path": f"manifest:{p.id}",
                    "type": "manifest_paper",
                    "local_id": p.id,
                },
            )
        )
    return docs


def build_index(
    manifest_path: Path | None = None,
    persist_directory: Path | None = None,
    project_root_path: Path | None = None,
) -> int:
    """Chunk, embed, and persist. Returns number of chunks stored."""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required to build the embedding index.")

    mpath = manifest_path or _default_manifest_path()
    root = project_root_path or project_root()
    persist = persist_directory or default_chroma_persist_path()
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    raw = json.loads(mpath.read_text(encoding="utf-8"))
    manifest = LocalManifest.model_validate(raw)
    base_docs = load_documents_from_manifest(manifest, root)
    if not base_docs:
        logger.warning("No documents to index (missing files, or empty sources and papers).")
        return 0

    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", str(_DEFAULT_CHUNK)))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", str(_DEFAULT_OVERLAP)))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    base_docs = _expand_latex_docs(base_docs)
    chunks = splitter.split_documents(base_docs)
    if not chunks:
        logger.warning("No chunks after splitting.")
        return 0

    embeddings = OpenAIEmbeddings(
        model=os.getenv("RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)
    # Clean rebuild of the entire index.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist),
    )
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build Chroma RAG index from local manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to local_manifest.json (default: auto)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Chroma persist directory (default: local_kb/chroma_index)",
    )
    args = parser.parse_args(argv)
    try:
        n = build_index(
            manifest_path=args.manifest,
            persist_directory=args.out,
        )
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    out = default_chroma_persist_path() if args.out is None else args.out
    print(f"Indexed {n} chunks into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
