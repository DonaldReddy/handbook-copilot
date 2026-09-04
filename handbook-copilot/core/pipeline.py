"""Reusable ingest pipeline shared by the CLI and the Streamlit uploader.

Everything below is pure Python — no Streamlit, no argparse — so it can be
driven from any surface. Callers pass an optional `progress` callback that
receives (stage, current, total) tuples for UI updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .chunker import chunk_pages
from .embedder import embed_documents
from .parser import parse_pdf
from .store import Store

ProgressFn = Callable[[str, int, int], None]

EMBED_BATCH = 64


@dataclass
class IngestResult:
    source: str
    pages: int
    chunks: int
    skipped: bool = False
    error: Optional[str] = None


def ingest_pdf(
    path: Path,
    store: Store,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> IngestResult:
    """Parse → chunk → embed → store one PDF. Idempotent unless force=True."""
    src = path.name

    if not force and src in store.sources():
        return IngestResult(source=src, pages=0, chunks=0, skipped=True)

    def _p(stage: str, cur: int, total: int) -> None:
        if progress:
            progress(stage, cur, total)

    try:
        _p("parsing", 0, 1)
        pages = parse_pdf(path)
        if not pages:
            return IngestResult(source=src, pages=0, chunks=0,
                                error="no extractable text")

        _p("chunking", 0, 1)
        chunks = chunk_pages(pages)
        if not chunks:
            return IngestResult(source=src, pages=len(pages), chunks=0,
                                error="no chunks produced")

        # Overwrite any prior copy so re-uploading a changed file is clean.
        if src in store.sources():
            store.delete_source(src)

        total = len(chunks)
        for i in range(0, total, EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            vecs = embed_documents([c.text for c in batch])
            store.add(batch, vecs)
            _p("embedding", min(i + len(batch), total), total)

        return IngestResult(source=src, pages=len(pages), chunks=len(chunks))
    except Exception as e:  # surface, don't crash the whole batch
        return IngestResult(source=src, pages=0, chunks=0, error=str(e))
