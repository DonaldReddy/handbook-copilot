"""Reusable ingest pipeline shared by the CLI and the Streamlit uploader.

Every ingested chunk is tagged with the `part` it belongs to so the store
can filter queries and deletions by part.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .chunker import chunk_pages
from .embedder import embed_documents
from .parser import parse_pdf
from .parts import DEFAULT_PART
from .store import Store

ProgressFn = Callable[[str, int, int], None]

EMBED_BATCH = 64


@dataclass
class IngestResult:
    source: str
    part: str
    pages: int
    chunks: int
    skipped: bool = False
    error: Optional[str] = None


def ingest_pdf(
    path: Path,
    store: Store,
    *,
    part: str = DEFAULT_PART,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> IngestResult:
    """Parse → chunk → embed → store one PDF into a given part.

    Idempotent per (part, filename). Use force=True to overwrite.
    """
    src = path.name

    if not force and src in store.sources(part=part):
        return IngestResult(source=src, part=part, pages=0, chunks=0, skipped=True)

    def _p(stage: str, cur: int, total: int) -> None:
        if progress:
            progress(stage, cur, total)

    try:
        _p("parsing", 0, 1)
        pages = parse_pdf(path)
        if not pages:
            return IngestResult(source=src, part=part, pages=0, chunks=0,
                                error="no extractable text")

        _p("chunking", 0, 1)
        chunks = chunk_pages(pages)
        if not chunks:
            return IngestResult(source=src, part=part, pages=len(pages), chunks=0,
                                error="no chunks produced")
        for c in chunks:
            c.part = part

        # Overwrite any prior copy of this source in this part.
        if src in store.sources(part=part):
            store.delete_source(src, part=part)

        total = len(chunks)
        for i in range(0, total, EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            vecs = embed_documents([c.text for c in batch])
            store.add(batch, vecs)
            _p("embedding", min(i + len(batch), total), total)

        return IngestResult(source=src, part=part, pages=len(pages), chunks=len(chunks))
    except Exception as e:  # surface, don't crash the whole batch
        return IngestResult(source=src, part=part, pages=0, chunks=0, error=str(e))
