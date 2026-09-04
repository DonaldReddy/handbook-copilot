"""CLI: ingest one PDF or a directory of PDFs into the Chroma store.

Usage:
    python ingest.py data/pdfs
    python ingest.py path/to/handbook.pdf
    python ingest.py data/pdfs --reset      # wipe and rebuild

Idempotent: sources already in the store are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from core.chunker import chunk_pages
from core.embedder import embed_documents
from core.parser import parse_pdf
from core.store import DEFAULT_DB_DIR, Store


def _pdf_paths(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.pdf"))
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest PDFs into the Handbook Copilot store.")
    ap.add_argument("path", type=Path, help="PDF file or directory of PDFs")
    ap.add_argument("--reset", action="store_true", help="Wipe the vector DB before ingesting")
    ap.add_argument("--force", action="store_true", help="Re-ingest sources already present")
    args = ap.parse_args()

    if args.reset and DEFAULT_DB_DIR.exists():
        shutil.rmtree(DEFAULT_DB_DIR)

    pdfs = _pdf_paths(args.path)
    if not pdfs:
        print(f"No PDFs found at {args.path}", file=sys.stderr)
        return 1

    store = Store()
    existing = store.sources()

    for pdf in pdfs:
        if pdf.name in existing and not args.force:
            print(f"skip  {pdf.name} (already ingested; use --force to re-ingest)")
            continue

        print(f"parse {pdf.name}")
        pages = parse_pdf(pdf)
        if not pages:
            print(f"  (no text found, skipping)")
            continue

        chunks = chunk_pages(pages)
        print(f"      {len(pages)} pages → {len(chunks)} chunks")

        # Overwrite any prior copy of this source.
        if pdf.name in existing:
            store.delete_source(pdf.name)

        # Embed in batches so we don't blow memory on huge docs.
        BATCH = 64
        for i in tqdm(range(0, len(chunks), BATCH), desc="      embedding", leave=False):
            batch = chunks[i : i + BATCH]
            vecs = embed_documents([c.text for c in batch])
            store.add(batch, vecs)

    print(f"\nDone. Vector store now holds {store.count()} chunks across "
          f"{len(store.sources())} source(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
