"""CLI: ingest one PDF or a directory of PDFs into the Chroma store.

Usage:
    python ingest.py data/pdfs
    python ingest.py path/to/handbook.pdf
    python ingest.py data/pdfs --reset      # wipe and rebuild
    python ingest.py data/pdfs --force      # re-ingest existing sources

Idempotent by default — files already in the store are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from core.pipeline import ingest_pdf
from core.pods import DEFAULT_POD, create_pod, validate_name
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
    ap.add_argument("--pod", default=DEFAULT_POD,
                    help=f"Group these documents under this Pod (default: {DEFAULT_POD!r}).")
    ap.add_argument("--reset", action="store_true", help="Wipe the vector DB before ingesting")
    ap.add_argument("--force", action="store_true", help="Re-ingest sources already present")
    args = ap.parse_args()

    pod = create_pod(validate_name(args.pod))

    if args.reset and DEFAULT_DB_DIR.exists():
        shutil.rmtree(DEFAULT_DB_DIR)

    pdfs = _pdf_paths(args.path)
    if not pdfs:
        print(f"No PDFs found at {args.path}", file=sys.stderr)
        return 1

    store = Store()

    for pdf in pdfs:
        bar: tqdm | None = None

        def progress(stage: str, cur: int, total: int) -> None:
            nonlocal bar
            if stage == "embedding":
                if bar is None:
                    bar = tqdm(total=total, desc=f"      embedding", leave=False)
                bar.n = cur
                bar.refresh()

        result = ingest_pdf(pdf, store, pod=pod, force=args.force, progress=progress)
        if bar is not None:
            bar.close()

        if result.skipped:
            print(f"skip  [{result.pod}] {result.source} (already ingested; use --force to re-ingest)")
        elif result.error:
            print(f"error [{result.pod}] {result.source}: {result.error}", file=sys.stderr)
        else:
            print(f"ok    [{result.pod}] {result.source}  "
                  f"{result.pages} pages → {result.chunks} chunks")

    print(f"\nDone. Pod {pod!r} now holds {store.count(pod=pod)} chunks; "
          f"whole store: {store.count()} across {len(store.sources())} source(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
