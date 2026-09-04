"""PDF → structured text with page numbers.

Uses PyMuPDF (fitz) which preserves layout well enough for handbook-style docs
and gives us page numbers cheaply (critical for citations).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Page:
    source: str          # filename (not full path — used in citations)
    page_number: int     # 1-indexed, as shown to a human reader
    text: str


def file_hash(path: Path) -> str:
    """SHA1 of file bytes — used to skip re-ingesting unchanged PDFs."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pdf(path: Path) -> list[Page]:
    """Extract one Page per PDF page. Empty pages are skipped."""
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(source=path.name, page_number=i, text=text))
    return pages
