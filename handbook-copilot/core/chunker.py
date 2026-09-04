"""Heading-aware chunking.

Strategy:
  1. Concatenate pages (keeping a page-offset map so we can recover page nums).
  2. Split on numbered-section headings (1., 1.1, 1.1.1, Chapter N, Section N)
     when we can find them — these are semantic boundaries in a handbook.
  3. Fall back to RecursiveCharacterTextSplitter within any section that
     exceeds the token budget.

Every chunk carries the source file, the page it started on, and the section
path it belongs to — that metadata is what makes citations meaningful.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser import Page

# Rough tokens: bge-small handles ~512 tokens; keep chunks comfortably under.
CHUNK_CHARS = 2400   # ~600 tokens
CHUNK_OVERLAP = 400  # ~100 tokens

# Matches "1.", "1.1", "1.1.2", "Chapter 3", "Section 4.2" at line start.
HEADING_RE = re.compile(
    r"^(?:"
    r"(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>[A-Z][^\n]{2,80})"
    r"|"
    r"(?P<kind>Chapter|Section|Appendix)\s+(?P<n>[\dIVX]+)[:.\s]+(?P<t>[^\n]{2,80})"
    r")$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    source: str
    page_number: int         # first page this chunk starts on
    section_path: str = ""   # e.g. "3.2 Academic Integrity" or "" if unknown
    chunk_index: int = 0     # position within the document
    part: str = ""           # user-defined group; set by the pipeline
    metadata: dict = field(default_factory=dict)


def _build_page_map(pages: list[Page]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate page texts and return (full_text, [(offset, page_number)])."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for p in pages:
        offsets.append((cursor, p.page_number))
        parts.append(p.text)
        parts.append("\n\n")
        cursor += len(p.text) + 2
    return "".join(parts), offsets


def _page_for_offset(offset: int, offsets: list[tuple[int, int]]) -> int:
    """Binary-search-lite: which page does this character offset fall on?"""
    page = offsets[0][1] if offsets else 1
    for start, pn in offsets:
        if start > offset:
            break
        page = pn
    return page


def _find_headings(full_text: str) -> list[tuple[int, str]]:
    """Return [(char_offset, section_label)] for every heading detected."""
    out: list[tuple[int, str]] = []
    for m in HEADING_RE.finditer(full_text):
        if m.group("num"):
            label = f"{m.group('num')} {m.group('title').strip()}"
        else:
            label = f"{m.group('kind')} {m.group('n')}: {m.group('t').strip()}"
        out.append((m.start(), label))
    return out


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    if not pages:
        return []

    source = pages[0].source
    full_text, offsets = _build_page_map(pages)
    headings = _find_headings(full_text)

    # Build (start, end, section_label) spans between consecutive headings.
    if headings:
        spans: list[tuple[int, int, str]] = []
        for i, (start, label) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
            spans.append((start, end, label))
        # Anything before the first heading gets an empty section label.
        if headings[0][0] > 0:
            spans.insert(0, (0, headings[0][0], ""))
    else:
        spans = [(0, len(full_text), "")]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    idx = 0
    for start, end, label in spans:
        section_text = full_text[start:end].strip()
        if not section_text:
            continue

        if len(section_text) <= CHUNK_CHARS:
            pieces = [section_text]
        else:
            pieces = splitter.split_text(section_text)

        # Best-effort: attribute each piece to the page where its first
        # characters live in the original document.
        piece_cursor = start
        for piece in pieces:
            # Locate this piece back in full_text starting from piece_cursor
            found = full_text.find(piece[:80], piece_cursor)
            if found == -1:
                found = piece_cursor
            page_num = _page_for_offset(found, offsets)
            piece_cursor = found + len(piece) - CHUNK_OVERLAP
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    page_number=page_num,
                    section_path=label,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_documents(docs: Iterable[list[Page]]) -> list[Chunk]:
    out: list[Chunk] = []
    for pages in docs:
        out.extend(chunk_pages(pages))
    return out
