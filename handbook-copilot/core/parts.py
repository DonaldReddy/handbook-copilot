"""Parts — user-defined groups of documents.

A Part is just a name. The registry is a tiny JSON file so we can list
empty parts (created but no documents yet) which Chroma alone can't tell us.
Chunks belonging to a part carry `part=<name>` in their metadata.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_PART = "Default"
_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "parts.json"
_PDF_ROOT = Path(__file__).resolve().parents[1] / "data" / "pdfs"

# Keep names filesystem-safe — we use them as directory names too.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-.]{0,63}$")


def _load() -> list[str]:
    if not _REGISTRY.exists():
        return [DEFAULT_PART]
    try:
        data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
        parts = [str(p) for p in data.get("parts", [])]
    except (json.JSONDecodeError, OSError):
        parts = []
    if DEFAULT_PART not in parts:
        parts.insert(0, DEFAULT_PART)
    # Preserve order but dedupe.
    seen: set[str] = set()
    return [p for p in parts if not (p in seen or seen.add(p))]


def _save(parts: list[str]) -> None:
    _REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps({"parts": parts}, indent=2), encoding="utf-8")


def list_parts() -> list[str]:
    return _load()


def validate_name(name: str) -> str:
    """Return a cleaned name or raise ValueError."""
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "Part names must be 1–64 chars, start alphanumeric, and contain "
            "only letters, digits, spaces, dashes, dots, or underscores."
        )
    return name


def create_part(name: str) -> str:
    name = validate_name(name)
    parts = _load()
    if name not in parts:
        parts.append(name)
        _save(parts)
    return name


def delete_part(name: str) -> None:
    """Remove from registry. Chunks and files must be deleted separately."""
    if name == DEFAULT_PART:
        raise ValueError("The Default part cannot be deleted.")
    parts = [p for p in _load() if p != name]
    _save(parts)


def rename_part(old: str, new: str) -> str:
    """Registry-only rename. Chunk metadata and folder must be moved separately."""
    new = validate_name(new)
    if old == DEFAULT_PART:
        raise ValueError("The Default part cannot be renamed.")
    parts = _load()
    if old not in parts:
        raise ValueError(f"Unknown part: {old}")
    if new in parts and new != old:
        raise ValueError(f"A part named {new!r} already exists.")
    parts = [new if p == old else p for p in parts]
    _save(parts)
    return new


def pdf_dir(part: str) -> Path:
    """Filesystem directory where a part's uploaded PDFs are saved."""
    d = _PDF_ROOT / part
    d.mkdir(parents=True, exist_ok=True)
    return d
