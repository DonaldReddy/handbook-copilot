"""Pods — user-defined groups of documents.

A Pod is just a name. The registry is a tiny JSON file so we can list
empty pods (created but no documents yet) which Chroma alone can't tell us.
Chunks belonging to a pod carry `pod=<name>` in their metadata.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_POD = "Default"
_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "pods.json"
_LEGACY_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "parts.json"
_PDF_ROOT = Path(__file__).resolve().parents[1] / "data" / "pdfs"

# Keep names filesystem-safe — we use them as directory names too.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-.]{0,63}$")


def _load() -> list[str]:
    # One-shot migration from the old file name.
    if _LEGACY_REGISTRY.exists() and not _REGISTRY.exists():
        try:
            _REGISTRY.write_bytes(_LEGACY_REGISTRY.read_bytes())
            _LEGACY_REGISTRY.unlink()
        except OSError:
            pass

    if not _REGISTRY.exists():
        return [DEFAULT_POD]
    try:
        data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
        pods = [str(p) for p in data.get("pods", data.get("parts", []))]
    except (json.JSONDecodeError, OSError):
        pods = []
    if DEFAULT_POD not in pods:
        pods.insert(0, DEFAULT_POD)
    seen: set[str] = set()
    return [p for p in pods if not (p in seen or seen.add(p))]


def _save(pods: list[str]) -> None:
    _REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps({"pods": pods}, indent=2), encoding="utf-8")


def list_pods() -> list[str]:
    return _load()


def validate_name(name: str) -> str:
    """Return a cleaned name or raise ValueError."""
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "Pod names must be 1–64 chars, start alphanumeric, and contain "
            "only letters, digits, spaces, dashes, dots, or underscores."
        )
    return name


def create_pod(name: str) -> str:
    name = validate_name(name)
    pods = _load()
    if name not in pods:
        pods.append(name)
        _save(pods)
    return name


def delete_pod(name: str) -> None:
    """Remove from registry. Chunks and files must be deleted separately."""
    if name == DEFAULT_POD:
        raise ValueError("The Default pod cannot be deleted.")
    pods = [p for p in _load() if p != name]
    _save(pods)


def rename_pod(old: str, new: str) -> str:
    """Registry-only rename. Chunk metadata and folder must be moved separately."""
    new = validate_name(new)
    if old == DEFAULT_POD:
        raise ValueError("The Default pod cannot be renamed.")
    if old == new:
        return new
    pods = _load()
    if old not in pods:
        raise ValueError(f"Unknown pod: {old}")
    if new in pods:
        raise ValueError(f"A pod named {new!r} already exists.")
    pods = [new if p == old else p for p in pods]
    _save(pods)
    return new


def pdf_dir(pod: str) -> Path:
    """Filesystem directory where a pod's uploaded PDFs are saved."""
    d = _PDF_ROOT / pod
    d.mkdir(parents=True, exist_ok=True)
    return d
