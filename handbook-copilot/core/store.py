"""ChromaDB persistence, part-aware.

Every chunk carries a `part` metadata field. Queries, listings, and deletes
accept an optional `part` filter; `part=None` means "across every part".

Chunk IDs are `<part>::<source>::<chunk_index>` so the same filename can
live in two different parts without collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .parts import DEFAULT_PART

DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION = "handbook"


@dataclass
class Hit:
    text: str
    source: str
    page_number: int
    section_path: str
    part: str
    score: float


class Store:
    def __init__(self, db_dir: Path | str = DEFAULT_DB_DIR) -> None:
        db_dir = Path(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_dir), settings=Settings(anonymized_telemetry=False)
        )
        self._coll = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._migrate_legacy_chunks()

    # ---- migration ----------------------------------------------------

    def _migrate_legacy_chunks(self) -> None:
        """Backfill part='Default' for any chunks written before parts existed."""
        try:
            rows = self._coll.get(include=["metadatas"])
        except Exception:
            return
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        stale_ids: list[str] = []
        new_metas: list[dict] = []
        for i, m in zip(ids, metas):
            if not m or "part" not in m:
                stale_ids.append(i)
                new_metas.append({**(m or {}), "part": DEFAULT_PART})
        if stale_ids:
            self._coll.update(ids=stale_ids, metadatas=new_metas)

    # ---- ingest side --------------------------------------------------

    def sources(self, part: str | None = None) -> set[str]:
        """Filenames indexed in a part (or in the whole store)."""
        where = {"part": part} if part else None
        try:
            rows = self._coll.get(where=where, include=["metadatas"])
        except Exception:
            return set()
        return {m["source"] for m in (rows.get("metadatas") or []) if m and "source" in m}

    def delete_source(self, source: str, part: str | None = None) -> None:
        if part:
            self._coll.delete(where={"$and": [{"source": source}, {"part": part}]})
        else:
            self._coll.delete(where={"source": source})

    def delete_part(self, part: str) -> None:
        """Remove every chunk belonging to the given part."""
        self._coll.delete(where={"part": part})

    def rename_part(self, old: str, new: str) -> None:
        """Rewrite the `part` metadata field on every chunk in `old`."""
        rows = self._coll.get(where={"part": old}, include=["metadatas"])
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        if not ids:
            return
        # Also update the ID prefix so future upserts don't collide.
        new_ids: list[str] = []
        new_docs = self._coll.get(where={"part": old}, include=["documents", "embeddings", "metadatas"])
        for i, m in zip(ids, metas):
            _, source, idx = i.split("::", 2)
            new_ids.append(f"{new}::{source}::{idx}")
        # Chroma has no rename-id; re-add under new IDs, then delete old.
        self._coll.upsert(
            ids=new_ids,
            embeddings=new_docs.get("embeddings"),
            documents=new_docs.get("documents"),
            metadatas=[{**m, "part": new} for m in metas],
        )
        self._coll.delete(ids=ids)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [f"{c.part or DEFAULT_PART}::{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page_number": c.page_number,
                "section_path": c.section_path,
                "chunk_index": c.chunk_index,
                "part": c.part or DEFAULT_PART,
            }
            for c in chunks
        ]
        self._coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    # ---- query side ---------------------------------------------------

    def search(
        self, query_embedding: list[float], k: int = 20, part: str | None = None
    ) -> list[Hit]:
        where = {"part": part} if part else None
        res = self._coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                Hit(
                    text=doc,
                    source=meta.get("source", "?"),
                    page_number=int(meta.get("page_number", 0)),
                    section_path=meta.get("section_path", ""),
                    part=meta.get("part", DEFAULT_PART),
                    score=1.0 - float(dist),
                )
            )
        return hits

    def count(self, part: str | None = None) -> int:
        if part is None:
            return self._coll.count()
        try:
            rows = self._coll.get(where={"part": part}, include=[])
        except Exception:
            return 0
        return len(rows.get("ids") or [])
