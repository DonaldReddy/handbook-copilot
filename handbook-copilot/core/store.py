"""ChromaDB persistence, pod-aware.

Every chunk carries a `pod` metadata field. Queries, listings, and deletes
accept an optional `pod` filter; `pod=None` means "across every pod".

Chunk IDs are `<pod>::<source>::<chunk_index>` so the same filename can
live in two different pods without collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .pods import DEFAULT_POD

DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION = "handbook"


@dataclass
class Hit:
    text: str
    source: str
    page_number: int
    section_path: str
    pod: str
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
        """Ensure every chunk has a `pod` metadata field.

        Handles two upgrade paths:
          - No group metadata at all → set pod='Default'.
          - Old `part` field from the earlier naming → rename to `pod`.
        """
        try:
            rows = self._coll.get(include=["metadatas"])
        except Exception:
            return
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        stale_ids: list[str] = []
        new_metas: list[dict] = []
        for i, m in zip(ids, metas):
            m = dict(m or {})
            if "pod" in m:
                continue
            if "part" in m:
                m["pod"] = m.pop("part")
            else:
                m["pod"] = DEFAULT_POD
            stale_ids.append(i)
            new_metas.append(m)
        if stale_ids:
            self._coll.update(ids=stale_ids, metadatas=new_metas)

    # ---- ingest side --------------------------------------------------

    def sources(self, pod: str | None = None) -> set[str]:
        """Filenames indexed in a pod (or in the whole store)."""
        where = {"pod": pod} if pod else None
        try:
            rows = self._coll.get(where=where, include=["metadatas"])
        except Exception:
            return set()
        return {m["source"] for m in (rows.get("metadatas") or []) if m and "source" in m}

    def delete_source(self, source: str, pod: str | None = None) -> None:
        if pod:
            self._coll.delete(where={"$and": [{"source": source}, {"pod": pod}]})
        else:
            self._coll.delete(where={"source": source})

    def delete_pod(self, pod: str) -> None:
        """Remove every chunk belonging to the given pod."""
        self._coll.delete(where={"pod": pod})

    def rename_pod(self, old: str, new: str) -> None:
        """Rewrite the `pod` metadata field on every chunk in `old`.

        Chunk IDs also carry the pod name as a prefix, so we re-upsert
        under new IDs and delete the old ones (Chroma has no rename-id).
        """
        if old == new:
            return
        rows = self._coll.get(
            where={"pod": old}, include=["documents", "embeddings", "metadatas"]
        )
        ids = rows.get("ids") or []
        if not ids:
            return
        metas = rows.get("metadatas") or []
        docs = rows.get("documents") or []
        embeds = rows.get("embeddings") or []
        new_ids: list[str] = []
        new_metas: list[dict] = []
        for i, m in zip(ids, metas):
            _, source, idx = i.split("::", 2)
            new_ids.append(f"{new}::{source}::{idx}")
            new_metas.append({**(m or {}), "pod": new})
        self._coll.upsert(
            ids=new_ids, embeddings=embeds, documents=docs, metadatas=new_metas
        )
        self._coll.delete(ids=ids)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [f"{c.pod or DEFAULT_POD}::{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page_number": c.page_number,
                "section_path": c.section_path,
                "chunk_index": c.chunk_index,
                "pod": c.pod or DEFAULT_POD,
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
        self, query_embedding: list[float], k: int = 20, pod: str | None = None
    ) -> list[Hit]:
        where = {"pod": pod} if pod else None
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
                    pod=meta.get("pod", DEFAULT_POD),
                    score=1.0 - float(dist),
                )
            )
        return hits

    def count(self, pod: str | None = None) -> int:
        if pod is None:
            return self._coll.count()
        try:
            rows = self._coll.get(where={"pod": pod}, include=[])
        except Exception:
            return 0
        return len(rows.get("ids") or [])
