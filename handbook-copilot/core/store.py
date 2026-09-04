"""ChromaDB persistence.

One collection ("handbook") for everything. Each document ID is
<source>::<chunk_index> so re-ingesting a changed PDF cleanly overwrites.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunker import Chunk

DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION = "handbook"


@dataclass
class Hit:
    text: str
    source: str
    page_number: int
    section_path: str
    score: float  # cosine similarity in [-1, 1] (higher = closer)


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

    # ---- ingest side --------------------------------------------------

    def sources(self) -> set[str]:
        """Filenames already in the store."""
        try:
            rows = self._coll.get(include=["metadatas"])
        except Exception:
            return set()
        return {m["source"] for m in (rows.get("metadatas") or []) if m}

    def delete_source(self, source: str) -> None:
        self._coll.delete(where={"source": source})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page_number": c.page_number,
                "section_path": c.section_path,
                "chunk_index": c.chunk_index,
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

    def search(self, query_embedding: list[float], k: int = 20) -> list[Hit]:
        res = self._coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
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
                    # Chroma returns cosine *distance* (1 - cosine sim).
                    score=1.0 - float(dist),
                )
            )
        return hits

    def count(self) -> int:
        return self._coll.count()
