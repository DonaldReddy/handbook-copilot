"""Embeddings via bge-small-en-v1.5.

Loaded once and cached. 384-dim, runs on CPU in a couple of ms per chunk.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models want a query prefix for asymmetric search — documents are
# embedded plain, queries get this prefix. It's a documented ~2-3% recall win.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _model()
    vec = model.encode(
        [QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
    )[0]
    return vec.tolist()
