"""Retrieve → rerank.

Vector search casts a wide net (top-k=20), a cross-encoder reranker then
keeps the top-5 for the LLM. Reranking is the single biggest quality lever
after chunking.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .embedder import embed_query
from .store import Hit, Store

RERANKER_MODEL = "BAAI/bge-reranker-base"
CANDIDATES = 20
FINAL_K = 5


@dataclass
class Retrieved:
    hits: list[Hit]


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def retrieve(store: Store, query: str, final_k: int = FINAL_K) -> Retrieved:
    q_vec = embed_query(query)
    candidates = store.search(q_vec, k=CANDIDATES)
    if not candidates:
        return Retrieved(hits=[])

    reranker = _reranker()
    pairs = [(query, h.text) for h in candidates]
    scores = reranker.predict(pairs)
    for h, s in zip(candidates, scores):
        h.score = float(s)  # overwrite with rerank score for display
    candidates.sort(key=lambda h: h.score, reverse=True)
    return Retrieved(hits=candidates[:final_k])
