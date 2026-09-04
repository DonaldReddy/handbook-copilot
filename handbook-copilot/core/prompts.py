"""Prompt templates. Kept in one place so citation rules stay consistent
across providers."""
from __future__ import annotations

from .store import Hit

SYSTEM = """You are Handbook Copilot, a careful assistant that answers questions strictly from provided document excerpts.

Rules:
- Use ONLY the information in the excerpts below. If the answer is not there, say so plainly.
- Cite every factual claim inline using the format [source, p.N, §section] — use the exact source filename, page number, and section shown above each excerpt.
- Prefer short, direct answers. Quote sparingly.
- If excerpts conflict, note the conflict rather than guessing.
"""


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks with the metadata the model needs for citations."""
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        header = f"[Excerpt {i}] part={h.part} | source={h.source} | p.{h.page_number}"
        if h.section_path:
            header += f" | §{h.section_path}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    if not hits:
        return (
            f"Question: {question}\n\n"
            "No excerpts were retrieved. Reply that you cannot answer from the handbook."
        )
    return (
        f"Question: {question}\n\n"
        f"Excerpts:\n{format_context(hits)}\n\n"
        "Answer the question using only these excerpts, with inline citations."
    )
