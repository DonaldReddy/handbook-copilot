from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(Protocol):
    name: str

    def available(self) -> bool:
        """Cheap check that this provider can be used right now (key set, server up)."""
        ...

    def generate(self, messages: list[Message]) -> Iterator[str]:
        """Stream response tokens. Callers concatenate."""
        ...
