from __future__ import annotations

import os
from typing import Iterator

from .base import Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic  # deferred import so app runs without the SDK
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, messages: list[Message]) -> Iterator[str]:
        client = self._get_client()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        with client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system or None,
            messages=convo,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
