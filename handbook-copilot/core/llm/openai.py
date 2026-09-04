from __future__ import annotations

import os
from typing import Iterator

from .base import Message


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, messages: list[Message]) -> Iterator[str]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield token
