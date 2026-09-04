from __future__ import annotations

import json
import os
from typing import Iterator

import requests

from .base import Message


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=1.5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": 0.2},
        }
        with requests.post(
            f"{self.host}/api/chat", json=payload, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    yield token
                if obj.get("done"):
                    break
