from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any
from urllib import error, request


def _normalize_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        return "https://api.openai.com/v1"
    if value.endswith("/v1"):
        return value
    return value + "/v1"


@dataclass
class OpenAIChatConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout: int = 60

    def normalized_base_url(self) -> str:
        return _normalize_base_url(self.base_url)

    def chat_url(self) -> str:
        return f"{self.normalized_base_url()}/chat/completions"


class OpenAICompatibleChatClient:
    def __init__(self, config: OpenAIChatConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.config.api_key.strip():
            raise ValueError("API key is required for model chat")
        if not messages:
            raise ValueError("messages must contain at least one message")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.config.chat_url(),
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout, context=ssl.create_default_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                response_payload = json.loads(raw)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        choices = response_payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content", "")
        return {
            "reply": content,
            "model": response_payload.get("model", self.config.model),
            "usage": response_payload.get("usage", {}),
            "raw": response_payload,
        }
