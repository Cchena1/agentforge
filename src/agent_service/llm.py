from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from .schemas import ChatMessage, ModelToolCall, ModelTurn
from .settings import ModelRoute, Settings

logger = logging.getLogger(__name__)
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class AllModelRoutesFailed(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures))
        self.failures = failures


class ModelGateway:
    """Async model router: retry transient failures, then fall back across configured routes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.routes = settings.model_routes()
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client(self, route: ModelRoute) -> AsyncOpenAI:
        client = self._clients.get(route.name)
        if client is None:
            client = AsyncOpenAI(
                api_key=route.api_key.get_secret_value() or "offline-placeholder",
                base_url=route.base_url,
                timeout=route.timeout_seconds,
                max_retries=0,
            )
            self._clients[route.name] = client
        return client

    @property
    def online(self) -> bool:
        return any(route.api_key.get_secret_value().strip() for route in self.routes)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        if not self.online:
            return _offline_turn(messages)
        failures: list[str] = []
        for route_index, route in enumerate(self.routes):
            if not route.api_key.get_secret_value().strip():
                continue
            try:
                client = self._client(route)
                create: Any = client.chat.completions.create
                kwargs: dict[str, Any] = {
                    "model": route.model,
                    "messages": [_message_dict(item) for item in messages],
                    "temperature": self.settings.temperature,
                }
                if tools:
                    kwargs.update(tools=tools, tool_choice="auto", parallel_tool_calls=True)
                if self.settings.max_tokens is not None:
                    kwargs["max_tokens"] = self.settings.max_tokens

                async def invoke(create_call: Any = create, request_kwargs: dict[str, Any] = kwargs) -> Any:
                    return await create_call(**request_kwargs)

                response = await self._retry_route(route, invoke)
                choice = response.choices[0]
                message = choice.message
                calls = []
                for call in message.tool_calls or []:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {"__invalid_json__": call.function.arguments}
                    calls.append(ModelToolCall(call_id=call.id, name=call.function.name, arguments=arguments))
                return ModelTurn(
                    content=message.content or "",
                    tool_calls=calls,
                    model=response.model or route.model,
                    route=route.name,
                    degraded=route_index > 0,
                    raw_finish_reason=choice.finish_reason,
                )
            except Exception as exc:  # noqa: BLE001 - normalize provider SDKs at route boundary
                failures.append(f"{route.name}: {type(exc).__name__}: {exc}")
                logger.warning("Model route %s failed: %s", route.name, exc)
        raise AllModelRoutesFailed(failures)

    async def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        response_model: type[StructuredT],
    ) -> StructuredT:
        schema = response_model.model_json_schema()
        repair_messages = list(messages)
        last_error: Exception | None = None
        for repair_attempt in range(2):
            if repair_attempt:
                repair_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON failed validation. Return only one corrected JSON object matching "
                            f"this schema exactly: {json.dumps(schema, ensure_ascii=False)}. Error: {last_error}"
                        ),
                    }
                )
            turn = await self.complete_with_tools(repair_messages, tools=[])
            try:
                return response_model.model_validate(_extract_json_object(turn.content), strict=True)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"structured model output failed validation after repair: {last_error}")

    async def _retry_route(self, route: ModelRoute, operation: Any) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(route.max_attempts),
            wait=wait_random_exponential(multiplier=0.4, max=8),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return await operation()
        raise RuntimeError("unreachable")


_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, TimeoutError)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE)


def _message_dict(item: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, ChatMessage):
        return item.model_dump(exclude_none=True)
    return {key: value for key, value in item.items() if value is not None}


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response contains no JSON object") from None
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _offline_turn(messages: Sequence[ChatMessage | dict[str, Any]]) -> ModelTurn:
    content = ""
    for message in reversed(messages):
        data = _message_dict(message)
        if data.get("role") == "user":
            content = str(data.get("content", ""))
            break
    path_match = re.search(r"(?:读取|read|打开|inspect)\s*[`\"']?([^\n`\"']+\.[A-Za-z0-9]{1,8})", content, re.I)
    if path_match:
        return ModelTurn(
            content="",
            tool_calls=[ModelToolCall(call_id="offline-read-1", name="read_file", arguments={"file_path": path_match.group(1).strip(), "start_line": 1, "max_lines": 300})],
            model="offline-rule-engine",
            route="offline",
            degraded=True,
            raw_finish_reason="tool_calls",
        )
    if any(word in content.lower() for word in ("文档", "知识库", "引用", "source", "citation")):
        return ModelTurn(
            content="",
            tool_calls=[ModelToolCall(call_id="offline-search-1", name="search_knowledge", arguments={"query": content, "top_k": 6, "source_ids": None})],
            model="offline-rule-engine",
            route="offline",
            degraded=True,
            raw_finish_reason="tool_calls",
        )
    return ModelTurn(
        content="当前未配置模型 API Key。我可以执行本地文件读取和知识库检索；通用生成任务已降级为人工确认。",
        model="offline-rule-engine",
        route="offline",
        degraded=True,
        raw_finish_reason="stop",
    )
