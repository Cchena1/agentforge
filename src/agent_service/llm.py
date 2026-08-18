from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from contextlib import nullcontext
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

from .errors import classify_exception
from .observability import Observability
from .schemas import ChatMessage, ModelToolCall, ModelTurn, ModelUsage
from .settings import ModelRoute, Settings

logger = logging.getLogger(__name__)
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class AllModelRoutesFailed(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures))
        self.failures = failures


class ModelGateway:
    """Async model router with bounded retry, fallback, usage accounting and tracing."""

    def __init__(self, settings: Settings, observability: Observability | None = None) -> None:
        self.settings = settings
        self.observability = observability
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
            degraded = route_index > 0
            span_context = (
                self.observability.span(
                    "gen_ai.chat",
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": route.model,
                        "agentforge.model.route": route.name,
                        "agentforge.model.degraded": degraded,
                    },
                )
                if self.observability is not None
                else nullcontext(None)
            )
            try:
                with span_context as span:
                    client = self._client(route)
                    create: Any = client.chat.completions.create
                    kwargs: dict[str, Any] = {
                        "model": route.model,
                        "messages": [_message_dict(item) for item in messages],
                        "temperature": self.settings.temperature,
                    }
                    if tools:
                        kwargs.update(
                            tools=tools, tool_choice="auto", parallel_tool_calls=True
                        )
                    if self.settings.max_tokens is not None:
                        kwargs["max_tokens"] = self.settings.max_tokens

                    async def invoke(
                        create_call: Any = create,
                        request_kwargs: dict[str, Any] = kwargs,
                    ) -> Any:
                        return await create_call(**request_kwargs)

                    response = await self._retry_route(route, invoke)
                    usage = _model_usage(response, route)
                    choice = response.choices[0]
                    message = choice.message
                    calls = []
                    for call in message.tool_calls or []:
                        try:
                            arguments = json.loads(call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {"__invalid_json__": call.function.arguments}
                        calls.append(
                            ModelToolCall(
                                call_id=call.id,
                                name=call.function.name,
                                arguments=arguments,
                            )
                        )
                    turn = ModelTurn(
                        content=message.content or "",
                        tool_calls=calls,
                        model=response.model or route.model,
                        route=route.name,
                        degraded=degraded,
                        raw_finish_reason=choice.finish_reason,
                        usage=usage,
                    )
                    if span is not None:
                        span.set_attribute("gen_ai.response.model", turn.model)
                        span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                        span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
                        span.set_attribute(
                            "agentforge.model.estimated_cost_usd", usage.estimated_cost_usd
                        )
                    self._record_usage(route.name, "success", degraded, usage)
                    return turn
            except Exception as exc:  # noqa: BLE001 - normalize provider SDKs at route boundary
                category = classify_exception(exc).value
                failures.append(f"{route.name}: {type(exc).__name__}: {exc}")
                logger.warning("Model route %s failed: %s", route.name, exc)
                if self.observability is not None:
                    self.observability.record_model_usage(
                        route.name, "failure", degraded=degraded
                    )
                    self.observability.record_failure("model", category)
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
                return response_model.model_validate(
                    _extract_json_object(turn.content), strict=True
                )
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if self.observability is not None:
                    self.observability.record_failure("model_output", "schema")
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

    def _record_usage(
        self, route: str, outcome: str, degraded: bool, usage: ModelUsage
    ) -> None:
        if self.observability is None:
            return
        self.observability.record_model_usage(
            route,
            outcome,
            degraded=degraded,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_prompt_tokens=usage.cached_prompt_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
        )


_RETRYABLE = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    TimeoutError,
)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE)


def _message_dict(item: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, ChatMessage):
        return item.model_dump(exclude_none=True)
    return {key: value for key, value in item.items() if value is not None}


def _model_usage(response: Any, route: ModelRoute) -> ModelUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached_prompt_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    estimated_cost = (
        prompt_tokens * route.input_cost_per_million_tokens
        + completion_tokens * route.output_cost_per_million_tokens
    ) / 1_000_000
    return ModelUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        reasoning_tokens=reasoning_tokens,
        estimated_cost_usd=estimated_cost,
    )


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
    path_match = re.search(
        r"(?:读取|read|打开|inspect)\s*[`\"']?([^\n`\"']+\.[A-Za-z0-9]{1,8})",
        content,
        re.I,
    )
    if path_match:
        return ModelTurn(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="offline-read-1",
                    name="read_file",
                    arguments={
                        "file_path": path_match.group(1).strip(),
                        "start_line": 1,
                        "max_lines": 300,
                    },
                )
            ],
            model="offline-rule-engine",
            route="offline",
            degraded=True,
            raw_finish_reason="tool_calls",
        )
    if any(
        word in content.lower() for word in ("文档", "知识库", "引用", "source", "citation")
    ):
        return ModelTurn(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="offline-search-1",
                    name="search_knowledge",
                    arguments={"query": content, "top_k": 6, "source_ids": None},
                )
            ],
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
