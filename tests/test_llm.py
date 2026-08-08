from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_service.llm import ModelGateway
from agent_service.schemas import ChatMessage, ModelTurn
from agent_service.settings import ModelRoute, Settings


@pytest.mark.asyncio
async def test_retry_recovers_from_transient_timeout() -> None:
    settings = Settings(workspace_root=".")
    gateway = ModelGateway(settings)
    route = ModelRoute(name="test", model="test", api_key="key", max_attempts=3)
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("transient")
        return "ok"

    assert await gateway._retry_route(route, operation) == "ok"
    assert attempts == 3


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str
    count: int


class RepairGateway(ModelGateway):
    def __init__(self) -> None:
        super().__init__(Settings())
        self.calls = 0

    async def complete_with_tools(
        self,
        messages: Any,
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        self.calls += 1
        content = '{"answer":"bad"}' if self.calls == 1 else '```json\n{"answer":"fixed","count":2}\n```'
        return ModelTurn(content=content, model="fake", route="fake")


@pytest.mark.asyncio
async def test_structured_output_missing_field_is_repaired_and_revalidated() -> None:
    gateway = RepairGateway()
    answer = await gateway.structured([ChatMessage(role="user", content="return JSON")], StructuredAnswer)
    assert answer == StructuredAnswer(answer="fixed", count=2)
    assert gateway.calls == 2
