from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_service.graph import AgentGraph
from agent_service.schemas import ChatMessage, ModelToolCall, ModelTurn
from agent_service.tools.base import AsyncToolExecutor, ToolDefinition, ToolRegistry


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class RepeatingGateway:
    online = True

    async def complete_with_tools(self, messages: Any, tools: list[dict[str, Any]]) -> ModelTurn:
        return ModelTurn(
            content="",
            tool_calls=[ModelToolCall(call_id=f"call-{len(messages)}", name="echo", arguments={"value": 1})],
            model="fake",
            route="fake",
        )


@pytest.mark.asyncio
async def test_langgraph_stops_repeated_tool_loop() -> None:
    registry = ToolRegistry()

    async def echo(args: Args) -> dict[str, object]:
        return {"summary": str(args.value), "value": args.value}

    registry.register(
        ToolDefinition(
            name="echo",
            summary="echo",
            when_to_use="test",
            when_not_to_use="never",
            args_model=Args,
            handler=echo,
            output_contract="echo",
        )
    )
    executor = AsyncToolExecutor(registry, timeout_seconds=1, max_parallel=2)
    graph = AgentGraph(RepeatingGateway(), registry, executor, max_steps=6, max_repeated_calls=1)  # type: ignore[arg-type]
    state = await graph.run([ChatMessage(role="user", content="loop")])
    assert state["handoff_required"] is True
    assert "REPEATED_TOOL_CALL_STOP" in state["warnings"]
    assert "止损" in state["final_reply"]
