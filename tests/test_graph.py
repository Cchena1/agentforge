from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_service.context_runtime import ContextBudgetManager, ToolResultSpillStore
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


class SpillGateway:
    online = True

    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict[str, Any]] = []

    async def complete_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        _ = tools
        self.calls += 1
        self.last_messages = messages
        if self.calls == 1:
            return ModelTurn(
                content="",
                tool_calls=[
                    ModelToolCall(
                        call_id="large-1", name="large", arguments={"value": 1}
                    )
                ],
                model="fake",
                route="fake",
            )
        return ModelTurn(content="done", model="fake", route="fake")


@pytest.mark.asyncio
async def test_graph_spills_oversized_tool_output_before_next_model_call(tmp_path) -> None:
    registry = ToolRegistry()

    async def large(args: Args) -> dict[str, object]:
        return {"summary": str(args.value), "content": "evidence" * 2000}

    registry.register(
        ToolDefinition(
            name="large",
            summary="large",
            when_to_use="test",
            when_not_to_use="never",
            args_model=Args,
            handler=large,
            output_contract="large result",
        )
    )
    gateway = SpillGateway()
    executor = AsyncToolExecutor(registry, timeout_seconds=1, max_parallel=2)
    graph = AgentGraph(
        gateway,  # type: ignore[arg-type]
        registry,
        executor,
        max_steps=4,
        max_repeated_calls=2,
        context_budget=ContextBudgetManager(
            max_input_tokens=2048, reserved_output_tokens=256
        ),
        tool_result_spill=ToolResultSpillStore(
            tmp_path / "tool-results", inline_token_limit=128, preview_chars=256
        ),
    )

    state = await graph.run([ChatMessage(role="user", content="run")])

    assert state["final_reply"] == "done"
    assert "TOOL_RESULT_SPILLED" in state["warnings"]
    tool_message = next(
        message for message in gateway.last_messages if message.get("role") == "tool"
    )
    tool_envelope = json.loads(tool_message["content"])
    assert tool_envelope["spilled"] is True
    assert tool_envelope["artifact"]["schema_version"] == 1
    assert len(list((tmp_path / "tool-results").glob("*.json"))) == 1
