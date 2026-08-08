from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_service.schemas import ExecutionPlan, PlannedToolCall, ToolStatus
from agent_service.tools.base import AsyncToolExecutor, ToolDefinition, ToolRegistry


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int = Field(ge=0, le=100)
    resource: str = "none"


@pytest.mark.asyncio
async def test_schema_first_tool_contract_rejects_wrong_types_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Args.model_validate({"value": "3"}, strict=True)
    with pytest.raises(ValidationError):
        Args.model_validate({"value": 3, "invented": True}, strict=True)


@pytest.mark.asyncio
async def test_tool_descriptions_explain_selection_boundaries() -> None:
    async def handler(args: Args) -> dict[str, object]:
        return {"summary": str(args.value)}

    tool = ToolDefinition(
        name="bounded_tool",
        summary="Perform one bounded operation.",
        when_to_use="A validated integer is provided.",
        when_not_to_use="The integer is missing or ambiguous.",
        args_model=Args,
        handler=handler,
        output_contract="A JSON summary.",
    )
    schema = tool.openai_schema()
    assert schema["function"]["strict"] is True
    parameters = schema["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(parameters["properties"])
    assert "USE ONLY WHEN" in schema["function"]["description"]
    assert "DO NOT USE WHEN" in schema["function"]["description"]


@pytest.mark.asyncio
async def test_independent_tools_run_concurrently_without_load_testing() -> None:
    registry = ToolRegistry()
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(args: Args) -> dict[str, object]:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return {"summary": str(args.value), "value": args.value}

    registry.register(
        ToolDefinition(
            name="work",
            summary="test",
            when_to_use="test",
            when_not_to_use="never",
            args_model=Args,
            handler=handler,
            output_contract="test",
        )
    )
    executor = AsyncToolExecutor(registry, timeout_seconds=2, max_parallel=2)
    task = asyncio.create_task(
        executor.execute_plan(
            ExecutionPlan(
                calls=[
                    PlannedToolCall(call_id="a", tool_name="work", arguments={"value": 1, "resource": "none"}),
                    PlannedToolCall(call_id="b", tool_name="work", arguments={"value": 2, "resource": "none"}),
                ]
            )
        )
    )
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    results = await task
    assert [result.status for result in results] == [ToolStatus.SUCCESS, ToolStatus.SUCCESS]


@pytest.mark.asyncio
async def test_dependencies_resolve_results_and_cycles_are_rejected() -> None:
    registry = ToolRegistry()

    async def handler(args: Args) -> dict[str, object]:
        return {"summary": str(args.value), "value": args.value + 1}

    registry.register(
        ToolDefinition(
            name="increment",
            summary="test",
            when_to_use="test",
            when_not_to_use="never",
            args_model=Args,
            handler=handler,
            output_contract="test",
        )
    )
    executor = AsyncToolExecutor(registry, timeout_seconds=2, max_parallel=2)
    results = await executor.execute_plan(
        ExecutionPlan(
            calls=[
                PlannedToolCall(call_id="a", tool_name="increment", arguments={"value": 1}),
                PlannedToolCall(
                    call_id="b",
                    tool_name="increment",
                    arguments={"value": {"$ref": "a.data.value"}},
                    depends_on=["a"],
                ),
            ]
        )
    )
    assert results[1].data["value"] == 3

    with pytest.raises(ValueError, match="cycle"):
        await executor.execute_plan(
            ExecutionPlan(
                calls=[
                    PlannedToolCall(call_id="a", tool_name="increment", arguments={"value": 1}, depends_on=["b"]),
                    PlannedToolCall(call_id="b", tool_name="increment", arguments={"value": 2}, depends_on=["a"]),
                ]
            )
        )


@pytest.mark.asyncio
async def test_shared_resource_lock_prevents_conflicting_mutation() -> None:
    registry = ToolRegistry()
    active = 0
    max_active = 0

    async def handler(args: Args) -> dict[str, object]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return {"summary": args.resource}

    registry.register(
        ToolDefinition(
            name="write",
            summary="test",
            when_to_use="test",
            when_not_to_use="never",
            args_model=Args,
            handler=handler,
            output_contract="test",
            side_effects="write",
            resource_keys=lambda args: {args.resource},
        )
    )
    executor = AsyncToolExecutor(registry, timeout_seconds=2, max_parallel=4)
    await executor.execute_plan(
        ExecutionPlan(
            calls=[
                PlannedToolCall(call_id="a", tool_name="write", arguments={"value": 1, "resource": "same"}),
                PlannedToolCall(call_id="b", tool_name="write", arguments={"value": 2, "resource": "same"}),
            ]
        )
    )
    assert max_active == 1
