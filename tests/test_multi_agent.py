from __future__ import annotations

import asyncio

import pytest

from agent_service.multi_agent import AgentContext, AgentResult, MultiAgentCoordinator


@pytest.mark.asyncio
async def test_multi_agent_context_isolated_and_result_compressed() -> None:
    seen: dict[str, str] = {}
    barrier = asyncio.Event()
    entered = 0

    async def worker(context: AgentContext) -> AgentResult:
        nonlocal entered
        seen[context.agent_name] = context.memory_namespace
        entered += 1
        if entered == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=1)
        return AgentResult(
            agent_name=context.agent_name,
            summary="x" * 5000,
            facts=[str(index) for index in range(40)],
            citations=[{"id": index} for index in range(40)],
        )

    coordinator = MultiAgentCoordinator(max_parallel=2)
    results = await coordinator.run(
        [
            (AgentContext("research", "task", "agent:research"), worker),
            (AgentContext("review", "task", "agent:review"), worker),
        ]
    )
    assert seen["research"] != seen["review"]
    assert all(len(result.summary) == 3000 for result in results)
    assert all(len(result.facts) == 20 for result in results)
    assert all(len(result.citations) == 20 for result in results)


def test_subagent_capabilities_cannot_escalate_from_parent() -> None:
    from agent_service.multi_agent import AgentCapabilities

    parent = AgentContext(
        "parent",
        "coordinate",
        "agent:parent",
        capabilities=AgentCapabilities(
            allowed_tools=frozenset({"search_knowledge", "read_file"}),
            allow_subagent_spawn=True,
            max_delegation_depth=2,
        ),
    )
    child = parent.delegate(
        agent_name="research",
        task="retrieve evidence",
        memory_namespace="agent:research",
        capabilities=AgentCapabilities(
            allowed_tools=frozenset({"search_knowledge"}),
            max_delegation_depth=2,
        ),
    )

    assert child.parent_agent_name == "parent"
    assert child.delegation_depth == 1
    assert child.tool_scope().allowed_tools == frozenset({"search_knowledge"})
    assert child.tool_scope().allow_write is False

    with pytest.raises(PermissionError, match="subset"):
        parent.delegate(
            agent_name="unsafe",
            task="write",
            memory_namespace="agent:unsafe",
            capabilities=AgentCapabilities(
                allowed_tools=frozenset({"shell"}),
                max_delegation_depth=2,
            ),
        )


def test_subagent_spawn_is_denied_by_default() -> None:
    parent = AgentContext("parent", "coordinate", "agent:parent")
    with pytest.raises(PermissionError, match="not allowed"):
        parent.delegate(
            agent_name="child",
            task="work",
            memory_namespace="agent:child",
        )
