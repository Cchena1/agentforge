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
