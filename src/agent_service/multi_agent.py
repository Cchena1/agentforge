from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: str
    summary: str
    facts: list[str] = Field(default_factory=list, max_length=50)
    citations: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    artifacts: list[str] = Field(default_factory=list, max_length=50)
    warnings: list[str] = Field(default_factory=list, max_length=50)


@dataclass(slots=True, frozen=True)
class AgentContext:
    agent_name: str
    task: str
    memory_namespace: str
    shared_facts: tuple[str, ...] = ()
    private_context: dict[str, Any] = field(default_factory=dict)


AgentWorker = Callable[[AgentContext], Awaitable[AgentResult]]


class MultiAgentCoordinator:
    """Runs isolated workers concurrently and only returns compressed contracts to the parent."""

    def __init__(self, max_parallel: int = 4, per_agent_timeout: float = 90.0) -> None:
        self._semaphore = asyncio.Semaphore(max_parallel)
        self.per_agent_timeout = per_agent_timeout

    async def run(self, jobs: list[tuple[AgentContext, AgentWorker]]) -> list[AgentResult]:
        async def invoke(context: AgentContext, worker: AgentWorker) -> AgentResult:
            async with self._semaphore:
                try:
                    async with asyncio.timeout(self.per_agent_timeout):
                        result = await worker(context)
                    return _compress(result)
                except TimeoutError:
                    return AgentResult(
                        agent_name=context.agent_name,
                        summary="Agent timed out before returning a validated result.",
                        warnings=["AGENT_TIMEOUT"],
                    )
                except Exception as exc:  # noqa: BLE001 - isolation boundary
                    return AgentResult(
                        agent_name=context.agent_name,
                        summary="Agent failed before returning a validated result.",
                        warnings=[f"AGENT_FAILURE:{type(exc).__name__}:{exc}"],
                    )

        return await asyncio.gather(*(invoke(context, worker) for context, worker in jobs))


def _compress(result: AgentResult, max_summary_chars: int = 3000) -> AgentResult:
    """Drops chain-of-thought/raw transcripts; preserves facts, citations and artifact references."""
    return result.model_copy(
        update={
            "summary": result.summary[:max_summary_chars],
            "facts": result.facts[:20],
            "citations": result.citations[:20],
            "artifacts": result.artifacts[:20],
            "warnings": result.warnings[:20],
        }
    )
