from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .tools.base import ToolExecutionScope


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: str
    summary: str
    facts: list[str] = Field(default_factory=list, max_length=50)
    citations: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    artifacts: list[str] = Field(default_factory=list, max_length=50)
    warnings: list[str] = Field(default_factory=list, max_length=50)


class AgentCapabilities(BaseModel):
    """Capabilities are enforced by runtime adapters, never only described in prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    allow_write: bool = False
    allow_network: bool = False
    allow_subagent_spawn: bool = False
    max_delegation_depth: int = Field(default=1, ge=0, le=8)


@dataclass(slots=True, frozen=True)
class AgentContext:
    agent_name: str
    task: str
    memory_namespace: str
    shared_facts: tuple[str, ...] = ()
    private_context: dict[str, Any] = field(default_factory=dict)
    parent_agent_name: str | None = None
    delegation_depth: int = 0
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)

    def tool_scope(self) -> ToolExecutionScope:
        return ToolExecutionScope(
            actor_id=self.agent_name,
            allowed_tools=self.capabilities.allowed_tools,
            allow_write=self.capabilities.allow_write,
        )

    def delegate(
        self,
        *,
        agent_name: str,
        task: str,
        memory_namespace: str,
        shared_facts: tuple[str, ...] = (),
        private_context: dict[str, Any] | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> AgentContext:
        if not self.capabilities.allow_subagent_spawn:
            raise PermissionError("this agent is not allowed to spawn subagents")
        child_depth = self.delegation_depth + 1
        if child_depth > self.capabilities.max_delegation_depth:
            raise PermissionError("maximum subagent delegation depth exceeded")
        child_capabilities = capabilities or AgentCapabilities()
        if child_capabilities.max_delegation_depth > self.capabilities.max_delegation_depth:
            raise PermissionError("child delegation depth cannot exceed the parent limit")
        if not child_capabilities.allowed_tools.issubset(self.capabilities.allowed_tools):
            raise PermissionError("child tools must be a subset of parent capabilities")
        if child_capabilities.allow_write and not self.capabilities.allow_write:
            raise PermissionError("child cannot gain write capability from a read-only parent")
        if child_capabilities.allow_network and not self.capabilities.allow_network:
            raise PermissionError("child cannot gain network capability from an offline parent")
        return AgentContext(
            agent_name=agent_name,
            task=task,
            memory_namespace=memory_namespace,
            shared_facts=shared_facts,
            private_context=dict(private_context or {}),
            parent_agent_name=self.agent_name,
            delegation_depth=child_depth,
            capabilities=child_capabilities,
        )


AgentWorker = Callable[[AgentContext], Awaitable[AgentResult]]


class MultiAgentCoordinator:
    """Runs isolated workers concurrently and only returns compressed contracts to the parent."""

    def __init__(self, max_parallel: int = 4, per_agent_timeout: float = 90.0) -> None:
        self._semaphore = asyncio.Semaphore(max_parallel)
        self.per_agent_timeout = per_agent_timeout

    async def run(self, jobs: list[tuple[AgentContext, AgentWorker]]) -> list[AgentResult]:
        async def invoke(context: AgentContext, worker: AgentWorker) -> AgentResult:
            async with self._semaphore:
                if context.delegation_depth > context.capabilities.max_delegation_depth:
                    return AgentResult(
                        agent_name=context.agent_name,
                        summary="Agent delegation was rejected by the runtime capability boundary.",
                        warnings=["AGENT_DELEGATION_DEPTH_DENIED"],
                    )
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
