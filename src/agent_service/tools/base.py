from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..schemas import ExecutionPlan, PlannedToolCall, ToolError, ToolResult, ToolStatus

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ToolHandler = Callable[[ArgsT], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ToolDefinition(Generic[ArgsT]):
    name: str
    summary: str
    when_to_use: str
    when_not_to_use: str
    args_model: type[ArgsT]
    handler: ToolHandler[ArgsT]
    output_contract: str
    side_effects: Literal["none", "read", "write"] = "none"
    resource_keys: Callable[[ArgsT], set[str]] = field(default=lambda _args: set())

    def model_description(self) -> str:
        return (
            f"PURPOSE: {self.summary}\n"
            f"USE ONLY WHEN: {self.when_to_use}\n"
            f"DO NOT USE WHEN: {self.when_not_to_use}\n"
            f"SIDE EFFECTS: {self.side_effects}.\n"
            f"RETURNS: {self.output_contract}\n"
            "If required inputs are missing or ambiguous, do not guess; ask the user or choose escalate_to_human."
        )

    def openai_schema(self) -> dict[str, Any]:
        schema = _strict_json_schema(self.args_model.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.model_description(),
                "strict": True,
                "parameters": schema,
            },
        }


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic JSON Schema for OpenAI strict function tools."""
    normalized = dict(schema)
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["properties"] = {
            key: _strict_json_schema(value) if isinstance(value, dict) else value
            for key, value in properties.items()
        }
        normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    definitions = normalized.get("$defs")
    if isinstance(definitions, dict):
        normalized["$defs"] = {
            key: _strict_json_schema(value) if isinstance(value, dict) else value
            for key, value in definitions.items()
        }
    for keyword in ("anyOf", "oneOf", "allOf", "items"):
        value = normalized.get(keyword)
        if isinstance(value, list):
            normalized[keyword] = [
                _strict_json_schema(item) if isinstance(item, dict) else item for item in value
            ]
        elif isinstance(value, dict):
            normalized[keyword] = _strict_json_schema(value)
    return normalized


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition[Any]] = {}

    def register(self, tool: ToolDefinition[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition[Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def schemas(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        values = (
            self._tools.values()
            if names is None
            else (self._tools[name] for name in names if name in self._tools)
        )
        return [tool.openai_schema() for tool in values]

    @property
    def names(self) -> set[str]:
        return set(self._tools)


@dataclass(slots=True, frozen=True)
class ToolExecutionScope:
    """Runtime-enforced capabilities for one root agent or delegated worker."""

    actor_id: str = "root-agent"
    allowed_tools: frozenset[str] | None = None
    allow_write: bool = True


@dataclass(slots=True, frozen=True)
class ToolInvocationContext:
    call: PlannedToolCall
    tool: ToolDefinition[Any]
    arguments: BaseModel
    resource_keys: tuple[str, ...]
    scope: ToolExecutionScope


@dataclass(slots=True, frozen=True)
class ToolPolicyDecision:
    allowed: bool
    requires_approval: bool = False
    code: str = ""
    reason: str = ""


@dataclass(slots=True, frozen=True)
class ToolGateDecision:
    allowed: bool
    code: str = ""
    reason: str = ""


@dataclass(slots=True, frozen=True)
class ToolExecutionEvent:
    phase: Literal["policy", "approval", "guard", "execution"]
    outcome: Literal["allowed", "denied", "started", "success", "error", "timeout"]
    call_id: str
    tool_name: str
    side_effects: Literal["none", "read", "write"]
    code: str = ""
    latency_ms: float = 0.0


class ToolPolicy(Protocol):
    async def evaluate(self, context: ToolInvocationContext) -> ToolPolicyDecision: ...


class ToolApprover(Protocol):
    async def approve(self, context: ToolInvocationContext) -> ToolGateDecision: ...


class ToolGuard(Protocol):
    async def check(self, context: ToolInvocationContext) -> ToolGateDecision: ...


ToolEventSink = Callable[[ToolExecutionEvent], Awaitable[None]]


class AllowAllToolPolicy:
    """Compatibility policy used when callers do not opt into runtime restrictions."""

    async def evaluate(self, context: ToolInvocationContext) -> ToolPolicyDecision:
        _ = context
        return ToolPolicyDecision(allowed=True)


class CapabilityToolPolicy:
    """Enforces tool allowlists and turns write operations into explicit approvals."""

    def __init__(self, *, require_write_approval: bool = True) -> None:
        self.require_write_approval = require_write_approval

    async def evaluate(self, context: ToolInvocationContext) -> ToolPolicyDecision:
        allowed_tools = context.scope.allowed_tools
        if allowed_tools is not None and context.tool.name not in allowed_tools:
            return ToolPolicyDecision(
                allowed=False,
                code="TOOL_CAPABILITY_DENIED",
                reason=f"tool {context.tool.name!r} is outside the execution scope",
            )
        if context.tool.side_effects == "write" and not context.scope.allow_write:
            return ToolPolicyDecision(
                allowed=False,
                code="TOOL_WRITE_DENIED",
                reason="write side effects are disabled for this execution scope",
            )
        return ToolPolicyDecision(
            allowed=True,
            requires_approval=(
                self.require_write_approval and context.tool.side_effects == "write"
            ),
        )


class StaticToolApprover:
    """Fail-closed write approval until an interactive approver is configured."""

    def __init__(self, *, allow_write: bool = False) -> None:
        self.allow_write = allow_write

    async def approve(self, context: ToolInvocationContext) -> ToolGateDecision:
        if context.tool.side_effects != "write" or self.allow_write:
            return ToolGateDecision(allowed=True)
        return ToolGateDecision(
            allowed=False,
            code="TOOL_APPROVAL_DENIED",
            reason="write tool requires explicit approval",
        )


class AllowAllToolGuard:
    async def check(self, context: ToolInvocationContext) -> ToolGateDecision:
        _ = context
        return ToolGateDecision(allowed=True)


async def _discard_tool_event(event: ToolExecutionEvent) -> None:
    _ = event


class AsyncToolExecutor:
    """Executes a validated dependency DAG through policy, approval and guard gates."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float,
        max_parallel: int,
        policy: ToolPolicy | None = None,
        approver: ToolApprover | None = None,
        guard: ToolGuard | None = None,
        event_sink: ToolEventSink | None = None,
    ) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.policy = policy or AllowAllToolPolicy()
        self.approver = approver or StaticToolApprover(allow_write=True)
        self.guard = guard or AllowAllToolGuard()
        self.event_sink = event_sink or _discard_tool_event
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._resource_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        scope: ToolExecutionScope | None = None,
    ) -> list[ToolResult]:
        _validate_dag(plan)
        active_scope = scope or ToolExecutionScope()
        calls = {call.call_id: call for call in plan.calls}
        pending = set(calls)
        results: dict[str, ToolResult] = {}
        while pending:
            ready = [
                calls[call_id]
                for call_id in pending
                if all(dependency in results for dependency in calls[call_id].depends_on)
            ]
            if not ready:
                raise ValueError("tool dependency graph contains a cycle or unresolved dependency")
            batch_results = await asyncio.gather(
                *(self._execute_one(call, results, active_scope) for call in ready)
            )
            for result in batch_results:
                results[result.call_id] = result
                pending.remove(result.call_id)
        return [results[call.call_id] for call in plan.calls]

    async def _execute_one(
        self,
        call: PlannedToolCall,
        prior: dict[str, ToolResult],
        scope: ToolExecutionScope,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            tool = self.registry.get(call.tool_name)
        except KeyError as exc:
            return _error_result(call, started, "UNKNOWN_TOOL", str(exc), False)

        failed_dependencies = [
            dep for dep in call.depends_on if prior[dep].status is not ToolStatus.SUCCESS
        ]
        if failed_dependencies:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.SKIPPED,
                error=ToolError(
                    code="DEPENDENCY_FAILED",
                    message=f"Dependencies failed: {', '.join(failed_dependencies)}",
                    retryable=False,
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        try:
            resolved_arguments = _resolve_references(call.arguments, prior)
            args = tool.args_model.model_validate(resolved_arguments, strict=True)
            resource_keys = tuple(sorted(tool.resource_keys(args)))
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            return _error_result(call, started, "INVALID_ARGUMENTS", str(exc), False)

        context = ToolInvocationContext(
            call=call,
            tool=tool,
            arguments=args,
            resource_keys=resource_keys,
            scope=scope,
        )
        policy_failure = await self._apply_policy(context, started)
        if policy_failure is not None:
            return policy_failure
        guard_failure = await self._apply_guard(context, started)
        if guard_failure is not None:
            return guard_failure

        locks = [self._resource_locks[key] for key in resource_keys]
        await self._emit(context, "execution", "started", started=started)
        try:
            async with self._semaphore:
                for lock in locks:
                    await lock.acquire()
                try:
                    async with asyncio.timeout(self.timeout_seconds):
                        data = await tool.handler(args)
                    if not isinstance(data, dict):
                        raise TypeError("tool handler must return a JSON object")
                finally:
                    for lock in reversed(locks):
                        lock.release()
            result = ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.SUCCESS,
                data=data,
                summary=str(data.get("summary", ""))[:2000],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            await self._emit(context, "execution", "success", started=started)
            return result
        except TimeoutError:
            await self._emit(
                context,
                "execution",
                "timeout",
                code="TOOL_TIMEOUT",
                started=started,
            )
            return _error_result(
                call,
                started,
                "TOOL_TIMEOUT",
                "tool execution timed out",
                True,
                ToolStatus.TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            await self._emit(
                context,
                "execution",
                "error",
                code="TOOL_FAILURE",
                started=started,
            )
            return _error_result(call, started, "TOOL_FAILURE", str(exc), False)

    async def _apply_policy(
        self,
        context: ToolInvocationContext,
        started: float,
    ) -> ToolResult | None:
        try:
            decision = await self.policy.evaluate(context)
        except Exception as exc:  # noqa: BLE001 - policy boundary must fail closed
            await self._emit(
                context,
                "policy",
                "denied",
                code="TOOL_POLICY_FAILURE",
                started=started,
            )
            return _error_result(
                context.call,
                started,
                "TOOL_POLICY_FAILURE",
                f"tool policy failed closed: {type(exc).__name__}",
                False,
            )
        if not decision.allowed:
            code = decision.code or "TOOL_POLICY_DENIED"
            await self._emit(context, "policy", "denied", code=code, started=started)
            return _error_result(
                context.call,
                started,
                code,
                decision.reason or "tool policy denied execution",
                False,
            )
        await self._emit(context, "policy", "allowed", started=started)
        if not decision.requires_approval:
            return None
        try:
            approval = await self.approver.approve(context)
        except Exception as exc:  # noqa: BLE001 - approval boundary must fail closed
            await self._emit(
                context,
                "approval",
                "denied",
                code="TOOL_APPROVAL_FAILURE",
                started=started,
            )
            return _error_result(
                context.call,
                started,
                "TOOL_APPROVAL_FAILURE",
                f"tool approval failed closed: {type(exc).__name__}",
                False,
            )
        if not approval.allowed:
            code = approval.code or "TOOL_APPROVAL_DENIED"
            await self._emit(context, "approval", "denied", code=code, started=started)
            return _error_result(
                context.call,
                started,
                code,
                approval.reason or "tool approval denied execution",
                False,
            )
        await self._emit(context, "approval", "allowed", started=started)
        return None

    async def _apply_guard(
        self,
        context: ToolInvocationContext,
        started: float,
    ) -> ToolResult | None:
        try:
            decision = await self.guard.check(context)
        except Exception as exc:  # noqa: BLE001 - guard boundary must fail closed
            await self._emit(
                context,
                "guard",
                "denied",
                code="TOOL_GUARD_FAILURE",
                started=started,
            )
            return _error_result(
                context.call,
                started,
                "TOOL_GUARD_FAILURE",
                f"tool guard failed closed: {type(exc).__name__}",
                False,
            )
        if not decision.allowed:
            code = decision.code or "TOOL_GUARD_DENIED"
            await self._emit(context, "guard", "denied", code=code, started=started)
            return _error_result(
                context.call,
                started,
                code,
                decision.reason or "tool guard denied execution",
                False,
            )
        await self._emit(context, "guard", "allowed", started=started)
        return None

    async def _emit(
        self,
        context: ToolInvocationContext,
        phase: Literal["policy", "approval", "guard", "execution"],
        outcome: Literal["allowed", "denied", "started", "success", "error", "timeout"],
        *,
        code: str = "",
        started: float,
    ) -> None:
        event = ToolExecutionEvent(
            phase=phase,
            outcome=outcome,
            call_id=context.call.call_id,
            tool_name=context.tool.name,
            side_effects=context.tool.side_effects,
            code=code,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        try:
            await self.event_sink(event)
        except Exception:  # noqa: BLE001 - telemetry must not break tool execution
            return


def _error_result(
    call: PlannedToolCall,
    started: float,
    code: str,
    message: str,
    retryable: bool,
    status: ToolStatus = ToolStatus.ERROR,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        status=status,
        error=ToolError(code=code, message=message, retryable=retryable),
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def _validate_dag(plan: ExecutionPlan) -> None:
    ids = [call.call_id for call in plan.calls]
    if len(ids) != len(set(ids)):
        raise ValueError("tool call ids must be unique")
    known = set(ids)
    for call in plan.calls:
        missing = set(call.depends_on) - known
        if missing:
            raise ValueError(f"unknown dependencies for {call.call_id}: {sorted(missing)}")
    visiting: set[str] = set()
    visited: set[str] = set()
    graph = {call.call_id: call.depends_on for call in plan.calls}

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("tool dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _resolve_references(value: Any, results: dict[str, ToolResult]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref"} and isinstance(value["$ref"], str):
            return _lookup_reference(value["$ref"], results)
        return {key: _resolve_references(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, results) for item in value]
    return value


def _lookup_reference(reference: str, results: dict[str, ToolResult]) -> Any:
    parts = reference.split(".")
    if len(parts) < 2 or parts[0] not in results:
        raise KeyError(f"invalid result reference: {reference}")
    value: Any = results[parts[0]].model_dump(mode="json")
    for part in parts[1:]:
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"invalid result reference: {reference}")
        value = value[part]
    return value
