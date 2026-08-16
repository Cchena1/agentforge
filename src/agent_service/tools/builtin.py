from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..documents import _read_text, resolve_workspace_file
from ..rag import RAGService
from .base import ToolDefinition, ToolRegistry


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadFileArgs(ToolArgs):
    file_path: str = Field(min_length=1, max_length=4096, description="Workspace-relative path; never invent a path.")
    start_line: int = Field(default=1, ge=1, le=10_000_000)
    max_lines: int = Field(default=300, ge=1, le=2000)


class SearchKnowledgeArgs(ToolArgs):
    query: str = Field(
        min_length=2,
        max_length=10_000,
        description=(
            "Standalone evidence query. Resolve conversational references, but preserve exact "
            "entities, versions, dates, numbers, units, quoted phrases, and negation. Never add "
            "facts that are absent from the user request or trusted memory context."
        ),
    )
    top_k: int = Field(default=6, ge=1, le=20)
    source_ids: list[str] | None = Field(default=None, max_length=100)


class EscalateToHumanArgs(ToolArgs):
    reason: str = Field(min_length=5, max_length=2000)
    suggested_action: str = Field(min_length=3, max_length=2000)
    severity: Literal["low", "medium", "high", "critical"] = "medium"


def build_registry(workspace_root: Path, rag: RAGService) -> ToolRegistry:
    registry = ToolRegistry()

    async def read_file(args: ReadFileArgs) -> dict[str, Any]:
        path = resolve_workspace_file(workspace_root, args.file_path)
        text = _read_text(path)
        lines = text.splitlines()
        start = args.start_line - 1
        selected = lines[start : start + args.max_lines]
        return {
            "summary": f"Read {len(selected)} lines from {path.name}",
            "file_path": str(path.relative_to(workspace_root)),
            "start_line": args.start_line,
            "end_line": start + len(selected),
            "content": "\n".join(selected),
            "truncated": start + len(selected) < len(lines),
        }

    async def search_knowledge(args: SearchKnowledgeArgs) -> dict[str, Any]:
        response = await rag.retrieve(args.query, args.top_k, args.source_ids)
        return {
            "summary": f"Retrieved {len(response.hits)} evidence chunks in {response.latency_ms:.1f} ms",
            "hits": [hit.model_dump(mode="json") for hit in response.hits],
            "latency_ms": response.latency_ms,
            "warnings": response.warnings,
            "degraded_retrieval": response.degraded_retrieval,
        }

    async def escalate(args: EscalateToHumanArgs) -> dict[str, Any]:
        return {
            "summary": f"Human review requested: {args.reason}",
            "handoff_required": True,
            "reason": args.reason,
            "suggested_action": args.suggested_action,
            "severity": args.severity,
        }

    registry.register(
        ToolDefinition(
            name="read_file",
            summary="Read a known text-like file inside the configured workspace with explicit line bounds.",
            when_to_use="The user supplied a concrete file path or explicitly asks to inspect that file's contents.",
            when_not_to_use="The path is unknown, the task is general knowledge, or semantic search across ingested documents is needed.",
            args_model=ReadFileArgs,
            handler=read_file,
            output_contract="JSON with file_path, line range, content, truncation flag, and summary.",
            side_effects="read",
            resource_keys=lambda args: {f"file:{args.file_path.lower()}"},
        )
    )
    registry.register(
        ToolDefinition(
            name="search_knowledge",
            summary="Search the local indexed document corpus and return ranked evidence with source citations.",
            when_to_use="The answer must be grounded in one or more previously ingested documents, especially when citations are requested.",
            when_not_to_use="The user asks to read an exact path, the corpus is not relevant, or the answer requires changing external state.",
            args_model=SearchKnowledgeArgs,
            handler=search_knowledge,
            output_contract="Ranked hits containing text, retrieval scores, and source/page/chunk citations.",
            side_effects="read",
        )
    )
    registry.register(
        ToolDefinition(
            name="escalate_to_human",
            summary="Stop autonomous execution and return an explicit human-handoff request.",
            when_to_use="Required data is missing after validation, all safe fallbacks failed, authorization is needed, or the action is high-risk/irreversible.",
            when_not_to_use="A safe retry, a validated read-only tool, or a clear user clarification can resolve the issue.",
            args_model=EscalateToHumanArgs,
            handler=escalate,
            output_contract="Handoff flag, reason, severity, and a concrete suggested human action.",
            side_effects="none",
        )
    )
    return registry
