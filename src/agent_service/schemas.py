from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class BoundingBoxModel(StrictModel):
    left: float
    top: float
    right: float
    bottom: float


class PdfLocation(StrictModel):
    kind: Literal["pdf"] = "pdf"
    page_number: int = Field(ge=1)
    bbox: BoundingBoxModel | None = None
    block_id: str
    reading_order: int = Field(ge=0)


class DocxLocation(StrictModel):
    kind: Literal["docx"] = "docx"
    section_index: int = Field(default=0, ge=0)
    heading_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    asset_id: str | None = None
    block_id: str | None = None


class TextLocation(StrictModel):
    kind: Literal["text"] = "text"
    locator: str | None = None
    block_id: str | None = None


DocumentLocation = PdfLocation | DocxLocation | TextLocation

class ChatMessage(StrictModel):
    role: str = Field(min_length=1, max_length=32)
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=100_000)
    history: list[ChatMessage] | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[\w.:-]+$")
    user_id: str = Field(default="anonymous", min_length=1, max_length=128, pattern=r"^[\w.:-]+$")


class Citation(StrictModel):
    source_id: str
    source_name: str
    chunk_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    locator: str | None = None
    quote: str | None = Field(default=None, max_length=500)
    score: float | None = None
    citation_id: str | None = None
    location: DocumentLocation | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parser: str | None = None

class ToolCallRecord(StrictModel):
    call_id: str
    tool_name: str
    name: str = Field(description="Deprecated compatibility alias for tool_name; retained through 0.x.")
    arguments: dict[str, Any]
    status: Literal["success", "error", "timeout", "skipped"]
    latency_ms: float = Field(ge=0)
    error_code: str | None = None
    result: str = Field(default="", description="Deprecated preview field retained for the demo frontend.")


class ChatResponse(StrictModel):
    reply: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    session_id: str
    degraded: bool = False
    handoff_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ToolError(StrictModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    call_id: str
    tool_name: str
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    citations: list[Citation] = Field(default_factory=list)
    error: ToolError | None = None
    latency_ms: float = Field(ge=0)


class PlannedToolCall(StrictModel):
    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("depends_on")
    @classmethod
    def no_self_dependency(cls, value: list[str], info: Any) -> list[str]:
        call_id = info.data.get("call_id")
        if call_id and call_id in value:
            raise ValueError("a tool call cannot depend on itself")
        return list(dict.fromkeys(value))


class ExecutionPlan(StrictModel):
    calls: list[PlannedToolCall] = Field(default_factory=list, max_length=64)


class ModelToolCall(StrictModel):
    call_id: str
    name: str
    arguments: dict[str, Any]


class ModelTurn(StrictModel):
    content: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    model: str
    route: str
    degraded: bool = False
    raw_finish_reason: str | None = None


class DocumentIngestRequest(StrictModel):
    file_path: str = Field(min_length=1, max_length=4096)
    source_id: str | None = Field(default=None, max_length=256)
    tenant_id: str = Field(
        default="public", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    acl: list[str] = Field(default_factory=list, max_length=100)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("acl")
    @classmethod
    def normalize_acl(cls, value: list[str]) -> list[str]:
        if any(
            not item
            or len(item) > 128
            or any(character.isspace() or ord(character) < 32 for character in item)
            for item in value
        ):
            raise ValueError(
                "ACL principals must contain 1-128 non-whitespace printable characters"
            )
        return sorted(set(value))


class DocumentIngestResponse(StrictModel):
    source_id: str
    source_name: str
    chunks_created: int = Field(ge=0)
    parser: str
    warnings: list[str] = Field(default_factory=list)
    version_id: str | None = Field(
        default=None,
        description="Active immutable document version; additive in the 0.x compatibility window.",
    )
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Content identity used for idempotent ingestion.",
    )
    idempotent: bool = False


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    QUALITY_CHECK = "quality_check"
    FALLBACK = "fallback"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CANCELLED = "cancelled"


class IngestionJobResponse(StrictModel):
    job_id: str
    status: IngestionJobStatus
    stage: str
    created_at: datetime
    updated_at: datetime
    result: DocumentIngestResponse | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

class RetrievalRequest(StrictModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=6, ge=1, le=30)
    source_ids: list[str] | None = Field(default=None, max_length=100)
    tenant_id: str = Field(
        default="public", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    principals: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("principals")
    @classmethod
    def normalize_principals(cls, value: list[str]) -> list[str]:
        if any(
            not item
            or len(item) > 128
            or any(character.isspace() or ord(character) < 32 for character in item)
            for item in value
        ):
            raise ValueError(
                "principals must contain 1-128 non-whitespace printable characters"
            )
        return sorted(set(value))


class RetrievalHit(StrictModel):
    text: str
    citation: Citation
    vector_score: float
    lexical_score: float
    fused_score: float
    rerank_score: float


class RetrievalResponse(StrictModel):
    hits: list[RetrievalHit]
    latency_ms: float = Field(ge=0)
    index_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Active source-version snapshot used by retrieval.",
    )
    warnings: list[str] = Field(default_factory=list)
    degraded_retrieval: bool = False

class MemoryWrite(StrictModel):
    namespace: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20_000)
    importance: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MemoryRecord(StrictModel):
    memory_id: str
    namespace: str
    text: str
    importance: float
    metadata: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
