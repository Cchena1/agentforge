from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

CANONICAL_DOCUMENT_SCHEMA_VERSION = 1


class ParseStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(slots=True, frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("bounding-box coordinates are inverted")

    def as_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }


@dataclass(slots=True)
class ParsedBlock:
    text: str
    page: int | None = None
    kind: str = "paragraph"
    locator: str | None = None
    block_id: str = ""
    reading_order: int = 0
    bbox: BoundingBox | None = None
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id:
            identity = f"{self.page}:{self.locator}:{self.kind}:{self.text}"
            self.block_id = hashlib.sha256(identity.encode()).hexdigest()[:24]

    @property
    def location(self) -> dict[str, Any]:
        if self.page is not None:
            return {
                "kind": "pdf",
                "page_number": self.page,
                "bbox": self.bbox.as_dict() if self.bbox else None,
                "block_id": self.block_id,
                "reading_order": self.reading_order,
            }
        if self.heading_path:
            return {
                "kind": "docx",
                "section_index": int(self.metadata.get("section_index", 0)),
                "heading_path": list(self.heading_path),
                "paragraph_index": self.metadata.get("paragraph_index"),
                "table_index": self.metadata.get("table_index"),
                "row_index": self.metadata.get("row_index"),
                "column_index": self.metadata.get("column_index"),
                "asset_id": self.metadata.get("asset_id"),
                "block_id": self.block_id,
            }
        return {"kind": "text", "locator": self.locator, "block_id": self.block_id}


@dataclass(slots=True)
class DocumentAsset:
    asset_id: str
    asset_type: str
    location: dict[str, Any]
    caption: str | None = None
    extracted_text: str | None = None
    storage_uri: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParseAttempt:
    parser: str
    started_at: datetime
    duration_ms: float
    status: ParseStatus
    quality_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    failure_code: str | None = None

    @classmethod
    def started(cls, parser: str) -> ParseAttempt:
        return cls(parser, datetime.now(UTC), 0, ParseStatus.FAILED)


@dataclass(slots=True)
class ParseQualityReport:
    score: float
    accepted: bool
    needs_ocr: bool = False
    needs_review: bool = False
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    source_name: str
    blocks: list[ParsedBlock]
    parser: str
    warnings: list[str] = field(default_factory=list)
    assets: list[DocumentAsset] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    attempts: list[ParseAttempt] = field(default_factory=list)
    quality_report: ParseQualityReport | None = None
    schema_version: int = CANONICAL_DOCUMENT_SCHEMA_VERSION


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    page: int | None
    locator: str | None
    metadata: dict[str, Any]
    parent_id: str | None = None
    location: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ParserCapabilities:
    formats: frozenset[str]
    supports_ocr: bool = False
    supports_tables: bool = False
    supports_images: bool = False
    supports_formulas: bool = False
    supports_headers_footers: bool = False


@dataclass(slots=True, frozen=True)
class ParseRequest:
    source_path: Path
    media_type: str
    tenant_id: str = "public"
    requested_features: frozenset[str] = frozenset()
    deadline: float | None = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
