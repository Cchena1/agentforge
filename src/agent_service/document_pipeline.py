from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .document_models import (
    BoundingBox,
    Chunk,
    DocumentAsset,
    ParseAttempt,
    ParsedBlock,
    ParsedDocument,
    ParseQualityReport,
    ParserCapabilities,
    ParseRequest,
    ParseStatus,
)
from .documents import BuiltinDocumentParser, _text_blocks


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


class DocumentNeedsReviewError(RuntimeError):
    """Parsing exhausted bounded fallbacks without trustworthy evidence."""


class ParserBackend(Protocol):
    name: str

    @property
    def profile_id(self) -> str: ...

    @property
    def capabilities(self) -> ParserCapabilities: ...

    async def parse(self, request: ParseRequest) -> ParsedDocument: ...


class CloudDocumentParser(Protocol):
    name: str
    profile_id: str

    async def parse(self, request: ParseRequest) -> ParsedDocument: ...


_NON_INDEXABLE_KINDS = {"running_header", "running_footer", "page_number"}
_VISUAL_ASSET_TYPES = {"image", "table", "chart", "formula"}


class ParseQualityEvaluator:
    """Deterministic quality gate; parser selection is never delegated to an LLM."""

    def __init__(
        self,
        acceptance_score: float = 0.58,
        *,
        min_visual_text_chars_per_page: int = 160,
        min_table_text_chars: int = 160,
    ) -> None:
        if not 0 < acceptance_score <= 1:
            raise ValueError("acceptance_score must be in (0, 1]")
        if min_visual_text_chars_per_page < 1 or min_table_text_chars < 1:
            raise ValueError("content completeness thresholds must be positive")
        self.acceptance_score = acceptance_score
        self.min_visual_text_chars_per_page = min_visual_text_chars_per_page
        self.min_table_text_chars = min_table_text_chars

    def evaluate(self, document: ParsedDocument) -> ParseQualityReport:
        indexable_blocks = [
            block
            for block in document.blocks
            if block.kind not in _NON_INDEXABLE_KINDS and block.text.strip()
        ]
        texts = [block.text.strip() for block in indexable_blocks]
        total_chars = sum(len(text) for text in texts)
        replacement_ratio = sum(text.count("\ufffd") for text in texts) / max(total_chars, 1)
        normalized = [" ".join(text.lower().split()) for text in texts]
        counts = Counter(normalized)
        duplicate_blocks = sum(count - 1 for count in counts.values() if count > 1)
        duplicate_ratio = duplicate_blocks / max(len(normalized), 1)
        page_count = int(document.provenance.get("page_count", 0) or 0)
        empty_pages = int(document.provenance.get("empty_pages", 0) or 0)
        empty_page_ratio = empty_pages / max(page_count, 1)
        characters_per_page = total_chars / max(page_count, 1)

        visual_assets = [
            asset for asset in document.assets if asset.asset_type in _VISUAL_ASSET_TYPES
        ]
        table_block_chars = sum(
            len(block.text.strip()) for block in indexable_blocks if block.kind == "table"
        )
        table_asset_chars = sum(
            len((asset.extracted_text or "").strip())
            for asset in document.assets
            if asset.asset_type == "table"
        )
        table_text_chars = max(table_block_chars, table_asset_chars)
        has_table = table_block_chars > 0 or any(
            asset.asset_type == "table" for asset in document.assets
        )
        is_pdf = Path(document.source_name).suffix.lower() == ".pdf"
        sparse_visual_output = (
            is_pdf
            and bool(visual_assets)
            and characters_per_page < self.min_visual_text_chars_per_page
        )
        sparse_table_output = is_pdf and has_table and table_text_chars < self.min_table_text_chars

        issues: list[str] = []
        if not texts:
            issues.append("NO_INDEXABLE_TEXT")
        if replacement_ratio > 0.02:
            issues.append("HIGH_GARBLED_CHARACTER_RATIO")
        if duplicate_ratio > 0.35:
            issues.append("HIGH_DUPLICATE_BLOCK_RATIO")
        if sparse_visual_output:
            issues.append("LOW_TEXT_COVERAGE_WITH_VISUAL_ASSETS")
        if sparse_table_output:
            issues.append("SPARSE_TABLE_EXTRACTION")

        scanned_signal = any("SCANNED_PDF_SUSPECTED" in item for item in document.warnings)
        incomplete_pdf = sparse_visual_output or sparse_table_output
        needs_ocr = scanned_signal or (page_count > 0 and empty_page_ratio >= 0.5) or incomplete_pdf
        if needs_ocr:
            issues.append("OCR_REQUIRED")

        score = 1.0
        if not texts:
            score -= 0.7
        score -= min(replacement_ratio * 5, 0.3)
        score -= min(duplicate_ratio * 0.5, 0.25)
        score -= min(empty_page_ratio * 0.5, 0.45)
        if sparse_visual_output:
            score -= 0.45
        if sparse_table_output:
            score -= 0.25
        score = max(0.0, min(score, 1.0))
        accepted = score >= self.acceptance_score and not needs_ocr
        return ParseQualityReport(
            score=score,
            accepted=accepted,
            needs_ocr=needs_ocr,
            needs_review=not accepted and not needs_ocr,
            metrics={
                "blocks": len(texts),
                "characters": total_chars,
                "characters_per_page": characters_per_page,
                "replacement_ratio": replacement_ratio,
                "duplicate_ratio": duplicate_ratio,
                "page_count": page_count,
                "empty_pages": empty_pages,
                "empty_page_ratio": empty_page_ratio,
                "assets": len(document.assets),
                "visual_assets": len(visual_assets),
                "table_text_characters": table_text_chars,
                "sparse_visual_output": sparse_visual_output,
                "sparse_table_output": sparse_table_output,
            },
            issues=issues,
        )


class BuiltinParserBackend:
    name = "builtin"

    def __init__(self) -> None:
        self.parser = BuiltinDocumentParser()

    @property
    def profile_id(self) -> str:
        return self.parser.profile_id

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            formats=frozenset({".pdf", ".txt", ".md", ".rst", ".csv", ".tsv"}),
            supports_tables=True,
        )

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        legacy = await self.parser.parse(request.source_path)
        page_count = max((block.page or 0 for block in legacy.blocks), default=0)
        empty_pages = page_count if page_count and not legacy.blocks else 0
        blocks = [
            ParsedBlock(
                block.text,
                block.page,
                block.kind,
                block.locator,
                reading_order=index,
            )
            for index, block in enumerate(legacy.blocks, 1)
        ]
        return ParsedDocument(
            legacy.source_name,
            blocks,
            legacy.parser,
            list(legacy.warnings),
            provenance={"page_count": page_count, "empty_pages": empty_pages},
        )


def _docling_runtime_policy(
    *,
    platform_name: str,
    utf8_mode: bool,
    torch_compile_enabled: bool,
    msvc_available: bool,
) -> bool:
    """Return whether Docling torch compilation must be disabled for this runtime."""
    if platform_name == "win32" and not utf8_mode:
        raise RuntimeError(
            "Docling on Windows requires UTF-8 mode; restart with `python -X utf8` "
            "or set PYTHONUTF8=1 before process startup"
        )
    return platform_name == "win32" and torch_compile_enabled and not msvc_available


def _prepare_docling_runtime() -> list[str]:
    platform_name = sys.platform
    utf8_mode = bool(sys.flags.utf8_mode)
    if platform_name == "win32" and not utf8_mode:
        _docling_runtime_policy(
            platform_name=platform_name,
            utf8_mode=utf8_mode,
            torch_compile_enabled=False,
            msvc_available=False,
        )

    from docling.datamodel.settings import settings as docling_settings

    disable_compile = _docling_runtime_policy(
        platform_name=platform_name,
        utf8_mode=utf8_mode,
        torch_compile_enabled=bool(docling_settings.inference.compile_torch_models),
        msvc_available=shutil.which("cl") is not None,
    )
    if not disable_compile:
        return []
    docling_settings.inference.compile_torch_models = False
    return ["DOCLING_TORCH_COMPILE_DISABLED_NO_MSVC"]


def _docling_page_count(document: Any, blocks: list[ParsedBlock]) -> int:
    pages = getattr(document, "pages", None)
    if isinstance(pages, (dict, list, tuple)) and pages:
        return len(pages)
    return max((block.page or 0 for block in blocks), default=0)


class DoclingParserBackend:
    name = "docling"
    profile_id = f"docling-native-v3:docling={_package_version('docling')}"

    def __init__(self, converter_factory: Callable[[], Any] | None = None) -> None:
        self._converter_factory = converter_factory
        self._converter: Any | None = None
        self._runtime_warnings: list[str] = []
        self._operation_lock = asyncio.Lock()

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            formats=frozenset({".pdf", ".docx"}),
            supports_ocr=True,
            supports_tables=True,
            supports_images=True,
            supports_formulas=True,
            supports_headers_footers=True,
        )

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        async with self._operation_lock:
            return await asyncio.to_thread(self._convert, request)

    def _convert(self, request: ParseRequest) -> ParsedDocument:
        if self._converter is None:
            factory = self._converter_factory
            if factory is None:
                self._runtime_warnings = _prepare_docling_runtime()
                try:
                    from docling.document_converter import DocumentConverter
                except ImportError as exc:
                    raise RuntimeError(
                        "Docling is not installed; run `uv sync --extra documents`"
                    ) from exc
                factory = DocumentConverter
            self._converter = factory()

        result = self._converter.convert(str(request.source_path))
        document = result.document
        blocks = _docling_blocks(document, request.source_path.suffix.lower())
        warnings = list(self._runtime_warnings)
        if not blocks:
            warnings.append("DOCLING_EMPTY_STRUCTURE")
            blocks = [
                ParsedBlock(item.text, item.page, item.kind, item.locator)
                for item in _text_blocks(document.export_to_markdown())
            ]
        assets = _docling_assets(document, blocks)
        page_count = _docling_page_count(document, blocks)
        text_pages = {
            block.page
            for block in blocks
            if block.page is not None
            and block.kind not in _NON_INDEXABLE_KINDS
            and block.text.strip()
        }
        empty_pages = max(page_count - len(text_pages), 0)
        return ParsedDocument(
            request.source_path.name,
            blocks,
            "docling",
            warnings,
            assets=assets,
            provenance={
                "page_count": page_count,
                "empty_pages": empty_pages,
                "canonical_model": "docling-document",
            },
        )


class PaddleOCRParserBackend:
    """Optional OCR fallback. The heavy package is imported only when routing selects it."""

    name = "paddleocr"
    profile_id = f"paddleocr-v1:paddleocr={_package_version('paddleocr')}"

    def __init__(self, pipeline_factory: Callable[[], Any] | None = None) -> None:
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._operation_lock = asyncio.Lock()

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            formats=frozenset({".pdf"}),
            supports_ocr=True,
            supports_tables=True,
            supports_images=True,
            supports_formulas=True,
        )

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        async with self._operation_lock:
            return await asyncio.to_thread(self._convert, request)

    def _convert(self, request: ParseRequest) -> ParsedDocument:
        if self._pipeline is None:
            factory = self._pipeline_factory
            if factory is None:
                try:
                    from paddleocr import PPStructureV3  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise RuntimeError(
                        "PaddleOCR is not installed; install the reviewed OCR profile before enabling it"
                    ) from exc
                factory = PPStructureV3
            self._pipeline = factory()

        results = list(self._pipeline.predict(input=str(request.source_path)))
        blocks: list[ParsedBlock] = []
        assets: list[DocumentAsset] = []
        reading_order = 0
        for page_number, result in enumerate(results, 1):
            payload = getattr(result, "json", result)
            payload = payload() if callable(payload) else payload
            if not isinstance(payload, dict):
                payload = {"text": str(payload)}
            page_blocks, page_assets = _paddle_payload(payload, page_number, reading_order)
            reading_order += len(page_blocks)
            blocks.extend(page_blocks)
            assets.extend(page_assets)
        return ParsedDocument(
            request.source_path.name,
            blocks,
            "paddleocr",
            assets=assets,
            provenance={"page_count": len(results), "empty_pages": 0},
        )


class QualityGatedDocumentParser:
    """Bounded parser registry that returns exactly one authoritative document."""

    def __init__(
        self,
        prefer_docling: bool = True,
        *,
        enable_ocr_fallback: bool = False,
        cloud_backend: CloudDocumentParser | None = None,
        max_attempts: int = 3,
        evaluator: ParseQualityEvaluator | None = None,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.prefer_docling = prefer_docling
        self.enable_ocr_fallback = enable_ocr_fallback
        self.cloud_backend = cloud_backend
        self.max_attempts = max_attempts
        self.evaluator = evaluator or ParseQualityEvaluator()
        self.builtin = BuiltinParserBackend()
        self.docling = DoclingParserBackend()
        self.ocr = PaddleOCRParserBackend()

    @property
    def profile_id(self) -> str:
        cloud_profile = self.cloud_backend.profile_id if self.cloud_backend else "disabled"
        return (
            f"registry-v3:prefer_docling={self.prefer_docling}:ocr={self.enable_ocr_fallback}:"
            f"max_attempts={self.max_attempts}:builtin={self.builtin.profile_id}:"
            f"docling={self.docling.profile_id}:paddle={self.ocr.profile_id}:cloud={cloud_profile}"
        )

    async def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix in {".doc", ".docm"}:
            raise ValueError(f"Unsupported Word format {suffix}; convert the file to .docx")
        request = ParseRequest(path, _media_type(suffix))
        if suffix == ".docx":
            if not self.prefer_docling:
                raise RuntimeError("DOCX requires Docling; run `uv sync --extra documents`")
            return await self._attempt_sequence(request, [self.docling], require_quality=True)
        if suffix == ".pdf" and self.prefer_docling:
            backends: list[ParserBackend | CloudDocumentParser] = [self.docling]
            if self.enable_ocr_fallback:
                backends.append(self.ocr)
            backends.append(self.builtin)
            if self.cloud_backend is not None:
                backends.append(self.cloud_backend)
            selected = backends[: self.max_attempts]
            if (
                self.cloud_backend is not None
                and self.max_attempts >= 2
                and len(backends) > self.max_attempts
            ):
                selected = [*backends[: self.max_attempts - 1], self.cloud_backend]
            return await self._attempt_sequence(request, selected)
        return await self._attempt_sequence(request, [self.builtin], require_quality=False)

    async def _attempt_sequence(
        self,
        request: ParseRequest,
        backends: list[ParserBackend | CloudDocumentParser],
        *,
        require_quality: bool = True,
    ) -> ParsedDocument:
        attempts: list[ParseAttempt] = []
        failures: list[str] = []
        for backend in backends:
            started = time.perf_counter()
            started_at = datetime.now(UTC)
            try:
                document = await backend.parse(request)
                report = self.evaluator.evaluate(document)
                accepted = report.accepted or not require_quality
                attempt = ParseAttempt(
                    parser=backend.name,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status=ParseStatus.SUCCESS if accepted else ParseStatus.REJECTED,
                    quality_score=report.score,
                    warnings=list(document.warnings),
                    failure_code=None if accepted else ",".join(report.issues) or "QUALITY_GATE",
                )
                attempts.append(attempt)
                document.attempts = list(attempts)
                document.quality_report = report
                if accepted:
                    return document
                failures.append(
                    f"{backend.name}: quality={report.score:.3f} ({attempt.failure_code})"
                )
            except (ImportError, RuntimeError, TimeoutError, ValueError) as exc:
                attempts.append(
                    ParseAttempt(
                        parser=backend.name,
                        started_at=started_at,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        status=ParseStatus.FAILED,
                        failure_code=type(exc).__name__,
                        warnings=[str(exc)],
                    )
                )
                failures.append(f"{backend.name}: {type(exc).__name__}: {exc}")
        detail = "; ".join(failures) or "no parser backend was available"
        raise DocumentNeedsReviewError(
            f"document requires manual review; parser attempts exhausted: {detail}"
        )


class ParentChildChunker:
    """Token-budgeted hierarchical chunks; tables stay atomic or split by rows."""

    profile_id = "parent-child-token-v1"

    def __init__(
        self, target_tokens: int = 500, max_tokens: int = 650, overlap_tokens: int = 60
    ) -> None:
        if not (0 <= overlap_tokens < target_tokens <= max_tokens):
            raise ValueError("expected 0 <= overlap < target <= max")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        section_blocks: list[ParsedBlock] = []
        heading_path: tuple[str, ...] = ()
        for block in document.blocks:
            if block.kind in {"running_header", "running_footer", "page_number"}:
                continue
            if not block.text.strip():
                continue
            if block.kind == "heading" and section_blocks:
                chunks.extend(
                    self._section_chunks(
                        document.source_name, heading_path, section_blocks, len(chunks)
                    )
                )
                section_blocks = []
            if block.kind == "heading":
                level = int(block.metadata.get("heading_level", 1))
                heading_path = (*heading_path[: max(level - 1, 0)], block.text.strip())
                block.heading_path = heading_path
            elif not block.heading_path:
                block.heading_path = heading_path
            section_blocks.append(block)
        chunks.extend(
            self._section_chunks(document.source_name, heading_path, section_blocks, len(chunks))
        )
        return chunks

    def _section_chunks(
        self,
        source_name: str,
        heading_path: tuple[str, ...],
        blocks: list[ParsedBlock],
        start_index: int,
    ) -> list[Chunk]:
        if not blocks:
            return []
        parent_text = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
        parent_id = hashlib.sha256(
            f"parent:{source_name}:{heading_path}:{parent_text}".encode()
        ).hexdigest()[:24]
        output: list[Chunk] = []
        buffer: list[ParsedBlock] = []
        token_count = 0
        for block in blocks:
            block_tokens = _estimate_tokens(block.text)
            if block.kind.startswith("table") and block_tokens > self.max_tokens:
                if buffer:
                    output.extend(
                        self._flush_child(
                            source_name, parent_id, heading_path, buffer, start_index + len(output)
                        )
                    )
                    buffer = []
                    token_count = 0
                for table_part in _split_table_block(block, self.max_tokens):
                    output.extend(
                        self._flush_child(
                            source_name,
                            parent_id,
                            heading_path,
                            [table_part],
                            start_index + len(output),
                        )
                    )
                continue
            if buffer and token_count + block_tokens > self.target_tokens:
                output.extend(
                    self._flush_child(
                        source_name, parent_id, heading_path, buffer, start_index + len(output)
                    )
                )
                buffer = _token_tail_overlap(buffer, self.overlap_tokens)
                token_count = sum(_estimate_tokens(item.text) for item in buffer)
            if block_tokens > self.max_tokens and not block.kind.startswith("table"):
                if buffer:
                    output.extend(
                        self._flush_child(
                            source_name, parent_id, heading_path, buffer, start_index + len(output)
                        )
                    )
                    buffer = []
                    token_count = 0
                for part in _hard_split_tokens(block.text, self.max_tokens, self.overlap_tokens):
                    split = ParsedBlock(
                        part,
                        block.page,
                        block.kind,
                        block.locator,
                        reading_order=block.reading_order,
                        bbox=block.bbox,
                        heading_path=heading_path,
                        metadata=dict(block.metadata),
                    )
                    output.extend(
                        self._flush_child(
                            source_name,
                            parent_id,
                            heading_path,
                            [split],
                            start_index + len(output),
                        )
                    )
                continue
            buffer.append(block)
            token_count += block_tokens
        output.extend(
            self._flush_child(
                source_name, parent_id, heading_path, buffer, start_index + len(output)
            )
        )
        return output

    def _flush_child(
        self,
        source_name: str,
        parent_id: str,
        heading_path: tuple[str, ...],
        blocks: list[ParsedBlock],
        index: int,
    ) -> list[Chunk]:
        if not blocks:
            return []
        body = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
        if not body:
            return []
        prefix = " > ".join(heading_path)
        text = f"{prefix}\n\n{body}" if prefix and prefix not in body[: len(prefix) + 4] else body
        digest = hashlib.sha256(f"{source_name}:{parent_id}:{index}:{text}".encode()).hexdigest()[
            :24
        ]
        page = blocks[0].page if all(block.page == blocks[0].page for block in blocks) else None
        locator = (
            blocks[0].locator if len(blocks) == 1 else f"section:{parent_id}:chunk:{index + 1}"
        )
        return [
            Chunk(
                digest,
                text,
                page,
                locator,
                {
                    "block_kinds": sorted({block.kind for block in blocks}),
                    "heading_path": list(heading_path),
                    "source_block_ids": [block.block_id for block in blocks],
                    "token_count": _estimate_tokens(text),
                    "location": blocks[0].location,
                },
                parent_id=parent_id,
                location=blocks[0].location,
            )
        ]


def _docling_blocks(document: Any, suffix: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    heading_path: tuple[str, ...] = ()
    iterator = getattr(document, "iterate_items", None)
    if not callable(iterator):
        return blocks
    for reading_order, entry in enumerate(iterator(), 1):
        item, level = entry if isinstance(entry, tuple) and len(entry) == 2 else (entry, 1)
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        label = str(getattr(item, "label", "paragraph")).split(".")[-1].lower()
        kind = _normalize_label(label)
        metadata: dict[str, Any] = {"docling_label": label}
        if kind == "heading":
            heading_level = max(int(level or 1), 1)
            heading_path = (*heading_path[: heading_level - 1], text)
            metadata["heading_level"] = heading_level
        provenance = list(getattr(item, "prov", []) or [])
        page = None
        bbox = None
        if provenance:
            first = provenance[0]
            page = int(getattr(first, "page_no", 0) or 0) or None
            bbox = _docling_bbox(getattr(first, "bbox", None))
        self_ref = str(getattr(item, "self_ref", "") or "")
        locator = self_ref or (
            f"page:{page}:block:{reading_order}" if page else f"docx:block:{reading_order}"
        )
        blocks.append(
            ParsedBlock(
                text=text,
                page=page,
                kind=kind,
                locator=locator,
                reading_order=reading_order,
                bbox=bbox,
                heading_path=heading_path if suffix == ".docx" else (),
                metadata=metadata,
            )
        )
    _classify_repeating_page_artifacts(blocks)
    return blocks


def _docling_assets(document: Any, blocks: list[ParsedBlock]) -> list[DocumentAsset]:
    assets: list[DocumentAsset] = []
    for asset_type, attribute in (("table", "tables"), ("image", "pictures")):
        for index, item in enumerate(list(getattr(document, attribute, []) or []), 1):
            self_ref = str(getattr(item, "self_ref", "") or f"{asset_type}:{index}")
            related = next((block for block in blocks if block.locator == self_ref), None)
            location = related.location if related else {"kind": "unknown", "locator": self_ref}
            text = str(getattr(item, "text", "") or "").strip() or None
            assets.append(
                DocumentAsset(
                    asset_id=hashlib.sha256(self_ref.encode()).hexdigest()[:24],
                    asset_type=asset_type,
                    location=location,
                    extracted_text=text,
                )
            )
    return assets


def _docling_bbox(value: Any) -> BoundingBox | None:
    if value is None:
        return None
    for left, top, right, bottom in (("l", "t", "r", "b"), ("left", "top", "right", "bottom")):
        if all(hasattr(value, key) for key in (left, top, right, bottom)):
            try:
                return BoundingBox(
                    float(getattr(value, left)),
                    float(getattr(value, top)),
                    float(getattr(value, right)),
                    float(getattr(value, bottom)),
                )
            except (TypeError, ValueError):
                return None
    return None


def _normalize_label(label: str) -> str:
    if any(token in label for token in ("title", "heading", "section_header")):
        return "heading"
    if "table" in label:
        return "table"
    if any(token in label for token in ("picture", "figure")):
        return "image"
    if "caption" in label:
        return "caption"
    if "formula" in label:
        return "formula"
    if "list" in label:
        return "list_item"
    return "paragraph"


def _classify_repeating_page_artifacts(blocks: list[ParsedBlock]) -> None:
    page_count = len({block.page for block in blocks if block.page is not None})
    if page_count < 3:
        return
    occurrences: dict[str, set[int]] = {}
    for block in blocks:
        if block.page is None or len(block.text) > 160:
            continue
        key = " ".join(block.text.lower().split())
        occurrences.setdefault(key, set()).add(block.page)
    repeated = {key for key, pages in occurrences.items() if len(pages) / page_count >= 0.6}
    for block in blocks:
        key = " ".join(block.text.lower().split())
        if key not in repeated:
            continue
        if re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?", key):
            block.kind = "page_number"
        elif block.bbox is not None and block.bbox.top <= 0.2:
            block.kind = "running_header"
        else:
            block.kind = "running_footer"


def _paddle_payload(
    payload: dict[str, Any], page_number: int, reading_order_start: int
) -> tuple[list[ParsedBlock], list[DocumentAsset]]:
    blocks: list[ParsedBlock] = []
    assets: list[DocumentAsset] = []
    raw_blocks = payload.get("parsing_res_list") or payload.get("layout_parsing_result") or []
    if isinstance(raw_blocks, dict):
        raw_blocks = raw_blocks.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raw_blocks = []
    for offset, raw in enumerate(raw_blocks, 1):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("block_content") or raw.get("text") or "").strip()
        if not text:
            continue
        raw_type = str(raw.get("block_label") or raw.get("type") or "paragraph").lower()
        kind = _normalize_label(raw_type)
        coordinate = raw.get("block_bbox") or raw.get("bbox")
        bbox = None
        if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 4:
            try:
                bbox = BoundingBox(*(float(item) for item in coordinate[:4]))
            except (TypeError, ValueError):
                bbox = None
        order = reading_order_start + offset
        block = ParsedBlock(
            text,
            page=page_number,
            kind=kind,
            locator=f"page:{page_number}:ocr-block:{offset}",
            reading_order=order,
            bbox=bbox,
            metadata={"ocr_backend": "paddleocr", "raw_type": raw_type},
        )
        blocks.append(block)
        if kind in {"table", "image", "formula"}:
            assets.append(
                DocumentAsset(
                    asset_id=hashlib.sha256((block.locator or block.block_id).encode()).hexdigest()[
                        :24
                    ],
                    asset_type="image" if kind == "image" else kind,
                    location=block.location,
                    extracted_text=text,
                )
            )
    if not blocks:
        text = str(payload.get("markdown") or payload.get("text") or "").strip()
        for offset, legacy in enumerate(_text_blocks(text), 1):
            blocks.append(
                ParsedBlock(
                    legacy.text,
                    page=page_number,
                    kind=legacy.kind,
                    locator=f"page:{page_number}:ocr-block:{offset}",
                    reading_order=reading_order_start + offset,
                )
            )
    return blocks, assets


def _estimate_tokens(text: str) -> int:
    terms = re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text)
    punctuation = re.findall(r"[^\w\s]", text, re.UNICODE)
    return max(1, len(terms) + len(punctuation) // 4)


def _hard_split_tokens(text: str, size: int, overlap: int) -> list[str]:
    units = re.findall(r"\S+\s*", text)
    if not units:
        return []
    output: list[str] = []
    buffer: list[str] = []
    for unit in units:
        if buffer and _estimate_tokens("".join([*buffer, unit])) > size:
            output.append("".join(buffer).strip())
            tail: list[str] = []
            for previous in reversed(buffer):
                candidate = "".join(reversed([*tail, previous]))
                if _estimate_tokens(candidate) > overlap:
                    break
                tail.append(previous)
            buffer = list(reversed(tail))
        buffer.append(unit)
    if buffer:
        output.append("".join(buffer).strip())
    return [item for item in output if item]


def _token_tail_overlap(blocks: list[ParsedBlock], limit: int) -> list[ParsedBlock]:
    if not blocks or limit <= 0 or blocks[-1].kind.startswith("table"):
        return []
    last = blocks[-1]
    parts = _hard_split_tokens(last.text, limit, 0)
    if not parts:
        return []
    return [
        ParsedBlock(
            parts[-1],
            last.page,
            "overlap",
            last.locator,
            reading_order=last.reading_order,
            bbox=last.bbox,
            heading_path=last.heading_path,
        )
    ]


def _split_table_block(block: ParsedBlock, max_tokens: int) -> list[ParsedBlock]:
    rows = [row for row in block.text.splitlines() if row.strip()]
    if len(rows) <= 2:
        return [block]
    header = rows[0]
    groups: list[ParsedBlock] = []
    current = [header]
    for row in rows[1:]:
        if len(current) > 1 and _estimate_tokens("\n".join([*current, row])) > max_tokens:
            groups.append(_table_group(block, current, bool(groups)))
            current = [header]
        current.append(row)
    if len(current) > 1:
        groups.append(_table_group(block, current, bool(groups)))
    return groups or [block]


def _table_group(block: ParsedBlock, rows: list[str], repeated: bool) -> ParsedBlock:
    return ParsedBlock(
        "\n".join(rows),
        block.page,
        "table",
        block.locator,
        reading_order=block.reading_order,
        bbox=block.bbox,
        heading_path=block.heading_path,
        metadata={**block.metadata, "repeated_table_header": repeated},
    )


def _media_type(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
    }.get(suffix, "text/plain")
