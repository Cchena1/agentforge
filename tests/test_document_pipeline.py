from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_service.document_models import (
    DocumentAsset,
    ParsedBlock,
    ParsedDocument,
    ParserCapabilities,
    ParseRequest,
)
from agent_service.document_pipeline import (
    DoclingParserBackend,
    DocumentNeedsReviewError,
    ParentChildChunker,
    ParseQualityEvaluator,
    QualityGatedDocumentParser,
    _docling_runtime_policy,
)
from agent_service.schemas import Citation


@dataclass
class FakeBackend:
    name: str
    document: ParsedDocument | None = None
    error: Exception | None = None
    calls: int = 0

    @property
    def profile_id(self) -> str:
        return f"fake:{self.name}"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(frozenset({".pdf"}), supports_ocr=self.name == "ocr")

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.document is not None
        return self.document


def _document(parser: str, text: str, *, scanned: bool = False) -> ParsedDocument:
    return ParsedDocument(
        source_name="fixture.pdf",
        blocks=[
            ParsedBlock(
                text=text,
                page=1,
                locator=f"page:1:{parser}",
                reading_order=1,
            )
        ],
        parser=parser,
        warnings=["SCANNED_PDF_SUSPECTED"] if scanned else [],
        provenance={"page_count": 1, "empty_pages": 1 if scanned else 0},
    )


@pytest.mark.asyncio
async def test_quality_gate_uses_one_authoritative_fallback_and_records_attempts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"fixture")
    primary = FakeBackend("docling", _document("docling", "", scanned=True))
    fallback = FakeBackend("ocr", _document("paddleocr", "verified OCR evidence"))
    emergency = FakeBackend("builtin", error=AssertionError("must not run after acceptance"))
    parser = QualityGatedDocumentParser(enable_ocr_fallback=True, max_attempts=3)
    parser.docling = primary  # type: ignore[assignment]
    parser.ocr = fallback  # type: ignore[assignment]
    parser.builtin = emergency  # type: ignore[assignment]

    parsed = await parser.parse(source)

    assert parsed.parser == "paddleocr"
    assert [attempt.parser for attempt in parsed.attempts] == ["docling", "ocr"]
    assert primary.calls == 1
    assert fallback.calls == 1
    assert emergency.calls == 0
    assert parsed.quality_report is not None and parsed.quality_report.accepted


@pytest.mark.asyncio
async def test_parser_attempts_are_bounded_and_low_quality_content_is_not_indexable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"fixture")
    first = FakeBackend("docling", error=RuntimeError("decoder failed"))
    second = FakeBackend("ocr", _document("paddleocr", "", scanned=True))
    third = FakeBackend("builtin", _document("pypdf", "should not be attempted"))
    parser = QualityGatedDocumentParser(enable_ocr_fallback=True, max_attempts=2)
    parser.docling = first  # type: ignore[assignment]
    parser.ocr = second  # type: ignore[assignment]
    parser.builtin = third  # type: ignore[assignment]

    with pytest.raises(DocumentNeedsReviewError, match="manual review"):
        await parser.parse(source)

    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".doc", ".docm"])
async def test_unsafe_or_legacy_word_formats_are_explicitly_rejected(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"legacy{suffix}"
    source.write_bytes(b"fixture")
    parser = QualityGatedDocumentParser()

    with pytest.raises(ValueError, match=r"convert the file to \.docx"):
        await parser.parse(source)


def test_parent_child_chunking_excludes_page_artifacts_and_preserves_table_headers() -> None:
    table = "name | value\n" + "\n".join(f"row-{index} | value-{index}" for index in range(30))
    document = ParsedDocument(
        source_name="report.pdf",
        parser="fixture",
        blocks=[
            ParsedBlock("Confidential report", page=1, kind="running_header"),
            ParsedBlock("Results", page=1, kind="heading", metadata={"heading_level": 1}),
            ParsedBlock(table, page=1, kind="table", locator="page:1:table:1"),
            ParsedBlock("1", page=1, kind="page_number"),
        ],
        provenance={"page_count": 1, "empty_pages": 0},
    )
    chunks = ParentChildChunker(target_tokens=20, max_tokens=24, overlap_tokens=2).chunk(document)

    assert len(chunks) > 1
    assert all("Confidential report" not in chunk.text for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if "row-" in chunk.text]
    assert table_chunks
    assert all("name | value" in chunk.text for chunk in table_chunks)
    assert all(chunk.parent_id for chunk in chunks)
    assert all(chunk.location for chunk in chunks)


def test_citation_schema_rejects_missing_or_wrong_typed_pdf_provenance() -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate(
            {
                "source_id": "source",
                "source_name": "report.pdf",
                "chunk_id": "chunk",
                "location": {
                    "kind": "pdf",
                    "page_number": "1",
                    "block_id": "block",
                    "reading_order": 1,
                },
            }
        )


@pytest.mark.asyncio
async def test_docling_converter_is_reused_across_documents(tmp_path: Path) -> None:
    factory_calls = 0

    class FakeDocument:
        def iterate_items(self):
            item = SimpleNamespace(
                text="reusable parser evidence",
                label="paragraph",
                prov=[],
                self_ref="docling:block:1",
            )
            return [(item, 1)]

    class FakeConverter:
        def convert(self, source_path: str):
            assert source_path.endswith(".pdf")
            return SimpleNamespace(document=FakeDocument())

    def factory() -> FakeConverter:
        nonlocal factory_calls
        factory_calls += 1
        return FakeConverter()

    backend = DoclingParserBackend(converter_factory=factory)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    await backend.parse(ParseRequest(first, "application/pdf"))
    await backend.parse(ParseRequest(second, "application/pdf"))

    assert factory_calls == 1


@pytest.mark.asyncio
async def test_cloud_fallback_remains_reachable_with_three_attempt_budget(tmp_path: Path) -> None:
    source = tmp_path / "cloud-required.pdf"
    source.write_bytes(b"fixture")
    primary = FakeBackend("docling", error=RuntimeError("primary unavailable"))
    ocr = FakeBackend("ocr", error=RuntimeError("ocr unavailable"))
    builtin = FakeBackend("builtin", error=AssertionError("local overflow must be skipped"))
    cloud = FakeBackend("cloud", _document("cloud", "verified cloud evidence"))
    parser = QualityGatedDocumentParser(
        enable_ocr_fallback=True,
        cloud_backend=cloud,
        max_attempts=3,
    )
    parser.docling = primary  # type: ignore[assignment]
    parser.ocr = ocr  # type: ignore[assignment]
    parser.builtin = builtin  # type: ignore[assignment]

    parsed = await parser.parse(source)

    assert parsed.parser == "cloud"
    assert [attempt.parser for attempt in parsed.attempts] == ["docling", "ocr", "cloud"]
    assert builtin.calls == 0
    assert cloud.calls == 1


def test_quality_gate_rejects_sparse_visual_and_table_extraction() -> None:
    document = ParsedDocument(
        source_name="sparse-table.pdf",
        parser="docling",
        blocks=[
            ParsedBlock(
                "column | value\nA | 1",
                page=1,
                kind="table",
                locator="page:1:table:1",
            )
        ],
        assets=[
            DocumentAsset(
                asset_id="table-1",
                asset_type="table",
                location={"kind": "pdf", "page_number": 1},
                extracted_text="column | value\nA | 1",
            )
        ],
        provenance={"page_count": 1, "empty_pages": 0},
    )

    report = ParseQualityEvaluator().evaluate(document)

    assert not report.accepted
    assert report.needs_ocr
    assert "LOW_TEXT_COVERAGE_WITH_VISUAL_ASSETS" in report.issues
    assert "SPARSE_TABLE_EXTRACTION" in report.issues
    assert "OCR_REQUIRED" in report.issues
    assert report.metrics["characters_per_page"] < 160
    assert report.metrics["sparse_visual_output"] is True
    assert report.metrics["sparse_table_output"] is True


def test_quality_gate_excludes_headers_footers_and_page_numbers_from_coverage() -> None:
    document = ParsedDocument(
        source_name="artifact-only.pdf",
        parser="docling",
        blocks=[
            ParsedBlock("Confidential report" * 20, page=1, kind="running_header"),
            ParsedBlock("1", page=1, kind="page_number"),
        ],
        assets=[
            DocumentAsset(
                asset_id="page-image",
                asset_type="image",
                location={"kind": "pdf", "page_number": 1},
            )
        ],
        provenance={"page_count": 1, "empty_pages": 1},
    )

    report = ParseQualityEvaluator().evaluate(document)

    assert not report.accepted
    assert report.needs_ocr
    assert report.metrics["characters"] == 0
    assert "NO_INDEXABLE_TEXT" in report.issues


@pytest.mark.asyncio
async def test_sparse_docling_output_routes_to_ocr_fallback(tmp_path: Path) -> None:
    source = tmp_path / "sparse-table.pdf"
    source.write_bytes(b"fixture")
    sparse = _document("docling", "A | 1")
    sparse.assets.append(
        DocumentAsset(
            asset_id="table-1",
            asset_type="table",
            location={"kind": "pdf", "page_number": 1},
            extracted_text="A | 1",
        )
    )
    recovered_text = "Verified OCR evidence with complete table rows. " * 8
    primary = FakeBackend("docling", sparse)
    fallback = FakeBackend("ocr", _document("paddleocr", recovered_text))
    emergency = FakeBackend("builtin", error=AssertionError("must not run after OCR acceptance"))
    parser = QualityGatedDocumentParser(enable_ocr_fallback=True, max_attempts=3)
    parser.docling = primary  # type: ignore[assignment]
    parser.ocr = fallback  # type: ignore[assignment]
    parser.builtin = emergency  # type: ignore[assignment]

    parsed = await parser.parse(source)

    assert parsed.parser == "paddleocr"
    assert [attempt.status.value for attempt in parsed.attempts] == ["rejected", "success"]
    assert parsed.attempts[0].failure_code is not None
    assert "SPARSE_TABLE_EXTRACTION" in parsed.attempts[0].failure_code
    assert fallback.calls == 1
    assert emergency.calls == 0


def test_docling_runtime_policy_fails_fast_without_windows_utf8() -> None:
    with pytest.raises(RuntimeError, match=r"python -X utf8"):
        _docling_runtime_policy(
            platform_name="win32",
            utf8_mode=False,
            torch_compile_enabled=True,
            msvc_available=False,
        )


def test_docling_runtime_policy_disables_compile_only_when_msvc_is_missing() -> None:
    assert _docling_runtime_policy(
        platform_name="win32",
        utf8_mode=True,
        torch_compile_enabled=True,
        msvc_available=False,
    )
    assert not _docling_runtime_policy(
        platform_name="win32",
        utf8_mode=True,
        torch_compile_enabled=True,
        msvc_available=True,
    )
    assert not _docling_runtime_policy(
        platform_name="linux",
        utf8_mode=False,
        torch_compile_enabled=True,
        msvc_available=False,
    )


@pytest.mark.asyncio
async def test_docling_uses_document_page_inventory_for_empty_page_count(tmp_path: Path) -> None:
    class FakeDocument:
        def __init__(self) -> None:
            self.pages = {1: object(), 2: object()}

        def iterate_items(self):
            provenance = SimpleNamespace(page_no=1, bbox=None)
            item = SimpleNamespace(
                text="Only the first page contains extracted evidence.",
                label="paragraph",
                prov=[provenance],
                self_ref="docling:block:1",
            )
            return [(item, 1)]

    class FakeConverter:
        def convert(self, source_path: str):
            return SimpleNamespace(document=FakeDocument())

    source = tmp_path / "partial.pdf"
    source.write_bytes(b"fixture")
    parsed = await DoclingParserBackend(converter_factory=FakeConverter).parse(
        ParseRequest(source, "application/pdf")
    )

    assert parsed.provenance["page_count"] == 2
    assert parsed.provenance["empty_pages"] == 1
    report = ParseQualityEvaluator().evaluate(parsed)
    assert not report.accepted
    assert report.needs_ocr
    assert "OCR_REQUIRED" in report.issues
