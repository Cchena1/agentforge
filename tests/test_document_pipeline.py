from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_service.document_models import (
    ParsedBlock,
    ParsedDocument,
    ParserCapabilities,
    ParseRequest,
)
from agent_service.document_pipeline import (
    DocumentNeedsReviewError,
    ParentChildChunker,
    QualityGatedDocumentParser,
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
    chunks = ParentChildChunker(target_tokens=20, max_tokens=24, overlap_tokens=2).chunk(
        document
    )

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
