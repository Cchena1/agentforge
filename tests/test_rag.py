from __future__ import annotations

import pytest
from pypdf import PdfWriter

from agent_service.documents import BuiltinDocumentParser, CompositeDocumentParser, SemanticChunker
from agent_service.embeddings import HashEmbedding
from agent_service.rag import RAGService
from agent_service.schemas import DocumentIngestRequest
from agent_service.vector_store import SQLiteVectorStore


@pytest.mark.asyncio
async def test_rag_semantic_chunks_and_returns_citations(tmp_path) -> None:
    document = tmp_path / "guide.md"
    document.write_text(
        "# Retry policy\n\nTransient timeouts use exponential backoff.\n\n"
        "# Fallback\n\nAfter retries, route to a backup model and then request human review.",
        encoding="utf-8",
    )
    embeddings = HashEmbedding()
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings)
    rag = RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        store,
    )
    ingested = await rag.ingest(DocumentIngestRequest(file_path="guide.md"))
    response = await rag.retrieve("backup model after transient timeout", top_k=3)
    assert ingested.chunks_created >= 2
    assert response.hits
    assert response.hits[0].citation.source_name == "guide.md"
    assert response.hits[0].citation.chunk_id
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_table_rows_are_preserved_as_semantic_blocks(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("name,value,unit\nlatency,12,ms\nrecall,0.91,ratio\n", encoding="utf-8")
    parsed = await BuiltinDocumentParser().parse(path)
    assert parsed.parser == "builtin-table"
    assert parsed.blocks[0].kind == "table_row"
    assert "name=latency" in parsed.blocks[0].text
    assert parsed.blocks[0].locator == "row:2"


@pytest.mark.asyncio
async def test_scanned_pdf_is_detected_and_requests_ocr_fallback(tmp_path) -> None:
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)
    parsed = await BuiltinDocumentParser().parse(path)
    assert any("SCANNED_PDF_SUSPECTED" in warning for warning in parsed.warnings)
    assert "Docling" in parsed.warnings[0]
