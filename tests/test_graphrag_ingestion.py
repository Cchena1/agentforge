from __future__ import annotations

import sqlite3

import pytest

from agent_service.documents import CompositeDocumentParser, SemanticChunker
from agent_service.embeddings import HashEmbedding
from agent_service.knowledge_graph import (
    HeuristicKnowledgeGraphExtractor,
    KnowledgeGraphBuilder,
    SQLiteKnowledgeGraphStore,
)
from agent_service.rag import RAGService
from agent_service.rag_registry import SQLiteVersionRegistry
from agent_service.schemas import DocumentIngestRequest
from agent_service.vector_store import SQLiteVectorStore


@pytest.mark.asyncio
async def test_rag_ingestion_persists_versioned_knowledge_graph(tmp_path) -> None:
    source = tmp_path / "architecture.md"
    source.write_text(
        "# Architecture\n\nAlpha Service depends on Beta Gateway. "
        "Beta Gateway connects Gamma Database.",
        encoding="utf-8",
    )
    embeddings = HashEmbedding()
    vector_store = SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings)
    graph_path = tmp_path / "knowledge_graph.sqlite3"
    graph_store = SQLiteKnowledgeGraphStore(graph_path)
    rag = RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        vector_store,
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
        knowledge_graph_builder=KnowledgeGraphBuilder(HeuristicKnowledgeGraphExtractor()),
        knowledge_graph_store=graph_store,
    )
    await rag.initialize()

    result = await rag.ingest(
        DocumentIngestRequest(
            file_path=source.name,
            source_id="architecture",
            tenant_id="tenant-a",
            acl=["engineering"],
        )
    )

    assert result.version_id is not None
    assert any(warning.startswith("knowledge_graph:indexed:") for warning in result.warnings)
    with sqlite3.connect(graph_path) as database:
        entity_count = database.execute(
            """
            SELECT COUNT(DISTINCT entity_id)
            FROM kg_mentions
            WHERE version_id = ? AND tenant_id = ?
            """,
            (result.version_id, "tenant-a"),
        ).fetchone()[0]
        relation_count = database.execute(
            "SELECT COUNT(*) FROM kg_relations WHERE version_id = ? AND tenant_id = ?",
            (result.version_id, "tenant-a"),
        ).fetchone()[0]
    assert entity_count >= 4
    assert relation_count >= 3
