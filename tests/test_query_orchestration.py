from __future__ import annotations

import asyncio

import pytest

from agent_service.documents import CompositeDocumentParser, SemanticChunker
from agent_service.embeddings import HashEmbedding
from agent_service.query_planning import (
    build_query_plan,
    extract_protected_anchors,
    preserves_protected_anchors,
)
from agent_service.rag import RAGService
from agent_service.rag_registry import SQLiteVersionRegistry
from agent_service.schemas import Citation, DocumentIngestRequest, RetrievalHit
from agent_service.vector_store import SQLiteVectorStore


def _build_rag(tmp_path, **kwargs) -> RAGService:
    embeddings = HashEmbedding()
    return RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings),
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
        **kwargs,
    )


def _hit(version_id: str, suffix: str, text: str, *, lexical: float, rerank: float) -> RetrievalHit:
    return RetrievalHit(
        text=text,
        citation=Citation(
            source_id="guide",
            source_name="guide.md",
            chunk_id=f"{version_id}:{suffix}",
            quote=text,
            score=rerank,
        ),
        vector_score=rerank,
        lexical_score=lexical,
        fused_score=0.01,
        rerank_score=rerank,
    )


def test_query_plan_preserves_original_and_protected_constraints() -> None:
    query = 'Do not change API v2.1 timeout from "250 ms", explain Retry?'
    plan = build_query_plan(query, max_variants=3)

    assert plan.variants[0].query == query
    assert plan.variants[0].kind == "original"
    assert len(plan.variants) <= 3
    anchors = extract_protected_anchors(query)
    assert {"do not", "API", "v2.1", "250 ms"} <= set(anchors)
    assert "5%" in extract_protected_anchors("keep error rate below 5%")
    normalized = next(item for item in plan.variants if item.kind == "normalized")
    assert preserves_protected_anchors(normalized.query, anchors)


def test_query_plan_only_decomposes_explicit_delimiters_and_is_bounded() -> None:
    plan = build_query_plan("alpha? beta; gamma\ndelta", max_variants=3)

    assert [item.kind for item in plan.variants] == ["original", "subquery", "subquery"]
    assert [item.query for item in plan.variants[1:]] == ["alpha", "beta"]
    assert plan.discarded_variants == ("subquery_limit",)


def test_query_plan_preserves_newline_as_explicit_boundary() -> None:
    plan = build_query_plan("alpha\nbeta", max_variants=3)

    assert [item.query for item in plan.variants] == ["alpha beta", "alpha", "beta"]
    assert [item.kind for item in plan.variants] == ["original", "subquery", "subquery"]


def test_query_plan_does_not_split_delimiters_inside_quotes() -> None:
    plan = build_query_plan('"alpha? beta"; gamma', max_variants=3)

    assert [item.query for item in plan.variants] == [
        '"alpha? beta"; gamma',
        '"alpha? beta"',
        "gamma",
    ]
    assert plan.variants[1].protected_anchors == ("alpha? beta",)


@pytest.mark.asyncio
async def test_sufficient_original_query_avoids_extra_retrieval_cost(tmp_path, monkeypatch) -> None:
    (tmp_path / "guide.md").write_text("retry evidence", encoding="utf-8")
    rag = _build_rag(tmp_path)
    active = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))
    calls: list[str] = []

    async def fake_search(query, **kwargs):
        calls.append(query)
        return [_hit(active.version_id, "original", "retry evidence", lexical=1.0, rerank=0.8)], 1.0

    monkeypatch.setattr(rag.store, "search", fake_search)
    response = await rag.retrieve("Retry, evidence?", top_k=2, source_ids=["guide"])

    assert calls == ["Retry, evidence?"]
    assert response.hits
    assert "corrective_retrieval:triggered" not in response.warnings
    assert response.degraded_retrieval is False


@pytest.mark.asyncio
async def test_low_confidence_query_uses_normalized_fallback_and_fuses_original(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "guide.md").write_text("evidence", encoding="utf-8")
    rag = _build_rag(tmp_path)
    active = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))
    calls: list[str] = []

    async def fake_search(query, **kwargs):
        calls.append(query)
        if query == "Retry, fallback?":
            return [
                _hit(active.version_id, "original", "original evidence", lexical=0.0, rerank=0.0)
            ], 1.0
        return [
            _hit(active.version_id, "normalized", "fallback evidence", lexical=1.0, rerank=0.9)
        ], 1.0

    monkeypatch.setattr(rag.store, "search", fake_search)
    response = await rag.retrieve("Retry, fallback?", top_k=2, source_ids=["guide"])

    assert calls == ["Retry, fallback?", "retry fallback"]
    assert {hit.text for hit in response.hits} == {"original evidence", "fallback evidence"}
    assert "corrective_retrieval:triggered" in response.warnings
    assert "query_fusion:rrf:2" in response.warnings
    assert response.degraded_retrieval is False


@pytest.mark.asyncio
async def test_explicit_subqueries_run_in_parallel_with_bounded_count(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "guide.md").write_text("evidence", encoding="utf-8")
    rag = _build_rag(tmp_path, query_max_parallel=2)
    active = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))
    active_calls = 0
    max_active_calls = 0
    original_completed = False
    subquery_started_before_original_completed = False
    calls: list[str] = []

    async def fake_search(query, **kwargs):
        nonlocal active_calls, max_active_calls
        nonlocal original_completed, subquery_started_before_original_completed
        calls.append(query)
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        if query != "alpha? beta? gamma?" and not original_completed:
            subquery_started_before_original_completed = True
        try:
            await asyncio.sleep(0.02)
            if query == "alpha? beta? gamma?":
                original_completed = True
                return [
                    _hit(
                        active.version_id, "original", "original evidence", lexical=1.0, rerank=0.8
                    )
                ], 20.0
            return [
                _hit(active.version_id, query, f"{query} evidence", lexical=1.0, rerank=0.8)
            ], 20.0
        finally:
            active_calls -= 1

    monkeypatch.setattr(rag.store, "search", fake_search)
    response = await rag.retrieve("alpha? beta? gamma?", top_k=3, source_ids=["guide"])

    assert set(calls) == {"alpha? beta? gamma?", "alpha", "beta"}
    assert subquery_started_before_original_completed is True
    assert max_active_calls == 2
    assert len(calls) == 3
    assert len(response.hits) == 3


@pytest.mark.asyncio
async def test_query_timeout_keeps_verified_original_evidence(tmp_path, monkeypatch) -> None:
    (tmp_path / "guide.md").write_text("evidence", encoding="utf-8")
    rag = _build_rag(tmp_path, query_timeout_seconds=0.01)
    active = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    async def fake_search(query, **kwargs):
        if query == "Retry, fallback?":
            return [
                _hit(active.version_id, "original", "original evidence", lexical=0.0, rerank=0.0)
            ], 1.0
        await asyncio.sleep(0.05)
        return [], 50.0

    monkeypatch.setattr(rag.store, "search", fake_search)
    response = await rag.retrieve("Retry, fallback?", top_k=2, source_ids=["guide"])

    assert [hit.text for hit in response.hits] == ["original evidence"]
    assert "query_search_timeout:normalized" in response.warnings
    assert response.degraded_retrieval is True
