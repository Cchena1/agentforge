from __future__ import annotations

import pytest

from agent_service.embeddings import HashEmbedding
from agent_service.vector_store import QdrantVectorStore, VectorDocument


@pytest.mark.asyncio
async def test_qdrant_adapter_in_memory_round_trip() -> None:
    embeddings = HashEmbedding(dimension=64)
    store = QdrantVectorStore(":memory:", "test_documents", embeddings)
    await store.initialize()
    texts = ["Retry uses exponential backoff and fallback routes.", "Semantic retrieval returns source citations."]
    vectors = await embeddings.embed(texts)
    await store.upsert(
        [
            VectorDocument(
                chunk_id=f"source:{index}",
                source_id="source",
                source_name="guide.md",
                text=text,
                embedding=vector,
            )
            for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ]
    )

    hits, latency_ms = await store.search("retry fallback", top_k=1, source_ids=["source"])

    assert len(hits) == 1
    assert hits[0].citation.source_name == "guide.md"
    assert latency_ms >= 0

    await store.delete_source("source")
    hits_after_delete, _ = await store.search("retry fallback", top_k=1)
    assert hits_after_delete == []
