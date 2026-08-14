from __future__ import annotations

import pytest

from agent_service.embeddings import HashEmbedding
from agent_service.vector_store import QdrantVectorStore, VectorDocument


@pytest.mark.asyncio
async def test_qdrant_adapter_in_memory_round_trip() -> None:
    embeddings = HashEmbedding(dimension=64)
    store = QdrantVectorStore(":memory:", "test_documents", embeddings)
    await store.initialize()
    texts = [
        "Retry uses exponential backoff and fallback routes.",
        "Semantic retrieval returns source citations.",
    ]
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


@pytest.mark.asyncio
async def test_qdrant_search_exposes_only_active_version() -> None:
    embeddings = HashEmbedding(dimension=64)
    store = QdrantVectorStore(":memory:", "versioned_documents", embeddings)
    texts = ["old active evidence", "new building evidence"]
    vectors = await embeddings.embed(texts)
    await store.upsert(
        [
            VectorDocument(
                chunk_id="old:1",
                source_id="source",
                source_name="guide.md",
                version_id="old",
                text=texts[0],
                embedding=vectors[0],
            ),
            VectorDocument(
                chunk_id="new:1",
                source_id="source",
                source_name="guide.md",
                version_id="new",
                text=texts[1],
                embedding=vectors[1],
            ),
        ]
    )

    hits, _ = await store.search(
        "evidence",
        top_k=5,
        active_versions={"source": "old"},
    )

    assert hits
    assert all("old active" in hit.text for hit in hits)


@pytest.mark.asyncio
async def test_qdrant_filters_tenant_and_acl_before_returning_candidates() -> None:
    embeddings = HashEmbedding(dimension=64)
    store = QdrantVectorStore(":memory:", "tenant_acl_documents", embeddings)
    texts = ["tenant public evidence", "engineering secret", "other tenant evidence"]
    vectors = await embeddings.embed(texts)
    await store.upsert(
        [
            VectorDocument(
                chunk_id="public:1",
                source_id="public-doc",
                source_name="public.md",
                version_id="public-v1",
                tenant_id="tenant-a",
                text=texts[0],
                embedding=vectors[0],
            ),
            VectorDocument(
                chunk_id="restricted:1",
                source_id="restricted-doc",
                source_name="restricted.md",
                version_id="restricted-v1",
                tenant_id="tenant-a",
                acl=("group:engineering",),
                text=texts[1],
                embedding=vectors[1],
            ),
            VectorDocument(
                chunk_id="other:1",
                source_id="other-doc",
                source_name="other.md",
                version_id="other-v1",
                tenant_id="tenant-b",
                text=texts[2],
                embedding=vectors[2],
            ),
        ]
    )
    active = {
        "public-doc": "public-v1",
        "restricted-doc": "restricted-v1",
        "other-doc": "other-v1",
    }

    anonymous, _ = await store.search(
        "evidence secret", top_k=10, active_versions=active, tenant_id="tenant-a"
    )
    engineer, _ = await store.search(
        "evidence secret",
        top_k=10,
        active_versions=active,
        tenant_id="tenant-a",
        principals=["group:engineering"],
    )
    other, _ = await store.search(
        "evidence", top_k=10, active_versions=active, tenant_id="tenant-b"
    )

    assert {hit.citation.source_id for hit in anonymous} == {"public-doc"}
    assert {hit.citation.source_id for hit in engineer} == {
        "public-doc",
        "restricted-doc",
    }
    assert {hit.citation.source_id for hit in other} == {"other-doc"}


@pytest.mark.asyncio
async def test_qdrant_legacy_points_are_public_only_during_migration() -> None:
    import uuid

    from qdrant_client.models import PointStruct

    embeddings = HashEmbedding(dimension=64)
    store = QdrantVectorStore(":memory:", "legacy_tenant_documents", embeddings)
    await store.initialize()
    vector = (await embeddings.embed(["legacy public evidence"]))[0]
    await store.client.upsert(
        collection_name=store.collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "chunk_id": "legacy:1",
                    "source_id": "legacy",
                    "source_name": "legacy.md",
                    "version_id": None,
                    "text": "legacy public evidence",
                },
            )
        ],
        wait=True,
    )

    public_hits, _ = await store.search("legacy evidence", top_k=5, tenant_id="public")
    private_hits, _ = await store.search("legacy evidence", top_k=5, tenant_id="tenant-a")

    assert public_hits
    assert private_hits == []

    await store.delete_source("legacy", tenant_id="public")
    deleted_hits, _ = await store.search("legacy evidence", top_k=5, tenant_id="public")
    assert deleted_hits == []
