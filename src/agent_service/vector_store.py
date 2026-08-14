from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from .embeddings import EmbeddingProvider, cosine_similarity
from .schemas import Citation, RetrievalHit


@dataclass(slots=True, frozen=True)
class VectorDocument:
    chunk_id: str
    source_id: str
    source_name: str
    text: str
    embedding: list[float]
    version_id: str | None = None
    tenant_id: str = "public"
    acl: tuple[str, ...] = ()
    page: int | None = None
    locator: str | None = None
    metadata: dict[str, Any] | None = None


class VectorStore(Protocol):
    async def initialize(self) -> None: ...

    async def upsert(self, documents: Sequence[VectorDocument]) -> None: ...

    async def delete_source(self, source_id: str, tenant_id: str = "public") -> None: ...

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: list[str] | None = None,
        active_versions: dict[str, str] | None = None,
        legacy_excluded_source_ids: set[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], float]: ...


class SQLiteVectorStore:
    """Exact cosine search for small/medium local corpora; no external service required."""

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        version_id TEXT,
                        tenant_id TEXT NOT NULL DEFAULT 'public',
                        acl_json TEXT NOT NULL DEFAULT '[]',
                        text TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        page INTEGER,
                        locator TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks(source_id);
                    """
                )
                cursor = await db.execute("PRAGMA table_info(document_chunks)")
                columns = {str(row[1]) for row in await cursor.fetchall()}
                if "version_id" not in columns:
                    await db.execute("ALTER TABLE document_chunks ADD COLUMN version_id TEXT")
                if "tenant_id" not in columns:
                    await db.execute(
                        "ALTER TABLE document_chunks ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'public'"
                    )
                if "acl_json" not in columns:
                    await db.execute(
                        "ALTER TABLE document_chunks ADD COLUMN acl_json TEXT NOT NULL DEFAULT '[]'"
                    )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_version ON document_chunks(version_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON document_chunks(tenant_id)"
                )
                await db.commit()
            self._initialized = True

    async def upsert(self, documents: Sequence[VectorDocument]) -> None:
        await self.initialize()
        rows = [
            (
                doc.chunk_id,
                doc.source_id,
                doc.source_name,
                doc.version_id,
                doc.tenant_id,
                json.dumps(sorted(set(doc.acl)), ensure_ascii=False),
                doc.text,
                json.dumps(doc.embedding),
                doc.page,
                doc.locator,
                json.dumps(doc.metadata or {}, ensure_ascii=False),
            )
            for doc in documents
        ]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO document_chunks
                    (chunk_id, source_id, source_name, version_id, tenant_id, acl_json,
                     text, embedding_json, page, locator, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    source_name=excluded.source_name,
                    version_id=excluded.version_id,
                    tenant_id=excluded.tenant_id,
                    acl_json=excluded.acl_json,
                    text=excluded.text,
                    embedding_json=excluded.embedding_json,
                    page=excluded.page,
                    locator=excluded.locator,
                    metadata_json=excluded.metadata_json
                """,
                rows,
            )
            await db.commit()

    async def delete_source(self, source_id: str, tenant_id: str = "public") -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM document_chunks WHERE source_id = ? AND tenant_id = ?",
                (source_id, tenant_id),
            )
            await db.commit()

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: list[str] | None = None,
        active_versions: dict[str, str] | None = None,
        legacy_excluded_source_ids: set[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], float]:
        await self.initialize()
        started = time.perf_counter()
        normalized_principals = sorted(set(principals or []))
        sql = "SELECT chunk_id, source_id, source_name, version_id, text, embedding_json, page, locator FROM document_chunks"
        params: list[Any] = [tenant_id]
        conditions: list[str] = ["tenant_id = ?"]
        if normalized_principals:
            placeholders = ",".join("?" for _ in normalized_principals)
            conditions.append(
                f"(acl_json = '[]' OR EXISTS (SELECT 1 FROM json_each(document_chunks.acl_json) WHERE value IN ({placeholders})))"
            )
            params.extend(normalized_principals)
        else:
            conditions.append("acl_json = '[]'")
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"source_id IN ({placeholders})")
            params.extend(source_ids)
        if active_versions is not None:
            visible = list(active_versions.items())
            legacy_excluded_sources = sorted(
                legacy_excluded_source_ids
                if legacy_excluded_source_ids is not None
                else active_versions
            )
            legacy_condition = "version_id IS NULL"
            visibility_params: list[str] = []
            if legacy_excluded_sources:
                active_placeholders = ",".join("?" for _ in legacy_excluded_sources)
                legacy_condition += f" AND source_id NOT IN ({active_placeholders})"
                visibility_params.extend(legacy_excluded_sources)
            visibility_conditions = [f"({legacy_condition})"]
            for source_id, version_id in visible:
                visibility_conditions.append("(source_id = ? AND version_id = ?)")
                visibility_params.extend((source_id, version_id))
            conditions.append(f"({' OR '.join(visibility_conditions)})")
            params.extend(visibility_params)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()

        query_vector = (await self.embedding_provider.embed([query]))[0]
        query_terms = _term_counts(query)
        candidates: list[tuple[sqlite3.Row, float, float]] = []
        max_lexical = 1.0
        for row in rows:
            vector = json.loads(row["embedding_json"])
            vector_score = cosine_similarity(query_vector, vector)
            lexical_score = _lexical_score(query_terms, _term_counts(row["text"]))
            max_lexical = max(max_lexical, lexical_score)
            candidates.append((row, vector_score, lexical_score))

        vector_ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        lexical_ranked = sorted(candidates, key=lambda item: item[2], reverse=True)
        vector_rank = {row["chunk_id"]: rank for rank, (row, _, _) in enumerate(vector_ranked, 1)}
        lexical_rank = {row["chunk_id"]: rank for rank, (row, _, _) in enumerate(lexical_ranked, 1)}

        hits: list[RetrievalHit] = []
        for row, vector_score, lexical_score in candidates:
            chunk_id = row["chunk_id"]
            fused = 1 / (60 + vector_rank[chunk_id]) + 1 / (60 + lexical_rank[chunk_id])
            normalized_lexical = lexical_score / max_lexical
            rerank = 0.55 * vector_score + 0.30 * normalized_lexical + 0.15 * fused * 60
            hits.append(
                RetrievalHit(
                    text=row["text"],
                    citation=Citation(
                        source_id=row["source_id"],
                        source_name=row["source_name"],
                        chunk_id=chunk_id,
                        page=row["page"],
                        locator=row["locator"],
                        quote=row["text"][:300],
                        score=rerank,
                    ),
                    vector_score=vector_score,
                    lexical_score=normalized_lexical,
                    fused_score=fused,
                    rerank_score=rerank,
                )
            )
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        return _mmr_deduplicate(hits, top_k), (time.perf_counter() - started) * 1000


def _term_counts(text: str) -> Counter[str]:
    terms = [term for term in text.lower().replace("_", " ").split() if term]
    return Counter(terms)


def _lexical_score(query: Counter[str], document: Counter[str]) -> float:
    return sum(min(count, document.get(term, 0)) for term, count in query.items())


def _mmr_deduplicate(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    seen_prefixes: set[str] = set()
    for hit in hits:
        prefix = " ".join(hit.text.lower().split()[:24])
        if prefix in seen_prefixes:
            continue
        selected.append(hit)
        seen_prefixes.add(prefix)
        if len(selected) >= top_k:
            break
    return selected


class QdrantVectorStore:
    """Optional HNSW-backed store for corpora that outgrow exact local scanning."""

    def __init__(self, url: str, collection: str, embedding_provider: EmbeddingProvider) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant client is not installed; run `uv sync --extra vector`"
            ) from exc
        self.client = (
            AsyncQdrantClient(location=":memory:")
            if url == ":memory:"
            else AsyncQdrantClient(url=url)
        )
        self.collection = collection
        self.embedding_provider = embedding_provider
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        from qdrant_client.models import Distance, VectorParams

        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_provider.dimension, distance=Distance.COSINE
                ),
            )
        self._initialized = True

    async def upsert(self, documents: Sequence[VectorDocument]) -> None:
        import uuid

        from qdrant_client.models import PointStruct

        await self.initialize()
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.chunk_id)),
                vector=doc.embedding,
                payload={
                    "chunk_id": doc.chunk_id,
                    "source_id": doc.source_id,
                    "source_name": doc.source_name,
                    "version_id": doc.version_id,
                    "tenant_id": doc.tenant_id,
                    **({"acl": list(doc.acl)} if doc.acl else {}),
                    "text": doc.text,
                    "page": doc.page,
                    "locator": doc.locator,
                    "metadata": doc.metadata or {},
                },
            )
            for doc in documents
        ]
        if points:
            await self.client.upsert(collection_name=self.collection, points=points, wait=True)

    async def delete_source(self, source_id: str, tenant_id: str = "public") -> None:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
        )

        await self.initialize()
        tenant_filters: list[Any] = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]
        if tenant_id == "public":
            tenant_filters.append(IsEmptyCondition(is_empty=PayloadField(key="tenant_id")))
        await self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                    Filter(should=tenant_filters),
                ]
            ),
            wait=True,
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: list[str] | None = None,
        active_versions: dict[str, str] | None = None,
        legacy_excluded_source_ids: set[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], float]:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchAny,
            MatchValue,
            PayloadField,
        )

        await self.initialize()
        started = time.perf_counter()
        normalized_principals = sorted(set(principals or []))
        tenant_filters: list[Any] = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]
        if tenant_id == "public":
            # Missing tenant metadata is the documented migration state for legacy points.
            tenant_filters.append(IsEmptyCondition(is_empty=PayloadField(key="tenant_id")))
        must_conditions: list[Any] = [Filter(should=tenant_filters)]
        acl_filters: list[Any] = [IsEmptyCondition(is_empty=PayloadField(key="acl"))]
        if normalized_principals:
            acl_filters.append(FieldCondition(key="acl", match=MatchAny(any=normalized_principals)))
        must_conditions.append(Filter(should=acl_filters))
        if source_ids:
            must_conditions.append(FieldCondition(key="source_id", match=MatchAny(any=source_ids)))
        if active_versions is not None:
            legacy_excluded_sources = sorted(
                legacy_excluded_source_ids
                if legacy_excluded_source_ids is not None
                else active_versions
            )
            legacy_must_not: list[Any] = []
            if legacy_excluded_sources:
                legacy_must_not.append(
                    FieldCondition(key="source_id", match=MatchAny(any=legacy_excluded_sources))
                )
            visible_filters = [
                Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="version_id"))],
                    must_not=legacy_must_not,
                )
            ]
            visible_filters.extend(
                Filter(
                    must=[
                        FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                        FieldCondition(key="version_id", match=MatchValue(value=version_id)),
                    ]
                )
                for source_id, version_id in active_versions.items()
            )
            must_conditions.append(Filter(should=visible_filters))
        query_filter = Filter(must=must_conditions) if must_conditions else None
        query_vector = (await self.embedding_provider.embed([query]))[0]
        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k * 3,
            with_payload=True,
        )
        query_terms = _term_counts(query)
        hits: list[RetrievalHit] = []
        for point in response.points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            lexical = _lexical_score(query_terms, _term_counts(text))
            vector_score = float(point.score)
            rerank = 0.8 * vector_score + 0.2 * min(1.0, lexical)
            hits.append(
                RetrievalHit(
                    text=text,
                    citation=Citation(
                        source_id=str(payload.get("source_id", "")),
                        source_name=str(payload.get("source_name", "")),
                        chunk_id=str(payload.get("chunk_id", "")),
                        page=payload.get("page"),
                        locator=payload.get("locator"),
                        quote=text[:300],
                        score=rerank,
                    ),
                    vector_score=vector_score,
                    lexical_score=min(1.0, lexical),
                    fused_score=vector_score,
                    rerank_score=rerank,
                )
            )
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        return _mmr_deduplicate(hits, top_k), (time.perf_counter() - started) * 1000
