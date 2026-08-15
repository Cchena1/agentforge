from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
from pydantic import TypeAdapter

from .embeddings import EmbeddingProvider, cosine_similarity
from .schemas import Citation, DocumentLocation, RetrievalHit

_LOCATION_ADAPTER: TypeAdapter[DocumentLocation] = TypeAdapter(DocumentLocation)


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

    def consume_diagnostics(self) -> list[str]: ...


class SQLiteVectorStore:
    """Exact cosine search for small/medium local corpora; no external service required."""

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._retrieval_diagnostics: ContextVar[tuple[str, ...]] = ContextVar(
            f"sqlite_retrieval_diagnostics_{id(self)}", default=()
        )

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
        sql = "SELECT chunk_id, source_id, source_name, version_id, text, embedding_json, page, locator, metadata_json FROM document_chunks"
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

        self._retrieval_diagnostics.set(())
        query_vector = (await self.embedding_provider.embed([query]))[0]
        query_terms = _term_counts(query)
        row_list = list(rows)
        async with asyncio.TaskGroup() as task_group:
            vector_task = task_group.create_task(
                asyncio.to_thread(
                    _safe_score_branch, "vector", _vector_scores, row_list, query_vector
                )
            )
            lexical_task = task_group.create_task(
                asyncio.to_thread(
                    _safe_score_branch, "lexical", _lexical_scores, row_list, query_terms
                )
            )
            metadata_task = task_group.create_task(
                asyncio.to_thread(
                    _safe_score_branch, "metadata", _metadata_scores, row_list, query_terms
                )
            )
        vector_scores, vector_warning = vector_task.result()
        lexical_scores, lexical_warning = lexical_task.result()
        metadata_scores, metadata_warning = metadata_task.result()
        diagnostics = tuple(
            warning
            for warning in (vector_warning, lexical_warning, metadata_warning)
            if warning is not None
        )
        self._retrieval_diagnostics.set(diagnostics)
        if row_list and not any((vector_scores, lexical_scores, metadata_scores)):
            raise RuntimeError("all retrieval scoring branches failed")
        vector_rank = _rank_scores(vector_scores)
        lexical_rank = _rank_scores(lexical_scores)
        metadata_rank = _rank_scores(metadata_scores)
        max_lexical = max([1.0, *lexical_scores.values()])
        max_metadata = max([1.0, *metadata_scores.values()])

        hits: list[RetrievalHit] = []
        for row in row_list:
            chunk_id = str(row["chunk_id"])
            vector_score = vector_scores.get(chunk_id, 0.0)
            lexical_score = lexical_scores.get(chunk_id, 0.0)
            metadata_score = metadata_scores.get(chunk_id, 0.0)
            fused = sum(
                1 / (60 + ranks[chunk_id])
                for ranks in (vector_rank, lexical_rank, metadata_rank)
                if chunk_id in ranks
            )
            normalized_lexical = lexical_score / max_lexical
            normalized_metadata = metadata_score / max_metadata
            rerank = (
                0.50 * vector_score
                + 0.25 * normalized_lexical
                + 0.10 * normalized_metadata
                + 0.15 * fused * 60
            )
            metadata = _decode_metadata(row["metadata_json"])
            hits.append(
                RetrievalHit(
                    text=str(row["text"]),
                    citation=_citation_from_evidence(
                        source_id=str(row["source_id"]),
                        source_name=str(row["source_name"]),
                        chunk_id=chunk_id,
                        text=str(row["text"]),
                        page=row["page"],
                        locator=row["locator"],
                        metadata=metadata,
                        query_terms=query_terms,
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

    def consume_diagnostics(self) -> list[str]:
        diagnostics = list(self._retrieval_diagnostics.get())
        self._retrieval_diagnostics.set(())
        return diagnostics


def _term_counts(text: str) -> Counter[str]:
    terms = [term for term in text.lower().replace("_", " ").split() if term]
    return Counter(terms)


def _lexical_score(query: Counter[str], document: Counter[str]) -> float:
    return sum(min(count, document.get(term, 0)) for term, count in query.items())


def _safe_score_branch(
    name: str,
    scorer: Any,
    *args: Any,
) -> tuple[dict[str, float], str | None]:
    try:
        return scorer(*args), None
    except Exception as exc:  # noqa: BLE001 - isolate independent retrieval branches
        return {}, f"degraded_retrieval:{name}:{type(exc).__name__}"


def _vector_scores(rows: list[sqlite3.Row], query_vector: list[float]) -> dict[str, float]:
    return {
        str(row["chunk_id"]): cosine_similarity(
            query_vector, [float(item) for item in json.loads(row["embedding_json"])]
        )
        for row in rows
    }


def _lexical_scores(rows: list[sqlite3.Row], query: Counter[str]) -> dict[str, float]:
    return {
        str(row["chunk_id"]): _lexical_score(query, _term_counts(str(row["text"])))
        for row in rows
    }


def _metadata_scores(rows: list[sqlite3.Row], query: Counter[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        metadata = _decode_metadata(row["metadata_json"])
        searchable = " ".join(
            [
                str(row["source_name"]),
                str(row["locator"] or ""),
                " ".join(str(item) for item in metadata.get("heading_path", [])),
                str(metadata.get("block_type", "")),
            ]
        )
        scores[str(row["chunk_id"])] = _lexical_score(query, _term_counts(searchable))
    return scores


def _rank_scores(scores: dict[str, float]) -> dict[str, int]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {chunk_id: rank for rank, (chunk_id, _) in enumerate(ranked, 1)}


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _evidence_quote(text: str, query_terms: Counter[str], max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    positions = [lowered.find(term) for term in query_terms if lowered.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - max_chars // 3)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return text[start:end]


def _citation_from_evidence(
    *,
    source_id: str,
    source_name: str,
    chunk_id: str,
    text: str,
    page: Any,
    locator: Any,
    metadata: dict[str, Any],
    query_terms: Counter[str],
    score: float,
) -> Citation:
    quote = _evidence_quote(text, query_terms)
    location = metadata.get("location")
    if not isinstance(location, dict):
        source_block_ids = metadata.get("source_block_ids") or []
        block_id = str(source_block_ids[0]) if source_block_ids else chunk_id
        if page is not None:
            location = {
                "kind": "pdf",
                "page_number": int(page),
                "bbox": None,
                "block_id": block_id,
                "reading_order": int(metadata.get("reading_order", 0)),
            }
        else:
            location = {"kind": "text", "locator": str(locator) if locator else None, "block_id": block_id}
    citation_id = "cit_" + hashlib.sha256(
        f"{source_id}:{chunk_id}:{quote}".encode()
    ).hexdigest()[:24]
    return Citation(
        citation_id=citation_id,
        source_id=source_id,
        source_name=source_name,
        chunk_id=chunk_id,
        page=int(page) if page is not None else None,
        locator=str(locator) if locator else None,
        location=_LOCATION_ADAPTER.validate_python(location, strict=True),
        quote=quote,
        score=score,
        content_sha256=metadata.get("content_sha256"),
        parser=metadata.get("parser"),
    )


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
        self._retrieval_diagnostics: ContextVar[tuple[str, ...]] = ContextVar(
            f"qdrant_retrieval_diagnostics_{id(self)}", default=()
        )

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
        self._retrieval_diagnostics.set(())
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
                    citation=_citation_from_evidence(
                        source_id=str(payload.get("source_id", "")),
                        source_name=str(payload.get("source_name", "")),
                        chunk_id=str(payload.get("chunk_id", "")),
                        text=text,
                        page=payload.get("page"),
                        locator=payload.get("locator"),
                        metadata=_decode_metadata(payload.get("metadata")),
                        query_terms=query_terms,
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

    def consume_diagnostics(self) -> list[str]:
        diagnostics = list(self._retrieval_diagnostics.get())
        self._retrieval_diagnostics.set(())
        return diagnostics
