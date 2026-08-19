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

from .embeddings import EmbeddingProvider, cosine_similarity, embed_query_compat
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
                    CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        text,
                        source_name,
                        locator,
                        metadata_text,
                        tokenize='unicode61 remove_diacritics 2'
                    );
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
                chunk_cursor = await db.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count_row = await chunk_cursor.fetchone()
                fts_cursor = await db.execute("SELECT COUNT(*) FROM document_chunks_fts")
                fts_count_row = await fts_cursor.fetchone()
                if chunk_count_row is None or fts_count_row is None:
                    raise RuntimeError("failed to inspect SQLite retrieval index state")
                chunk_count = int(chunk_count_row[0])
                fts_count = int(fts_count_row[0])
                if chunk_count != fts_count:
                    await db.execute("DELETE FROM document_chunks_fts")
                    cursor = await db.execute(
                        "SELECT chunk_id, text, source_name, locator, metadata_json FROM document_chunks"
                    )
                    existing_rows = await cursor.fetchall()
                    await db.executemany(
                        """
                        INSERT INTO document_chunks_fts(
                            chunk_id, text, source_name, locator, metadata_text
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                str(row[0]),
                                str(row[1]),
                                str(row[2]),
                                str(row[3] or ""),
                                _metadata_search_text(_decode_metadata(row[4])),
                            )
                            for row in existing_rows
                        ],
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
                "DELETE FROM document_chunks_fts WHERE chunk_id = ?",
                [(doc.chunk_id,) for doc in documents],
            )
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
            await db.executemany(
                """
                INSERT INTO document_chunks_fts(
                    chunk_id, text, source_name, locator, metadata_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        doc.chunk_id,
                        doc.text,
                        doc.source_name,
                        doc.locator or "",
                        _metadata_search_text(doc.metadata or {}),
                    )
                    for doc in documents
                ],
            )
            await db.commit()

    async def delete_source(self, source_id: str, tenant_id: str = "public") -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                DELETE FROM document_chunks_fts
                WHERE chunk_id IN (
                    SELECT chunk_id FROM document_chunks WHERE source_id = ? AND tenant_id = ?
                )
                """,
                (source_id, tenant_id),
            )
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
        query_vector = await embed_query_compat(self.embedding_provider, query)
        query_terms = _term_counts(query)
        row_list = list(rows)
        candidate_limit = max(top_k * 4, 20)
        async with asyncio.TaskGroup() as task_group:
            vector_task = task_group.create_task(
                asyncio.to_thread(
                    _safe_score_branch, "vector", _vector_scores, row_list, query_vector
                )
            )
            lexical_task = task_group.create_task(
                _fts_scores_with_fallback(
                    self.db_path,
                    query,
                    conditions,
                    params,
                    candidate_limit,
                    row_list,
                    query_terms,
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
        vector_rank = _rank_scores(vector_scores, limit=candidate_limit)
        lexical_rank = _rank_scores(lexical_scores, limit=candidate_limit, positive_only=True)
        metadata_rank = _rank_scores(metadata_scores, limit=candidate_limit, positive_only=True)
        max_lexical = max([1e-12, *lexical_scores.values()])
        max_metadata = max([1.0, *metadata_scores.values()])
        weights = _fusion_weights(self.embedding_provider.profile_id)
        rank_branches = [lexical_rank, metadata_rank]
        if weights.vector > 0:
            rank_branches.insert(0, vector_rank)

        hits: list[RetrievalHit] = []
        for row in row_list:
            chunk_id = str(row["chunk_id"])
            vector_score = vector_scores.get(chunk_id, 0.0)
            lexical_score = lexical_scores.get(chunk_id, 0.0)
            metadata_score = metadata_scores.get(chunk_id, 0.0)
            fused = sum(
                1 / (60 + ranks[chunk_id])
                for ranks in rank_branches
                if chunk_id in ranks
            )
            normalized_lexical = lexical_score / max_lexical
            normalized_metadata = metadata_score / max_metadata
            rerank = (
                weights.vector * vector_score
                + weights.lexical * normalized_lexical
                + weights.metadata * normalized_metadata
                + weights.rrf * fused * 60
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
        return _diversify_hits(hits, top_k), (time.perf_counter() - started) * 1000

    async def graph_neighbors(
        self,
        query: str,
        *,
        seed_hits: Sequence[RetrievalHit],
        max_neighbors: int,
        neighbors_per_seed: int,
        source_ids: list[str] | None = None,
        active_versions: dict[str, str] | None = None,
        legacy_excluded_source_ids: set[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], float]:
        """Return bounded one-hop neighbors from parent/heading structural edges."""

        if not 1 <= max_neighbors <= 20:
            raise ValueError("max_neighbors must be between 1 and 20")
        if not 1 <= neighbors_per_seed <= 5:
            raise ValueError("neighbors_per_seed must be between 1 and 5")
        await self.initialize()
        started = time.perf_counter()
        normalized_principals = sorted(set(principals or []))
        sql = (
            "SELECT chunk_id, source_id, source_name, version_id, text, page, locator, "
            "metadata_json FROM document_chunks"
        )
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
            legacy_excluded_sources = sorted(
                legacy_excluded_source_ids
                if legacy_excluded_source_ids is not None
                else active_versions
            )
            legacy_condition = "version_id IS NULL"
            visibility_params: list[str] = []
            if legacy_excluded_sources:
                placeholders = ",".join("?" for _ in legacy_excluded_sources)
                legacy_condition += f" AND source_id NOT IN ({placeholders})"
                visibility_params.extend(legacy_excluded_sources)
            visibility_conditions = [f"({legacy_condition})"]
            for source_id, version_id in active_versions.items():
                visibility_conditions.append("(source_id = ? AND version_id = ?)")
                visibility_params.extend((source_id, version_id))
            conditions.append(f"({' OR '.join(visibility_conditions)})")
            params.extend(visibility_params)
        sql += " WHERE " + " AND ".join(conditions)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(sql, params)
            rows = list(await cursor.fetchall())

        row_by_id = {str(row["chunk_id"]): row for row in rows}
        seed_ids = {
            hit.citation.chunk_id for hit in seed_hits if hit.citation.chunk_id is not None
        }
        query_terms = _term_counts(query)
        query_term_count = max(1, sum(query_terms.values()))
        selected: dict[str, RetrievalHit] = {}
        for seed in seed_hits:
            seed_id = seed.citation.chunk_id
            if seed_id is None or seed_id not in row_by_id:
                continue
            seed_row = row_by_id[seed_id]
            seed_metadata = _decode_metadata(seed_row["metadata_json"])
            seed_parent = _structural_parent(seed_metadata)
            seed_heading = _structural_heading(seed_metadata)
            if seed_parent is None and not seed_heading:
                continue
            seed_sequence = _structural_sequence(seed_row["locator"], seed_metadata)
            candidates: list[tuple[tuple[int, int, str], sqlite3.Row, float]] = []
            for row in rows:
                chunk_id = str(row["chunk_id"])
                if chunk_id in seed_ids:
                    continue
                if row["source_id"] != seed_row["source_id"]:
                    continue
                if row["version_id"] != seed_row["version_id"]:
                    continue
                metadata = _decode_metadata(row["metadata_json"])
                candidate_parent = _structural_parent(metadata)
                candidate_heading = _structural_heading(metadata)
                if seed_parent is not None and candidate_parent == seed_parent:
                    relation_rank = 0
                elif seed_heading and candidate_heading == seed_heading:
                    relation_rank = 1
                else:
                    continue
                candidate_sequence = _structural_sequence(row["locator"], metadata)
                distance = (
                    abs(candidate_sequence - seed_sequence)
                    if candidate_sequence is not None and seed_sequence is not None
                    else 1_000
                )
                structural_score = 1.0 / (1.0 + distance) if distance < 1_000 else 0.25
                candidates.append(((relation_rank, distance, chunk_id), row, structural_score))
            candidates.sort(key=lambda item: item[0])
            for _, row, structural_score in candidates[:neighbors_per_seed]:
                chunk_id = str(row["chunk_id"])
                text = str(row["text"])
                metadata = _decode_metadata(row["metadata_json"])
                lexical = min(
                    1.0, _lexical_score(query_terms, _term_counts(text)) / query_term_count
                )
                proposed_score = (
                    0.70 * seed.rerank_score + 0.15 * lexical + 0.15 * structural_score
                )
                rerank = min(proposed_score, max(0.0, seed.rerank_score * 0.95))
                hit = RetrievalHit(
                    text=text,
                    citation=_citation_from_evidence(
                        source_id=str(row["source_id"]),
                        source_name=str(row["source_name"]),
                        chunk_id=chunk_id,
                        text=text,
                        page=row["page"],
                        locator=row["locator"],
                        metadata=metadata,
                        query_terms=query_terms,
                        score=rerank,
                    ),
                    vector_score=0.0,
                    lexical_score=lexical,
                    fused_score=structural_score,
                    rerank_score=rerank,
                )
                current = selected.get(chunk_id)
                if current is None or hit.rerank_score > current.rerank_score:
                    selected[chunk_id] = hit
            if len(selected) >= max_neighbors:
                break
        hits = sorted(selected.values(), key=lambda hit: hit.rerank_score, reverse=True)
        return hits[:max_neighbors], (time.perf_counter() - started) * 1000

    def consume_diagnostics(self) -> list[str]:
        diagnostics = list(self._retrieval_diagnostics.get())
        self._retrieval_diagnostics.set(())
        return diagnostics


@dataclass(slots=True, frozen=True)
class _FusionWeights:
    vector: float
    lexical: float
    metadata: float
    rrf: float


def _fusion_weights(profile_id: str) -> _FusionWeights:
    if profile_id.startswith("hash-"):
        return _FusionWeights(vector=0.0, lexical=0.65, metadata=0.10, rrf=0.25)
    return _FusionWeights(vector=0.45, lexical=0.35, metadata=0.05, rrf=0.15)


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
        str(row["chunk_id"]): _lexical_score(query, _term_counts(str(row["text"]))) for row in rows
    }


async def _fts_scores_with_fallback(
    db_path: Path,
    query: str,
    conditions: list[str],
    params: list[Any],
    candidate_limit: int,
    rows: list[sqlite3.Row],
    query_terms: Counter[str],
) -> tuple[dict[str, float], str | None]:
    fts_query = _fts_match_query(query)
    if not fts_query:
        return {}, None
    try:
        sql = """
            SELECT document_chunks.chunk_id,
                   bm25(document_chunks_fts, 0.0, 1.0, 0.25, 0.15, 0.35) AS bm25_score
            FROM document_chunks_fts
            JOIN document_chunks
              ON document_chunks.chunk_id = document_chunks_fts.chunk_id
            WHERE document_chunks_fts MATCH ?
        """
        if conditions:
            sql += " AND " + " AND ".join(conditions)
        sql += " ORDER BY bm25_score LIMIT ?"
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(sql, [fts_query, *params, candidate_limit])
            matches = await cursor.fetchall()
        scores = {str(row[0]): max(0.0, -float(row[1])) for row in matches}
        return scores, None
    except Exception as exc:  # noqa: BLE001 - retain bounded lexical fallback
        scores, _ = await asyncio.to_thread(
            _safe_score_branch, "lexical_overlap", _lexical_scores, rows, query_terms
        )
        return scores, f"degraded_retrieval:fts5:{type(exc).__name__}"


def _fts_match_query(query: str) -> str:
    terms = list(dict.fromkeys(_term_counts(query)))[:32]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _metadata_search_text(metadata: dict[str, Any]) -> str:
    return " ".join(
        [
            " ".join(str(item) for item in metadata.get("heading_path", [])),
            str(metadata.get("block_type", "")),
            str(metadata.get("caption", "")),
        ]
    ).strip()


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


def _rank_scores(
    scores: dict[str, float],
    *,
    limit: int | None = None,
    positive_only: bool = False,
) -> dict[str, int]:
    eligible = (
        (chunk_id, score) for chunk_id, score in scores.items() if not positive_only or score > 0
    )
    ranked = sorted(eligible, key=lambda item: item[1], reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return {chunk_id: rank for rank, (chunk_id, _) in enumerate(ranked, 1)}


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _structural_parent(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("parent_id")
    return value if isinstance(value, str) and value else None


def _structural_heading(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = metadata.get("heading_path")
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _structural_sequence(locator: Any, metadata: dict[str, Any]) -> int | None:
    locator_text = str(locator or "")
    if ":chunk:" in locator_text:
        suffix = locator_text.rsplit(":chunk:", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    location = metadata.get("location")
    if isinstance(location, dict):
        reading_order = location.get("reading_order")
        if isinstance(reading_order, int):
            return reading_order
    return None


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
            location = {
                "kind": "text",
                "locator": str(locator) if locator else None,
                "block_id": block_id,
            }
    citation_id = (
        "cit_" + hashlib.sha256(f"{source_id}:{chunk_id}:{quote}".encode()).hexdigest()[:24]
    )
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


def _diversify_hits(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    deferred: list[RetrievalHit] = []
    seen_groups: set[tuple[str, int | None]] = set()
    seen_prefixes: set[str] = set()
    for hit in hits:
        prefix = " ".join(hit.text.lower().split()[:24])
        if prefix in seen_prefixes:
            continue
        group = (hit.citation.source_id, hit.citation.page)
        if group in seen_groups:
            deferred.append(hit)
            continue
        selected.append(hit)
        seen_groups.add(group)
        seen_prefixes.add(prefix)
        if len(selected) >= top_k:
            return selected
    for hit in deferred:
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
        query_vector = await embed_query_compat(self.embedding_provider, query)
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
        return _diversify_hits(hits, top_k), (time.perf_counter() - started) * 1000

    def consume_diagnostics(self) -> list[str]:
        diagnostics = list(self._retrieval_diagnostics.get())
        self._retrieval_diagnostics.set(())
        return diagnostics
