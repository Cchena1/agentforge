from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from .embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    embed_documents_compat,
    embed_query_compat,
)
from .schemas import ChatMessage, MemoryRecord, MemoryWrite

if TYPE_CHECKING:
    from .observability import Observability

_MEMORY_SCHEMA_VERSION = 2
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)


class MemoryStore:
    """Owns bounded session context and namespaced, validity-aware long-term memory."""

    def __init__(
        self,
        db_path: Path,
        embedding_provider: EmbeddingProvider,
        *,
        message_limit: int,
        char_budget: int,
        min_relevance_score: float = 0.35,
        recency_half_life_days: float = 180.0,
        observability: Observability | None = None,
    ) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.message_limit = message_limit
        self.char_budget = char_budget
        self.min_relevance_score = min_relevance_score
        self.recency_half_life_days = recency_half_life_days
        self.observability = observability
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
                    CREATE TABLE IF NOT EXISTS memory_schema_metadata (
                        component TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS session_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        name TEXT,
                        tool_call_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_namespace_id
                        ON session_messages(namespace, id DESC);
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        namespace TEXT PRIMARY KEY,
                        summary TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        memory_id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        importance REAL NOT NULL,
                        metadata_json TEXT NOT NULL,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        memory_key TEXT,
                        embedding_profile TEXT NOT NULL DEFAULT '',
                        valid_from TEXT,
                        expires_at TEXT,
                        superseded_at TEXT,
                        superseded_by TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_long_memory_namespace
                        ON long_term_memories(namespace);
                    """
                )
                await self._migrate_long_term_memory(db)
                await db.execute(
                    """CREATE INDEX IF NOT EXISTS idx_long_memory_active_key
                       ON long_term_memories(namespace, memory_key, superseded_at)"""
                )
                now = datetime.now(UTC).isoformat()
                await db.execute(
                    """INSERT INTO memory_schema_metadata(component, schema_version, updated_at)
                       VALUES ('memory', ?, ?)
                       ON CONFLICT(component) DO UPDATE SET
                           schema_version=excluded.schema_version,
                           updated_at=excluded.updated_at""",
                    (_MEMORY_SCHEMA_VERSION, now),
                )
                await db.commit()
            self._initialized = True

    async def _migrate_long_term_memory(self, db: aiosqlite.Connection) -> None:
        rows = await (await db.execute("PRAGMA table_info(long_term_memories)")).fetchall()
        columns = {str(row[1]) for row in rows}
        additions = {
            "memory_key": "TEXT",
            "embedding_profile": "TEXT NOT NULL DEFAULT ''",
            "valid_from": "TEXT",
            "expires_at": "TEXT",
            "superseded_at": "TEXT",
            "superseded_by": "TEXT",
        }
        for column, definition in additions.items():
            if column not in columns:
                await db.execute(
                    f"ALTER TABLE long_term_memories ADD COLUMN {column} {definition}"
                )
        await db.execute(
            """UPDATE long_term_memories
               SET valid_from = created_at
               WHERE valid_from IS NULL"""
        )

    @staticmethod
    def namespace(user_id: str, session_id: str) -> str:
        return f"user:{user_id}:session:{session_id}"

    async def append_messages(self, namespace: str, messages: list[ChatMessage]) -> None:
        await self.initialize()
        now = datetime.now(UTC).isoformat()
        rows = [(namespace, msg.role, msg.content, msg.name, msg.tool_call_id, now) for msg in messages]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT INTO session_messages
                   (namespace, role, content, name, tool_call_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()
        await self._compact(namespace)

    async def load_context(self, namespace: str) -> tuple[str, list[ChatMessage]]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            summary_row = await (
                await db.execute(
                    "SELECT summary FROM session_summaries WHERE namespace = ?", (namespace,)
                )
            ).fetchone()
            cursor = await db.execute(
                """SELECT role, content, name, tool_call_id FROM session_messages
                   WHERE namespace = ? ORDER BY id DESC LIMIT ?""",
                (namespace, self.message_limit),
            )
            rows = list(reversed(list(await cursor.fetchall())))
        messages = [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                name=row["name"],
                tool_call_id=row["tool_call_id"],
            )
            for row in rows
        ]
        return (summary_row["summary"] if summary_row else ""), self._fit_budget(messages)

    async def _compact(self, namespace: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """SELECT id, role, content FROM session_messages
                   WHERE namespace = ? ORDER BY id DESC""",
                (namespace,),
            )
            rows = list(await cursor.fetchall())
            overflow = rows[self.message_limit :]
            if not overflow:
                return
            previous = await (
                await db.execute(
                    "SELECT summary FROM session_summaries WHERE namespace = ?", (namespace,)
                )
            ).fetchone()
            existing = previous["summary"] if previous else ""
            chronological = list(reversed(overflow))
            compact = _deterministic_summary(
                existing, [(row["role"], row["content"]) for row in chronological]
            )
            await db.execute(
                """INSERT INTO session_summaries(namespace, summary, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(namespace) DO UPDATE SET
                       summary=excluded.summary, updated_at=excluded.updated_at""",
                (namespace, compact, datetime.now(UTC).isoformat()),
            )
            overflow_ids = [row["id"] for row in overflow]
            placeholders = ",".join("?" for _ in overflow_ids)
            await db.execute(
                f"DELETE FROM session_messages WHERE id IN ({placeholders})", overflow_ids
            )
            await db.commit()

    def _fit_budget(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        selected: list[ChatMessage] = []
        used = 0
        for message in reversed(messages):
            size = len(message.content)
            if selected and used + size > self.char_budget:
                break
            selected.append(message)
            used += size
        return list(reversed(selected))

    async def remember(self, item: MemoryWrite) -> MemoryRecord:
        await self.initialize()
        embedding = await embed_query_compat(self.embedding_provider, item.text)
        now = datetime.now(UTC)
        valid_from = item.valid_from or now
        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            namespace=item.namespace,
            text=item.text,
            importance=item.importance,
            metadata=dict(item.metadata),
            created_at=now,
            memory_key=item.memory_key,
            valid_from=valid_from,
            expires_at=item.expires_at,
            embedding_profile=self.embedding_provider.profile_id,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            if item.memory_key is not None:
                await db.execute(
                    """UPDATE long_term_memories
                       SET superseded_at = ?, superseded_by = ?, updated_at = ?
                       WHERE namespace = ? AND memory_key = ? AND superseded_at IS NULL""",
                    (
                        now.isoformat(),
                        record.memory_id,
                        now.isoformat(),
                        item.namespace,
                        item.memory_key,
                    ),
                )
            await db.execute(
                """INSERT INTO long_term_memories
                   (memory_id, namespace, text, embedding_json, importance, metadata_json,
                    created_at, updated_at, memory_key, embedding_profile, valid_from,
                    expires_at, superseded_at, superseded_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (
                    record.memory_id,
                    record.namespace,
                    record.text,
                    json.dumps(embedding),
                    record.importance,
                    json.dumps(record.metadata, ensure_ascii=False),
                    now.isoformat(),
                    now.isoformat(),
                    record.memory_key,
                    record.embedding_profile,
                    _isoformat(valid_from),
                    _isoformat(item.expires_at),
                ),
            )
            await db.commit()
        self._record_memory("write", "success")
        return record

    async def recall(self, namespace: str, query: str, top_k: int = 5) -> list[MemoryRecord]:
        await self.initialize()
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        query_vector = await embed_query_compat(self.embedding_provider, query)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            rows = list(
                await (
                    await db.execute(
                        """SELECT * FROM long_term_memories
                           WHERE namespace = ?
                             AND superseded_at IS NULL
                             AND valid_from <= ?
                             AND (expires_at IS NULL OR expires_at > ?)""",
                        (namespace, now_iso, now_iso),
                    )
                ).fetchall()
            )
            if not rows:
                self._record_memory("recall", "empty")
                return []
            rows = await self._refresh_stale_embeddings(db, rows)
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                vector = json.loads(row["embedding_json"])
                semantic = max(0.0, cosine_similarity(query_vector, vector))
                lexical = _lexical_overlap(query, str(row["text"]))
                importance = float(row["importance"])
                recency = _recency_score(str(row["updated_at"]), now, self.recency_half_life_days)
                score = 0.65 * semantic + 0.20 * lexical + 0.10 * importance + 0.05 * recency
                if score >= self.min_relevance_score:
                    scored.append((min(1.0, score), row))
            selected = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
            if selected:
                await db.executemany(
                    "UPDATE long_term_memories SET access_count = access_count + 1 WHERE memory_id = ?",
                    [(row["memory_id"],) for _, row in selected],
                )
                await db.commit()
        self._record_memory("recall", "hit" if selected else "abstained")
        return [_row_to_record(row, score) for score, row in selected]

    async def _refresh_stale_embeddings(
        self, db: aiosqlite.Connection, rows: list[sqlite3.Row]
    ) -> list[sqlite3.Row]:
        stale = []
        for row in rows:
            vector = json.loads(row["embedding_json"])
            if (
                row["embedding_profile"] != self.embedding_provider.profile_id
                or len(vector) != self.embedding_provider.dimension
            ):
                stale.append(row)
        if not stale:
            return rows
        vectors = await embed_documents_compat(
            self.embedding_provider, [str(row["text"]) for row in stale]
        )
        await db.executemany(
            """UPDATE long_term_memories
               SET embedding_json = ?, embedding_profile = ?
               WHERE memory_id = ?""",
            [
                (
                    json.dumps(vector),
                    self.embedding_provider.profile_id,
                    row["memory_id"],
                )
                for row, vector in zip(stale, vectors, strict=True)
            ],
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM long_term_memories WHERE memory_id IN ("
            + ",".join("?" for _ in rows)
            + ")",
            [row["memory_id"] for row in rows],
        )
        return list(await cursor.fetchall())

    async def delete(self, namespace: str, memory_id: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM long_term_memories WHERE namespace = ? AND memory_id = ?",
                (namespace, memory_id),
            )
            await db.commit()
            deleted = bool(cursor.rowcount and cursor.rowcount > 0)
        self._record_memory("delete", "success" if deleted else "not_found")
        return deleted

    def _record_memory(self, operation: str, outcome: str) -> None:
        if self.observability is not None:
            self.observability.record_memory_operation(operation, outcome)


def _row_to_record(row: sqlite3.Row, relevance_score: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"],
        namespace=row["namespace"],
        text=row["text"],
        importance=row["importance"],
        metadata=json.loads(row["metadata_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        memory_key=row["memory_key"],
        valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
        expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        embedding_profile=row["embedding_profile"] or None,
        relevance_score=relevance_score,
    )


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _recency_score(updated_at: str, now: datetime, half_life_days: float) -> float:
    timestamp = datetime.fromisoformat(updated_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    age_days = max(0.0, (now - timestamp.astimezone(UTC)).total_seconds() / 86_400)
    return math.exp(-math.log(2) * age_days / half_life_days)


def _deterministic_summary(
    existing: str, messages: list[tuple[str, str]], limit: int = 6000
) -> str:
    lines = [existing.strip()] if existing.strip() else []
    for role, content in messages:
        normalized = " ".join(content.split())
        if normalized:
            lines.append(f"{role}: {normalized[:600]}")
    combined = "\n".join(lines)
    return combined[-limit:]
