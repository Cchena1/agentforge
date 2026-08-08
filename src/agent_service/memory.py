from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from .embeddings import EmbeddingProvider, cosine_similarity
from .schemas import ChatMessage, MemoryRecord, MemoryWrite


class MemoryStore:
    """Owns short-term session history and long-term semantic memories in separate tables."""

    def __init__(
        self,
        db_path: Path,
        embedding_provider: EmbeddingProvider,
        *,
        message_limit: int,
        char_budget: int,
    ) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.message_limit = message_limit
        self.char_budget = char_budget
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
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_long_memory_namespace
                        ON long_term_memories(namespace);
                    """
                )
                await db.commit()
            self._initialized = True

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
                await db.execute("SELECT summary FROM session_summaries WHERE namespace = ?", (namespace,))
            ).fetchone()
            cursor = await db.execute(
                """SELECT role, content, name, tool_call_id FROM session_messages
                   WHERE namespace = ? ORDER BY id DESC LIMIT ?""",
                (namespace, self.message_limit),
            )
            fetched_rows = list(await cursor.fetchall())
            rows = list(reversed(fetched_rows))
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
            keep = rows[: self.message_limit]
            overflow = rows[self.message_limit :]
            if not overflow:
                return
            previous = await (
                await db.execute("SELECT summary FROM session_summaries WHERE namespace = ?", (namespace,))
            ).fetchone()
            existing = previous["summary"] if previous else ""
            chronological = list(reversed(overflow))
            compact = _deterministic_summary(existing, [(row["role"], row["content"]) for row in chronological])
            await db.execute(
                """INSERT INTO session_summaries(namespace, summary, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(namespace) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at""",
                (namespace, compact, datetime.now(UTC).isoformat()),
            )
            overflow_ids = [row["id"] for row in overflow]
            placeholders = ",".join("?" for _ in overflow_ids)
            await db.execute(f"DELETE FROM session_messages WHERE id IN ({placeholders})", overflow_ids)
            await db.commit()
            _ = keep

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
        embedding = (await self.embedding_provider.embed([item.text]))[0]
        now = datetime.now(UTC)
        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            namespace=item.namespace,
            text=item.text,
            importance=item.importance,
            metadata=dict(item.metadata),
            created_at=now,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO long_term_memories
                   (memory_id, namespace, text, embedding_json, importance, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.memory_id,
                    record.namespace,
                    record.text,
                    json.dumps(embedding),
                    record.importance,
                    json.dumps(record.metadata, ensure_ascii=False),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await db.commit()
        return record

    async def recall(self, namespace: str, query: str, top_k: int = 5) -> list[MemoryRecord]:
        await self.initialize()
        query_vector = (await self.embedding_provider.embed([query]))[0]
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            rows = await (
                await db.execute("SELECT * FROM long_term_memories WHERE namespace = ?", (namespace,))
            ).fetchall()
            scored = []
            for row in rows:
                similarity = cosine_similarity(query_vector, json.loads(row["embedding_json"]))
                score = 0.8 * similarity + 0.2 * float(row["importance"])
                scored.append((score, row))
            selected = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
            if selected:
                await db.executemany(
                    "UPDATE long_term_memories SET access_count = access_count + 1 WHERE memory_id = ?",
                    [(row["memory_id"],) for _, row in selected],
                )
                await db.commit()
        return [
            MemoryRecord(
                memory_id=row["memory_id"],
                namespace=row["namespace"],
                text=row["text"],
                importance=row["importance"],
                metadata=json.loads(row["metadata_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for _, row in selected
        ]


def _deterministic_summary(existing: str, messages: list[tuple[str, str]], limit: int = 6000) -> str:
    lines = [existing.strip()] if existing.strip() else []
    for role, content in messages:
        normalized = " ".join(content.split())
        if normalized:
            lines.append(f"{role}: {normalized[:600]}")
    combined = "\n".join(lines)
    return combined[-limit:]
