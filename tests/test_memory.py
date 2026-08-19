from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from agent_service.embeddings import HashEmbedding
from agent_service.memory import MemoryStore
from agent_service.schemas import ChatMessage, MemoryWrite


@pytest.mark.asyncio
async def test_short_term_memory_is_isolated_and_compacted(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3", HashEmbedding(), message_limit=4, char_budget=1000)
    ns_a = store.namespace("user-a", "session-1")
    ns_b = store.namespace("user-b", "session-1")
    for index in range(8):
        await store.append_messages(ns_a, [ChatMessage(role="user", content=f"message-{index}")])
    await store.append_messages(ns_b, [ChatMessage(role="user", content="private-b")])

    summary_a, context_a = await store.load_context(ns_a)
    summary_b, context_b = await store.load_context(ns_b)
    assert len(context_a) <= 4
    assert "message-0" in summary_a
    assert "private-b" not in summary_a
    assert summary_b == ""
    assert context_b[0].content == "private-b"


@pytest.mark.asyncio
async def test_long_term_memory_uses_semantic_namespace(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3", HashEmbedding(), message_limit=4, char_budget=1000)
    await store.remember(MemoryWrite(namespace="user:u1", text="Project Alpha uses COMSOL electrochemistry", importance=0.9))
    await store.remember(MemoryWrite(namespace="user:u2", text="Secret project Beta", importance=1.0))
    recalled = await store.recall("user:u1", "COMSOL project", top_k=3)
    assert recalled
    assert all(item.namespace == "user:u1" for item in recalled)
    assert "Alpha" in recalled[0].text


@pytest.mark.asyncio
async def test_long_term_memory_abstains_replaces_expires_and_deletes(tmp_path) -> None:
    store = MemoryStore(
        tmp_path / "memory.sqlite3",
        HashEmbedding(),
        message_limit=4,
        char_budget=1000,
        min_relevance_score=0.8,
    )
    old = await store.remember(
        MemoryWrite(
            namespace="user:u1",
            memory_key="preferred_solver",
            text="Use the segregated solver",
            importance=0.8,
        )
    )
    current = await store.remember(
        MemoryWrite(
            namespace="user:u1",
            memory_key="preferred_solver",
            text="Use the fully coupled solver",
            importance=0.8,
        )
    )
    await store.remember(
        MemoryWrite(
            namespace="user:u1",
            text="Expired COMSOL note",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    await store.remember(
        MemoryWrite(
            namespace="user:u1",
            text="Future COMSOL note",
            valid_from=datetime.now(UTC) + timedelta(days=1),
        )
    )

    recalled = await store.recall("user:u1", "Use the fully coupled solver", top_k=5)
    assert [item.memory_id for item in recalled] == [current.memory_id]
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        superseded_by = db.execute(
            "SELECT superseded_by FROM long_term_memories WHERE memory_id = ?",
            (old.memory_id,),
        ).fetchone()
    assert superseded_by == (current.memory_id,)
    assert recalled[0].relevance_score is not None
    assert await store.recall("user:u1", "Expired COMSOL note", top_k=5) == []
    assert await store.recall("user:u1", "Future COMSOL note", top_k=5) == []
    assert await store.recall("user:u1", "unrelated banana weather", top_k=5) == []
    assert await store.delete("user:u2", current.memory_id) is False
    assert await store.delete("user:u1", current.memory_id) is True
    assert await store.delete("user:u1", old.memory_id) is True


@pytest.mark.asyncio
async def test_memory_schema_v1_is_forward_migrated_and_reembedded(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """CREATE TABLE long_term_memories (
                memory_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                importance REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        now = datetime.now(UTC).isoformat()
        db.execute(
            """INSERT INTO long_term_memories
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            ("legacy", "user:u1", "legacy solver note", "[1.0]", 1.0, "{}", now, now),
        )
        db.commit()

    store = MemoryStore(
        db_path,
        HashEmbedding(),
        message_limit=4,
        char_budget=1000,
        min_relevance_score=0.2,
    )
    recalled = await store.recall("user:u1", "legacy solver note", top_k=1)

    assert recalled[0].memory_id == "legacy"
    assert recalled[0].embedding_profile == "hash-blake2b-test-only-v1:384"
    with sqlite3.connect(db_path) as db:
        version = db.execute(
            "SELECT schema_version FROM memory_schema_metadata WHERE component = 'memory'"
        ).fetchone()
        migrated = db.execute(
            "SELECT embedding_profile, updated_at, superseded_by "
            "FROM long_term_memories WHERE memory_id = 'legacy'"
        ).fetchone()
    assert version == (2,)
    assert migrated == ("hash-blake2b-test-only-v1:384", now, None)
