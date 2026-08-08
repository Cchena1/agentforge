from __future__ import annotations

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
