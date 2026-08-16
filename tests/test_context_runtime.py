from __future__ import annotations

import hashlib
import json

import pytest

from agent_service.context_runtime import ContextBudgetManager, ToolResultSpillStore
from agent_service.schemas import ToolResult, ToolStatus


def test_context_budget_trims_old_turns_but_preserves_latest_tool_sequence() -> None:
    manager = ContextBudgetManager(max_input_tokens=1024, reserved_output_tokens=128)
    messages = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "old question " + "x" * 2500},
        {"role": "assistant", "content": "old answer " + "y" * 2500},
        {"role": "user", "content": "latest question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "lookup",
            "content": '{"summary":"bounded result"}',
        },
    ]

    fitted = manager.fit(messages)

    assert fitted.trimmed_units == 1
    assert fitted.messages[0]["role"] == "system"
    assert any(message.get("content") == "latest question" for message in fitted.messages)
    assert any(message.get("role") == "tool" for message in fitted.messages)
    assert not any("old question" in str(message.get("content")) for message in fitted.messages)


@pytest.mark.asyncio
async def test_large_tool_result_is_spilled_by_hash_with_valid_preview(tmp_path) -> None:
    store = ToolResultSpillStore(
        tmp_path / "tool-results",
        inline_token_limit=128,
        preview_chars=256,
    )
    result = ToolResult(
        call_id="call-1",
        tool_name="large_tool",
        status=ToolStatus.SUCCESS,
        data={"summary": "large result", "content": "evidence" * 3000},
        summary="large result",
        latency_ms=1.0,
    )

    rendered = await store.render(result)

    assert rendered.spilled is True
    assert rendered.artifact is not None
    payload = json.loads(rendered.content)
    assert payload["spilled"] is True
    assert payload["artifact"]["artifact_id"] == rendered.artifact.artifact_id
    artifact_path = store.path_for(rendered.artifact)
    serialized = artifact_path.read_text(encoding="utf-8")
    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == rendered.artifact.content_sha256
    stored = json.loads(serialized)
    assert stored["schema_version"] == 1
    assert stored["kind"] == "agentforge.tool_result"
    assert stored["tool_result"]["data"]["content"] == "evidence" * 3000


def test_context_budget_reports_when_latest_complete_turn_exceeds_budget() -> None:
    manager = ContextBudgetManager(max_input_tokens=1024, reserved_output_tokens=128)
    messages = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "latest " + "x" * 5000},
    ]

    fitted = manager.fit(messages)

    assert fitted.trimmed_units == 0
    assert fitted.over_budget is True
    assert fitted.messages == messages
