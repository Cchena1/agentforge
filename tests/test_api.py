from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.app import create_app
from agent_service.settings import Settings


def test_backward_compatible_chat_contract_and_offline_fallback(tmp_path) -> None:
    (tmp_path / "test.txt").write_text("retry fallback memory", encoding="utf-8")
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        response = client.post("/chat", json={"message": "读取 test.txt", "history": []})
    assert response.status_code == 200
    payload = response.json()
    assert "reply" in payload
    assert "tool_calls" in payload
    assert payload["tool_calls"][0]["name"] == "read_file"
    assert payload["tool_calls"][0]["tool_name"] == "read_file"
    assert "result" in payload["tool_calls"][0]
    assert payload["degraded"] is True


def test_path_escape_is_rejected(tmp_path) -> None:
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        response = client.post("/documents/ingest", json={"file_path": "../outside.txt"})
    assert response.status_code == 400
