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


def test_document_ingest_adds_version_fields_without_removing_legacy_fields(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("versioned evidence", encoding="utf-8")
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        first = client.post(
            "/documents/ingest",
            json={"file_path": "guide.md", "source_id": "guide"},
        )
        second = client.post(
            "/documents/ingest",
            json={"file_path": "guide.md", "source_id": "guide"},
        )
        search = client.post(
            "/documents/search",
            json={"query": "versioned evidence", "source_ids": ["guide"]},
        )

    assert first.status_code == 200
    first_payload = first.json()
    assert {"source_id", "source_name", "chunks_created", "parser", "warnings"} <= set(
        first_payload
    )
    assert first_payload["version_id"]
    assert first_payload["content_sha256"]
    assert first_payload["idempotent"] is False

    assert second.status_code == 200
    assert second.json()["version_id"] == first_payload["version_id"]
    assert second.json()["idempotent"] is True

    assert search.status_code == 200
    assert search.json()["index_versions"] == {"guide": first_payload["version_id"]}


def test_document_api_additively_supports_tenant_and_acl_filters(tmp_path) -> None:
    (tmp_path / "restricted.md").write_text("restricted engineering evidence", encoding="utf-8")
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        ingest = client.post(
            "/documents/ingest",
            json={
                "file_path": "restricted.md",
                "source_id": "restricted",
                "tenant_id": "tenant-a",
                "acl": ["group:engineering"],
            },
        )
        anonymous = client.post(
            "/documents/search",
            json={"query": "engineering evidence", "tenant_id": "tenant-a"},
        )
        authorized = client.post(
            "/documents/search",
            json={
                "query": "engineering evidence",
                "tenant_id": "tenant-a",
                "principals": ["group:engineering"],
            },
        )
        other_tenant = client.post(
            "/documents/search",
            json={"query": "engineering evidence", "tenant_id": "tenant-b"},
        )

    assert ingest.status_code == 200
    assert anonymous.status_code == 200
    assert anonymous.json()["hits"] == []
    assert anonymous.json()["index_versions"] == {}
    assert authorized.status_code == 200
    assert authorized.json()["hits"]
    assert authorized.json()["index_versions"] == {
        "restricted": ingest.json()["version_id"]
    }
    assert other_tenant.status_code == 200
    assert other_tenant.json()["hits"] == []
    assert other_tenant.json()["index_versions"] == {}


def test_document_api_scopes_same_source_id_by_tenant(tmp_path) -> None:
    (tmp_path / "first.md").write_text("first tenant evidence", encoding="utf-8")
    (tmp_path / "second.md").write_text("second tenant evidence", encoding="utf-8")
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        first = client.post(
            "/documents/ingest",
            json={"file_path": "first.md", "source_id": "shared", "tenant_id": "tenant-a"},
        )
        second = client.post(
            "/documents/ingest",
            json={"file_path": "second.md", "source_id": "shared", "tenant_id": "tenant-b"},
        )
        search_a = client.post(
            "/documents/search",
            json={"query": "tenant evidence", "source_ids": ["shared"], "tenant_id": "tenant-a"},
        )
        search_b = client.post(
            "/documents/search",
            json={"query": "tenant evidence", "source_ids": ["shared"], "tenant_id": "tenant-b"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert search_a.json()["index_versions"] == {"shared": first.json()["version_id"]}
    assert search_b.json()["index_versions"] == {"shared": second.json()["version_id"]}
    assert all("first tenant" in hit["text"] for hit in search_a.json()["hits"])
    assert all("second tenant" in hit["text"] for hit in search_b.json()["hits"])


def test_document_authorization_context_rejects_whitespace_principals(tmp_path) -> None:
    config = Settings(workspace_root=tmp_path, state_dir=tmp_path / "state", api_key="")
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/documents/search",
            json={"query": "evidence", "tenant_id": "tenant-a", "principals": ["group bad"]},
        )

    assert response.status_code == 422
