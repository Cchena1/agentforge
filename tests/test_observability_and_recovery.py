from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_service.app import Services, create_app
from agent_service.backup import BackupError, BackupManager, StateDirectoryLock
from agent_service.observability import Observability
from agent_service.settings import Settings


def _settings(tmp_path: Path, state_name: str = "state") -> Settings:
    return Settings(
        workspace_root=tmp_path,
        state_dir=tmp_path / state_name,
        trace_jsonl_enabled=True,
        metrics_enabled=True,
    )


def test_metrics_trace_header_and_local_trace_evidence(tmp_path: Path) -> None:
    config = _settings(tmp_path)
    app = create_app(config)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert len(health.headers["X-Trace-ID"]) == 32
        assert client.get("/ready").json()["status"] == "ready"
        app.state.services.observability.record_tool_runtime_event(
            "policy", "denied", "write"
        )
        app.state.services.observability.record_model_usage(
            "primary",
            "success",
            degraded=False,
            prompt_tokens=100,
            completion_tokens=20,
            cached_prompt_tokens=10,
            reasoning_tokens=5,
            estimated_cost_usd=0.001,
        )
        app.state.services.observability.record_failure("model", "timeout")
        app.state.services.observability.record_memory_operation("recall", "abstained")
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "agentforge_http_requests_total" in metrics.text
        assert 'route="/health"' in metrics.text
        assert "agentforge_tool_runtime_events_total" in metrics.text
        assert 'phase="policy"' in metrics.text
        assert 'outcome="denied"' in metrics.text
        assert "agentforge_model_tokens_total" in metrics.text
        assert 'token_class="prompt"' in metrics.text
        assert "agentforge_model_estimated_cost_usd_total" in metrics.text
        assert 'component="model"' in metrics.text
        assert 'operation="recall"' in metrics.text
        with pytest.raises(BackupError, match="state directory is in use"):
            BackupManager(config.resolved_state_dir).create(tmp_path / "backups")

    trace_path = config.resolved_state_dir / "telemetry" / "traces.jsonl"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(record["name"] == "GET /health" for record in records)
    assert all("query" not in record["attributes"] for record in records)


def test_verified_backup_restore_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    with sqlite3.connect(state / "rag.sqlite3") as db:
        db.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        db.execute("INSERT INTO evidence(text) VALUES ('stable citation')")
        db.commit()
    with sqlite3.connect(state / "memory.sqlite3") as db:
        db.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        db.execute("INSERT INTO memories(text) VALUES ('layered memory')")
        db.commit()

    backup = BackupManager(state).create(tmp_path / "backups")
    verification = BackupManager.verify(backup)
    assert verification.valid is True
    assert verification.files_verified == 2

    restored = BackupManager.restore(backup, tmp_path / "restored")
    with sqlite3.connect(restored / "rag.sqlite3") as db:
        assert db.execute("SELECT text FROM evidence").fetchone() == ("stable citation",)
    status = json.loads((restored / "operations" / "status.json").read_text(encoding="utf-8"))
    assert status["last_restore_verification"] == "success"

    tampered = backup / "data" / "rag.sqlite3"
    with tampered.open("ab") as stream:
        stream.write(b"tamper")
    invalid = BackupManager.verify(backup)
    assert invalid.valid is False
    assert any(error.startswith(("size_mismatch", "hash_mismatch")) for error in invalid.errors)


@pytest.mark.asyncio
async def test_trace_jsonl_rotates_and_constructor_failure_releases_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = Observability(
        service_name="rotation-test",
        service_version="test",
        environment="test",
        state_dir=tmp_path / "trace-state",
        trace_jsonl_max_bytes=512,
        trace_jsonl_backup_count=2,
    )
    for index in range(20):
        with observer.span("rotation.span", {"index": index}):
            pass
    await observer.shutdown()
    trace_dir = tmp_path / "trace-state" / "telemetry"
    assert (trace_dir / "traces.jsonl").exists()
    assert (trace_dir / "traces.jsonl.1").exists()
    assert len(list(trace_dir.glob("traces.jsonl*"))) <= 3

    config = _settings(tmp_path, "locked-state")
    config.vector_backend = "qdrant"

    def fail_qdrant(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr("agent_service.app.QdrantVectorStore", fail_qdrant)
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        Services(config)
    with StateDirectoryLock(config.resolved_state_dir):
        pass


def test_qdrant_backup_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="Qdrant backup is fail-closed"):
        BackupManager(tmp_path / "state", vector_backend="qdrant").create(tmp_path / "backups")
