from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_service.app import Services
from agent_service.backup import BackupManager
from agent_service.observability import Observability
from agent_service.schemas import DocumentIngestRequest
from agent_service.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "state" / "benchmark-200p"
WORK_ROOT = PROJECT_ROOT / "state" / "benchmark-observability-200p"
EXPECTED_OMNI = 50
EXPECTED_OHR = 150
EXPECTED_TOTAL = 200


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pages() -> list[dict[str, str]]:
    omni = json.loads((SOURCE_ROOT / "omni_manifest.json").read_text(encoding="utf-8"))
    ohr = json.loads((SOURCE_ROOT / "ohr_manifest.json").read_text(encoding="utf-8"))
    if len(omni) != EXPECTED_OMNI or len(ohr) != EXPECTED_OHR:
        raise RuntimeError(f"page budget mismatch: Omni={len(omni)}, OHR={len(ohr)}")
    pages = [
        {
            "sample_id": str(item["sample_id"]),
            "benchmark": "OmniDocBench",
            "stratum": str(item["stratum"]),
            "text": str(item["ground_truth_text"]),
            "source_hash": str(item["pdf_sha256"]),
        }
        for item in omni
    ]
    pages.extend(
        {
            "sample_id": str(item["sample_id"]),
            "benchmark": "OHR-Bench",
            "stratum": str(item["domain"]),
            "text": str(item["gt_text"]),
            "source_hash": hashlib.sha256(
                f"{item['doc_name']}:{item['page_idx']}".encode()
            ).hexdigest(),
        }
        for item in ohr
    )
    if len(pages) != EXPECTED_TOTAL or len({item["sample_id"] for item in pages}) != EXPECTED_TOTAL:
        raise RuntimeError(
            "the operational benchmark must contain exactly 200 unique PDF page identities"
        )
    if any(not item["text"].strip() for item in pages):
        raise RuntimeError("ground-truth text is missing for at least one selected page")
    return pages


def _prepare_workspace(pages: list[dict[str, str]], workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for item in pages:
        content = (
            f"Benchmark: {item['benchmark']}\n"
            f"Sample: {item['sample_id']}\n"
            f"Stratum: {item['stratum']}\n\n"
            f"{item['text'].strip()}\n"
        )
        (workspace / f"{item['sample_id']}.txt").write_text(content, encoding="utf-8")


def _query_for(text: str) -> str:
    words = " ".join(text.split()).split(" ")
    return " ".join(words[: min(24, len(words))])


async def _ingest_all(services: Services, pages: list[dict[str, str]]) -> dict[str, Any]:
    parser_counts: Counter[str] = Counter()
    chunks = 0
    idempotent = 0
    for item in pages:
        response = await services.rag.ingest(
            DocumentIngestRequest(
                file_path=f"{item['sample_id']}.txt",
                source_id=item["sample_id"],
                metadata={
                    "benchmark": item["benchmark"],
                    "stratum": item["stratum"],
                    "source_pdf_hash": item["source_hash"],
                },
            )
        )
        parser_counts[response.parser] += 1
        chunks += response.chunks_created
        idempotent += int(response.idempotent)
    return {
        "pages_ingested": len(pages),
        "chunks_created": chunks,
        "idempotent_responses": idempotent,
        "parser_counts": dict(parser_counts),
    }


async def _retrieval_signatures(
    services: Services, pages: list[dict[str, str]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    signatures: dict[str, list[str]] = {}
    hit_pages = 0
    degraded = 0
    latency: list[float] = []
    for item in pages:
        response = await services.rag.retrieve(
            _query_for(item["text"]),
            top_k=3,
            source_ids=[item["sample_id"]],
        )
        chunk_ids = [hit.citation.chunk_id or "" for hit in response.hits]
        signatures[item["sample_id"]] = chunk_ids
        hit_pages += int(bool(chunk_ids))
        degraded += int(response.degraded_retrieval)
        latency.append(response.latency_ms)
    return signatures, {
        "queries": len(pages),
        "pages_with_hits": hit_pages,
        "degraded_queries": degraded,
        "single_query_latency_ms": {
            "min": min(latency),
            "max": max(latency),
            "mean": sum(latency) / len(latency),
        },
    }


def _trace_summary(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    names = Counter(str(record.get("name")) for record in records)
    forbidden_keys = {"query", "prompt", "document_text", "chunk_text", "user_message"}
    leaked = [
        key for record in records for key in record.get("attributes", {}) if key in forbidden_keys
    ]
    return {
        "span_count": len(records),
        "span_names": dict(names),
        "unique_trace_ids": len({record.get("trace_id") for record in records}),
        "forbidden_content_attribute_occurrences": len(leaked),
        "trace_file_sha256": _sha256_file(path),
    }


def _sqlite_counts(state_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(state_dir / "rag.sqlite3")) as db:
        row = db.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
        counts["vector_chunks"] = int(row[0] if row else 0)
    with closing(sqlite3.connect(state_dir / "rag_registry.sqlite3")) as db:
        row = db.execute(
            "SELECT COUNT(*) FROM rag_source_scopes WHERE active_version_id IS NOT NULL"
        ).fetchone()
        counts["active_versions"] = int(row[0] if row else 0)
    return counts


async def run(test_node_id: str) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    stages: dict[str, float] = {}
    pages = _load_pages()
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    workspace = WORK_ROOT / "workspace"
    source_state = WORK_ROOT / "source-state"
    restored_state = WORK_ROOT / "restored-state"
    backups = WORK_ROOT / "backups"

    stage = time.perf_counter()
    _prepare_workspace(pages, workspace)
    stages["prepare_derived_canonical_text_seconds"] = time.perf_counter() - stage

    config = Settings(
        workspace_root=workspace,
        state_dir=source_state,
        embedding_provider="hash",
        vector_backend="sqlite",
        trace_jsonl_enabled=True,
        metrics_enabled=True,
        rag_query_max_parallel=2,
    )
    source = Services(config)
    try:
        stage = time.perf_counter()
        await source.initialize()
        stages["source_initialize_seconds"] = time.perf_counter() - stage

        stage = time.perf_counter()
        ingestion = await _ingest_all(source, pages)
        stages["sequential_ingestion_seconds"] = time.perf_counter() - stage

        stage = time.perf_counter()
        baseline_signatures, baseline_retrieval = await _retrieval_signatures(source, pages)
        stages["sequential_baseline_retrieval_seconds"] = time.perf_counter() - stage
        source_metrics_payload, _ = source.observability.render_metrics()
        source_metrics_text = source_metrics_payload.decode("utf-8")
    finally:
        await source.close()

    stage = time.perf_counter()
    backup_path = BackupManager(source_state).create(backups)
    stages["backup_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    verification = BackupManager.verify(backup_path)
    stages["backup_verification_seconds"] = time.perf_counter() - stage
    if not verification.valid:
        raise RuntimeError(f"backup verification failed: {verification.errors}")

    backup_observer = Observability(
        service_name="agentforge-benchmark-backup-status",
        service_version="0.3.0",
        environment="benchmark",
        state_dir=source_state,
        trace_jsonl_enabled=False,
    )
    backup_metrics_payload, _ = backup_observer.render_metrics()
    backup_metrics_text = backup_metrics_payload.decode("utf-8")
    await backup_observer.shutdown()

    stage = time.perf_counter()
    BackupManager.restore(backup_path, restored_state)
    stages["restore_seconds"] = time.perf_counter() - stage

    restored_config = Settings(
        workspace_root=workspace,
        state_dir=restored_state,
        embedding_provider="hash",
        vector_backend="sqlite",
        trace_jsonl_enabled=True,
        metrics_enabled=True,
    )
    restored = Services(restored_config)
    try:
        stage = time.perf_counter()
        await restored.initialize()
        stages["restored_initialize_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        restored_signatures, restored_retrieval = await _retrieval_signatures(restored, pages)
        stages["sequential_restored_retrieval_seconds"] = time.perf_counter() - stage
        restored_metrics_payload, _ = restored.observability.render_metrics()
        restored_metrics_text = restored_metrics_payload.decode("utf-8")
    finally:
        await restored.close()

    signature_mismatches = [
        sample_id
        for sample_id in baseline_signatures
        if baseline_signatures[sample_id] != restored_signatures.get(sample_id)
    ]
    required_metrics = [
        "agentforge_rag_operations_total",
        "agentforge_rag_operation_duration_seconds",
        "agentforge_ingestion_transitions_total",
        "agentforge_backup_last_success_unixtime",
        "agentforge_restore_verifications_total",
        "agentforge_trace_export_failures_total",
    ]
    metrics_text = source_metrics_text + "\n" + backup_metrics_text + "\n" + restored_metrics_text
    metrics_payload = metrics_text.encode("utf-8")
    metric_presence = {name: name in metrics_text for name in required_metrics}
    backup_metric_loaded = any(
        line.startswith("agentforge_backup_last_success_unixtime ")
        and float(line.rsplit(" ", 1)[1]) > 0
        for line in backup_metrics_text.splitlines()
    )
    restore_metric_loaded = (
        'agentforge_restore_verifications_total{outcome="success"} 1.0' in restored_metrics_text
    )
    trace = _trace_summary(source_state / "telemetry" / "traces.jsonl")
    source_counts = _sqlite_counts(source_state)
    restored_counts = _sqlite_counts(restored_state)
    manifest = json.loads((backup_path / "backup-manifest.json").read_text(encoding="utf-8"))

    result = {
        "schema_version": 1,
        "benchmark_name": "OmniDocBench + OHR-Bench RAG Benchmark v2 - Observability and Disaster Recovery",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "test_node": {
            "node_id": test_node_id,
            "physical_nodes_used": 1,
            "additional_test_machine_permitted": True,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "page_budget": {
            "total_unique_pdf_pages": len(pages),
            "OmniDocBench": sum(item["benchmark"] == "OmniDocBench" for item in pages),
            "OHR-Bench": sum(item["benchmark"] == "OHR-Bench" for item in pages),
            "within_budget": len(pages) == EXPECTED_TOTAL,
        },
        "scope": {
            "mode": "sequential functional and recovery correctness; no pressure test",
            "input_layer": "canonical ground-truth text derived from the fixed 200 PDF pages",
            "pdf_parser_quality_retested": False,
        },
        "ingestion": ingestion,
        "baseline_retrieval": baseline_retrieval,
        "restored_retrieval": restored_retrieval,
        "state_counts": {
            "source": source_counts,
            "restored": restored_counts,
            "exact_match": source_counts == restored_counts,
        },
        "metrics": {
            "required_family_presence": metric_presence,
            "all_required_present": all(metric_presence.values()),
            "backup_success_status_loaded": backup_metric_loaded,
            "restore_success_status_loaded": restore_metric_loaded,
            "snapshot_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        },
        "trace": trace,
        "backup": {
            "backup_id": verification.backup_id,
            "files_verified": verification.files_verified,
            "valid": verification.valid,
            "errors": verification.errors,
            "manifest_files": manifest["files"],
        },
        "restore": {
            "signature_mismatch_count": len(signature_mismatches),
            "signature_mismatch_samples": signature_mismatches[:20],
            "exact_retrieval_signature_match": not signature_mismatches,
        },
        "stage_timings": stages,
        "acceptance": {
            "page_budget_exactly_200": len(pages) == EXPECTED_TOTAL,
            "all_pages_ingested": ingestion["pages_ingested"] == EXPECTED_TOTAL,
            "all_baseline_pages_retrievable": baseline_retrieval["pages_with_hits"]
            == EXPECTED_TOTAL,
            "all_restored_pages_retrievable": restored_retrieval["pages_with_hits"]
            == EXPECTED_TOTAL,
            "metrics_complete": all(metric_presence.values()),
            "backup_metric_loaded": backup_metric_loaded,
            "restore_metric_loaded": restore_metric_loaded,
            "trace_content_safe": trace["forbidden_content_attribute_occurrences"] == 0,
            "backup_verified": verification.valid,
            "state_counts_exact": source_counts == restored_counts,
            "retrieval_signatures_exact": not signature_mismatches,
        },
    }
    results_dir = BENCHMARK_ROOT / "results"
    _write_json(results_dir / "operational_benchmark.json", result)
    (results_dir / "metrics_snapshot.prom").write_text(metrics_text, encoding="utf-8")
    _write_json(results_dir / "trace_summary.json", trace)
    _write_json(
        results_dir / "backup_manifest_sanitized.json",
        {
            "schema_version": manifest["schema_version"],
            "backup_id": manifest["backup_id"],
            "app_version": manifest["app_version"],
            "vector_backend": manifest["vector_backend"],
            "files": manifest["files"],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-node-id", default=os.getenv("AGENTFORGE_TEST_NODE_ID", "local-node-1")
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.test_node_id))
    print(json.dumps(result["acceptance"], ensure_ascii=False, indent=2))
    if not all(result["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
