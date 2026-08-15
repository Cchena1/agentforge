from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from .document_pipeline import DocumentNeedsReviewError
from .rag import RAGService
from .schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    IngestionJobResponse,
    IngestionJobStatus,
)

ProgressCallback = Callable[[IngestionJobStatus], Awaitable[None]]
INGESTION_JOB_SCHEMA_VERSION = 1
_TERMINAL = {
    IngestionJobStatus.COMPLETED,
    IngestionJobStatus.FAILED,
    IngestionJobStatus.NEEDS_REVIEW,
    IngestionJobStatus.CANCELLED,
}


class IngestionJobManager:
    """Durable local job registry with bounded task ownership and restart recovery."""

    def __init__(self, db_path: Path, rag: RAGService, max_parallel: int = 2) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        self.db_path = db_path
        self.rag = rag
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._tasks: dict[str, asyncio.Task[None]] = {}
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
                    CREATE TABLE IF NOT EXISTS rag_ingestion_jobs (
                        job_id TEXT PRIMARY KEY,
                        request_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        result_json TEXT,
                        error TEXT,
                        warnings_json TEXT NOT NULL DEFAULT '[]',
                        schema_version INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_jobs_status
                    ON rag_ingestion_jobs(status);
                    """
                )
                cursor = await db.execute("PRAGMA table_info(rag_ingestion_jobs)")
                columns = {str(row[1]) for row in await cursor.fetchall()}
                if "schema_version" not in columns:
                    await db.execute(
                        "ALTER TABLE rag_ingestion_jobs "
                        "ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1"
                    )
                await db.commit()
                cursor = await db.execute(
                    """
                    SELECT job_id, request_json, schema_version FROM rag_ingestion_jobs
                    WHERE status IN ('queued', 'parsing', 'quality_check', 'fallback',
                                     'chunking', 'embedding', 'indexing', 'validating')
                    ORDER BY created_at
                    """
                )
                recoverable = await cursor.fetchall()
            self._initialized = True
            for job_id, request_json, schema_version in recoverable:
                if int(schema_version) != INGESTION_JOB_SCHEMA_VERSION:
                    await self._fail(
                        str(job_id),
                        IngestionJobStatus.FAILED,
                        f"unsupported ingestion job schema version: {schema_version}",
                    )
                    continue
                request = DocumentIngestRequest.model_validate_json(str(request_json))
                await self._set_state(str(job_id), IngestionJobStatus.QUEUED, "queued")
                self._start(str(job_id), request)

    async def create(self, request: DocumentIngestRequest) -> IngestionJobResponse:
        await self.initialize()
        job_id = f"ij_{uuid.uuid4().hex}"
        now = datetime.now(UTC)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO rag_ingestion_jobs(
                    job_id, request_json, status, stage, schema_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.model_dump_json(),
                    IngestionJobStatus.QUEUED.value,
                    "queued",
                    INGESTION_JOB_SCHEMA_VERSION,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await db.commit()
        self._start(job_id, request)
        return await self.get(job_id)

    async def get(self, job_id: str) -> IngestionJobResponse:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT * FROM rag_ingestion_jobs WHERE job_id = ?", (job_id,)
            )
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(job_id)
        result = (
            DocumentIngestResponse.model_validate_json(row["result_json"])
            if row["result_json"]
            else None
        )
        return IngestionJobResponse(
            job_id=str(row["job_id"]),
            status=IngestionJobStatus(str(row["status"])),
            stage=str(row["stage"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            result=result,
            error=str(row["error"]) if row["error"] else None,
            warnings=[str(item) for item in json.loads(row["warnings_json"])],
        )

    async def cancel(self, job_id: str) -> IngestionJobResponse:
        current = await self.get(job_id)
        if current.status in _TERMINAL:
            return current
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
        await self._set_state(job_id, IngestionJobStatus.CANCELLED, "cancelled")
        return await self.get(job_id)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start(self, job_id: str, request: DocumentIngestRequest) -> None:
        task = asyncio.create_task(self._run(job_id, request), name=f"rag-ingest:{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(job_id, None))

    async def _run(self, job_id: str, request: DocumentIngestRequest) -> None:
        try:
            async with self._semaphore:
                async def progress(status: IngestionJobStatus) -> None:
                    await self._set_state(job_id, status, status.value)

                result = await self.rag.ingest(request, progress=progress)
                await self._complete(job_id, result)
        except asyncio.CancelledError:
            await self._set_state(job_id, IngestionJobStatus.CANCELLED, "cancelled")
            raise
        except DocumentNeedsReviewError as exc:
            await self._fail(job_id, IngestionJobStatus.NEEDS_REVIEW, str(exc))
        except Exception as exc:  # noqa: BLE001 - durable job boundary records failures
            await self._fail(job_id, IngestionJobStatus.FAILED, f"{type(exc).__name__}: {exc}")

    async def _complete(self, job_id: str, result: DocumentIngestResponse) -> None:
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = ?, stage = ?, result_json = ?, error = NULL,
                    warnings_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    IngestionJobStatus.COMPLETED.value,
                    "completed",
                    result.model_dump_json(),
                    json.dumps(result.warnings, ensure_ascii=False),
                    now,
                    job_id,
                ),
            )
            await db.commit()

    async def _fail(
        self, job_id: str, status: IngestionJobStatus, error: str
    ) -> None:
        await self._set_state(job_id, status, status.value, error=error)

    async def _set_state(
        self,
        job_id: str,
        status: IngestionJobStatus,
        stage: str,
        *,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = ?, stage = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status.value, stage, error, now, job_id),
            )
            await db.commit()
