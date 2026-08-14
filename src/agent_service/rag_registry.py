from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiosqlite

_VERSION_UPSERT_SQL = """
    INSERT INTO {table}(
        version_id, source_id, tenant_id, acl_json, source_name,
        content_sha256, pipeline_profile, parser, chunks_count,
        warnings_json, status, error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'building', NULL)
    ON CONFLICT(version_id) DO UPDATE SET
        source_name=excluded.source_name,
        tenant_id=excluded.tenant_id,
        acl_json=excluded.acl_json,
        chunks_count=excluded.chunks_count,
        warnings_json=excluded.warnings_json,
        status=CASE
            WHEN {table}.status = 'active' THEN 'active'
            ELSE 'building'
        END,
        error=NULL,
        updated_at=CURRENT_TIMESTAMP
"""


@dataclass(slots=True, frozen=True)
class ActiveDocumentVersion:
    source_id: str
    version_id: str
    source_name: str
    content_sha256: str
    pipeline_profile: str
    parser: str
    chunks_count: int
    warnings: tuple[str, ...]
    tenant_id: str = "public"
    acl: tuple[str, ...] = ()


class VersionRegistry(Protocol):
    async def initialize(self) -> None: ...

    async def get_active(
        self, source_id: str, tenant_id: str = "public"
    ) -> ActiveDocumentVersion | None: ...

    async def list_active(
        self, source_ids: list[str] | None = None, tenant_id: str = "public"
    ) -> dict[str, ActiveDocumentVersion]: ...

    async def record_building(
        self,
        *,
        source_id: str,
        version_id: str,
        source_name: str,
        content_sha256: str,
        pipeline_profile: str,
        parser: str,
        chunks_count: int,
        warnings: list[str],
        tenant_id: str = "public",
        acl: list[str] | None = None,
    ) -> None: ...

    async def activate(self, version_id: str) -> ActiveDocumentVersion: ...

    async def mark_failed(self, version_id: str, error: str) -> None: ...


class SQLiteVersionRegistry:
    """Durable active-version pointers scoped by ``(tenant_id, source_id)``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
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
                await db.execute("PRAGMA foreign_keys=ON")
                await db.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS rag_sources (
                        source_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL DEFAULT 'public',
                        active_version_id TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS rag_document_versions (
                        version_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT 'public',
                        acl_json TEXT NOT NULL DEFAULT '[]',
                        source_name TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        pipeline_profile TEXT NOT NULL,
                        parser TEXT NOT NULL,
                        chunks_count INTEGER NOT NULL CHECK(chunks_count >= 0),
                        warnings_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL CHECK(status IN ('building', 'active', 'retired', 'failed')),
                        error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(source_id) REFERENCES rag_sources(source_id)
                    );
                    """
                )
                source_columns = await _table_columns(db, "rag_sources")
                if "tenant_id" not in source_columns:
                    await db.execute(
                        "ALTER TABLE rag_sources ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'public'"
                    )
                version_columns = await _table_columns(db, "rag_document_versions")
                if "tenant_id" not in version_columns:
                    await db.execute(
                        "ALTER TABLE rag_document_versions ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'public'"
                    )
                if "acl_json" not in version_columns:
                    await db.execute(
                        "ALTER TABLE rag_document_versions ADD COLUMN acl_json TEXT NOT NULL DEFAULT '[]'"
                    )
                await db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS rag_source_scopes (
                        tenant_id TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        active_version_id TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(tenant_id, source_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_sources_tenant
                        ON rag_sources(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_rag_versions_source
                        ON rag_document_versions(source_id);
                    CREATE INDEX IF NOT EXISTS idx_rag_versions_tenant
                        ON rag_document_versions(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_rag_versions_status
                        ON rag_document_versions(status);
                    """
                )
                # Preserve the original public-only tables during the migration window.
                # Their source rows become composite scopes without destructive rewriting.
                await db.execute(
                    """
                    INSERT INTO rag_source_scopes(
                        tenant_id, source_id, active_version_id, updated_at
                    )
                    SELECT tenant_id, source_id, active_version_id, updated_at
                    FROM rag_sources
                    WHERE 1
                    ON CONFLICT(tenant_id, source_id) DO UPDATE SET
                        active_version_id=excluded.active_version_id,
                        updated_at=excluded.updated_at
                    WHERE excluded.active_version_id IS NOT rag_source_scopes.active_version_id
                      AND excluded.updated_at >= rag_source_scopes.updated_at
                    """
                )
                # Defensive migration for registries that contain an orphan version row.
                await db.execute(
                    """
                    INSERT INTO rag_source_scopes(tenant_id, source_id)
                    SELECT DISTINCT tenant_id, source_id
                    FROM rag_document_versions
                    WHERE 1
                    ON CONFLICT(tenant_id, source_id) DO NOTHING
                    """
                )
                await db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS rag_document_versions_v2 (
                        version_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        acl_json TEXT NOT NULL DEFAULT '[]',
                        source_name TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        pipeline_profile TEXT NOT NULL,
                        parser TEXT NOT NULL,
                        chunks_count INTEGER NOT NULL CHECK(chunks_count >= 0),
                        warnings_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL CHECK(status IN ('building', 'active', 'retired', 'failed')),
                        error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(tenant_id, source_id)
                            REFERENCES rag_source_scopes(tenant_id, source_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_versions_v2_source
                        ON rag_document_versions_v2(tenant_id, source_id);
                    CREATE INDEX IF NOT EXISTS idx_rag_versions_v2_status
                        ON rag_document_versions_v2(status);
                    """
                )
                await db.execute(
                    """
                    INSERT INTO rag_document_versions_v2(
                        version_id, tenant_id, source_id, acl_json, source_name,
                        content_sha256, pipeline_profile, parser, chunks_count,
                        warnings_json, status, error, created_at, updated_at
                    )
                    SELECT version_id, tenant_id, source_id, acl_json, source_name,
                           content_sha256, pipeline_profile, parser, chunks_count,
                           warnings_json, status, error, created_at, updated_at
                    FROM rag_document_versions
                    WHERE 1
                    ON CONFLICT(version_id) DO UPDATE SET
                        acl_json=excluded.acl_json,
                        source_name=excluded.source_name,
                        chunks_count=excluded.chunks_count,
                        warnings_json=excluded.warnings_json,
                        status=excluded.status,
                        error=excluded.error,
                        updated_at=excluded.updated_at
                    WHERE excluded.tenant_id = 'public'
                      AND excluded.updated_at >= rag_document_versions_v2.updated_at
                    """
                )
                await db.commit()
            self._initialized = True

    async def get_active(
        self, source_id: str, tenant_id: str = "public"
    ) -> ActiveDocumentVersion | None:
        active = await self.list_active([source_id], tenant_id)
        return active.get(source_id)

    async def list_active(
        self, source_ids: list[str] | None = None, tenant_id: str = "public"
    ) -> dict[str, ActiveDocumentVersion]:
        await self.initialize()
        sql = """
            SELECT v.source_id, v.version_id, v.tenant_id, v.acl_json,
                   v.source_name, v.content_sha256, v.pipeline_profile,
                   v.parser, v.chunks_count, v.warnings_json
            FROM rag_source_scopes AS s
            JOIN rag_document_versions_v2 AS v
              ON v.version_id = s.active_version_id
             AND v.tenant_id = s.tenant_id
             AND v.source_id = s.source_id
            WHERE s.tenant_id = ?
        """
        params: list[str] = [tenant_id]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            sql += f" AND s.source_id IN ({placeholders})"
            params.extend(source_ids)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
        return {str(row["source_id"]): _row_to_version(row) for row in rows}

    async def record_building(
        self,
        *,
        source_id: str,
        version_id: str,
        source_name: str,
        content_sha256: str,
        pipeline_profile: str,
        parser: str,
        chunks_count: int,
        warnings: list[str],
        tenant_id: str = "public",
        acl: list[str] | None = None,
    ) -> None:
        await self.initialize()
        normalized_acl = sorted(set(acl or []))
        parameters = (
            version_id,
            source_id,
            tenant_id,
            json.dumps(normalized_acl, ensure_ascii=False),
            source_name,
            content_sha256,
            pipeline_profile,
            parser,
            chunks_count,
            json.dumps(warnings, ensure_ascii=False),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                """
                INSERT INTO rag_source_scopes(tenant_id, source_id)
                VALUES (?, ?)
                ON CONFLICT(tenant_id, source_id) DO NOTHING
                """,
                (tenant_id, source_id),
            )
            if tenant_id == "public":
                # Dual-write public records so rollback to the legacy reader remains viable.
                await db.execute(
                    """
                    INSERT INTO rag_sources(source_id, tenant_id)
                    VALUES (?, 'public')
                    ON CONFLICT(source_id) DO NOTHING
                    """,
                    (source_id,),
                )
            await db.execute(
                _VERSION_UPSERT_SQL.format(table="rag_document_versions_v2"),
                parameters,
            )
            if tenant_id == "public":
                await db.execute(
                    _VERSION_UPSERT_SQL.format(table="rag_document_versions"),
                    parameters,
                )
            await db.commit()

    async def activate(self, version_id: str) -> ActiveDocumentVersion:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            db.row_factory = sqlite3.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT * FROM rag_document_versions_v2 WHERE version_id = ?",
                (version_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                await db.rollback()
                raise KeyError(f"unknown document version: {version_id}")
            source_id = str(row["source_id"])
            tenant_id = str(row["tenant_id"])
            await db.execute(
                """
                UPDATE rag_document_versions_v2
                SET status = 'retired', updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = ? AND source_id = ?
                  AND status = 'active' AND version_id <> ?
                """,
                (tenant_id, source_id, version_id),
            )
            await db.execute(
                """
                UPDATE rag_document_versions_v2
                SET status = 'active', error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE version_id = ?
                """,
                (version_id,),
            )
            await db.execute(
                """
                UPDATE rag_source_scopes
                SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = ? AND source_id = ?
                """,
                (version_id, tenant_id, source_id),
            )
            if tenant_id == "public":
                await db.execute(
                    """
                    UPDATE rag_document_versions
                    SET status = 'retired', updated_at = CURRENT_TIMESTAMP
                    WHERE tenant_id = 'public' AND source_id = ?
                      AND status = 'active' AND version_id <> ?
                    """,
                    (source_id, version_id),
                )
                await db.execute(
                    """
                    UPDATE rag_document_versions
                    SET status = 'active', error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE version_id = ?
                    """,
                    (version_id,),
                )
                await db.execute(
                    """
                    UPDATE rag_sources
                    SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE source_id = ? AND tenant_id = 'public'
                    """,
                    (version_id, source_id),
                )
            await db.commit()
        active = await self._get_active_by_source(source_id, tenant_id)
        if active is None:
            raise RuntimeError(f"failed to activate document version: {version_id}")
        return active

    async def mark_failed(self, version_id: str, error: str) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            parameters = (error[:2000], version_id)
            await db.execute(
                """
                UPDATE rag_document_versions_v2
                SET status = CASE WHEN status = 'active' THEN status ELSE 'failed' END,
                    error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE version_id = ?
                """,
                parameters,
            )
            # The legacy table contains only public dual-writes and migrated records.
            await db.execute(
                """
                UPDATE rag_document_versions
                SET status = CASE WHEN status = 'active' THEN status ELSE 'failed' END,
                    error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE version_id = ?
                """,
                parameters,
            )
            await db.commit()

    async def _get_active_by_source(
        self, source_id: str, tenant_id: str
    ) -> ActiveDocumentVersion | None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT v.source_id, v.version_id, v.tenant_id, v.acl_json,
                       v.source_name, v.content_sha256, v.pipeline_profile,
                       v.parser, v.chunks_count, v.warnings_json
                FROM rag_source_scopes AS s
                JOIN rag_document_versions_v2 AS v
                  ON v.version_id = s.active_version_id
                 AND v.tenant_id = s.tenant_id
                 AND v.source_id = s.source_id
                WHERE s.tenant_id = ? AND s.source_id = ?
                """,
                (tenant_id, source_id),
            )
            row = await cursor.fetchone()
        return _row_to_version(row) if row is not None else None


class InMemoryVersionRegistry:
    """Test-friendly registry with tenant-scoped active-version semantics."""

    def __init__(self) -> None:
        self._versions: dict[str, ActiveDocumentVersion] = {}
        self._active: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def get_active(
        self, source_id: str, tenant_id: str = "public"
    ) -> ActiveDocumentVersion | None:
        version_id = self._active.get((tenant_id, source_id))
        return self._versions.get(version_id) if version_id else None

    async def list_active(
        self, source_ids: list[str] | None = None, tenant_id: str = "public"
    ) -> dict[str, ActiveDocumentVersion]:
        selected = set(source_ids) if source_ids else None
        return {
            source_id: self._versions[version_id]
            for (owner_tenant, source_id), version_id in self._active.items()
            if owner_tenant == tenant_id and (selected is None or source_id in selected)
        }

    async def record_building(
        self,
        *,
        source_id: str,
        version_id: str,
        source_name: str,
        content_sha256: str,
        pipeline_profile: str,
        parser: str,
        chunks_count: int,
        warnings: list[str],
        tenant_id: str = "public",
        acl: list[str] | None = None,
    ) -> None:
        async with self._lock:
            self._versions[version_id] = ActiveDocumentVersion(
                source_id=source_id,
                version_id=version_id,
                source_name=source_name,
                content_sha256=content_sha256,
                pipeline_profile=pipeline_profile,
                parser=parser,
                chunks_count=chunks_count,
                warnings=tuple(warnings),
                tenant_id=tenant_id,
                acl=tuple(sorted(set(acl or []))),
            )

    async def activate(self, version_id: str) -> ActiveDocumentVersion:
        async with self._lock:
            version = self._versions[version_id]
            self._active[(version.tenant_id, version.source_id)] = version_id
            return version

    async def mark_failed(self, version_id: str, error: str) -> None:
        del error
        if version_id not in self._versions:
            return


async def _table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in await cursor.fetchall()}


def _row_to_version(row: sqlite3.Row) -> ActiveDocumentVersion:
    warnings = tuple(str(item) for item in json.loads(row["warnings_json"]))
    acl = tuple(str(item) for item in json.loads(row["acl_json"]))
    return ActiveDocumentVersion(
        source_id=str(row["source_id"]),
        version_id=str(row["version_id"]),
        source_name=str(row["source_name"]),
        content_sha256=str(row["content_sha256"]),
        pipeline_profile=str(row["pipeline_profile"]),
        parser=str(row["parser"]),
        chunks_count=int(row["chunks_count"]),
        warnings=warnings,
        tenant_id=str(row["tenant_id"]),
        acl=acl,
    )
