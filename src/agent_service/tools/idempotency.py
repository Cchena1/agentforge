from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import aiosqlite

from ..schemas import ToolResult


class IdempotencyState(StrEnum):
    RESERVED = "reserved"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    INDETERMINATE = "indeterminate"
    CONFLICT = "conflict"


@dataclass(slots=True, frozen=True)
class IdempotencyReservation:
    state: IdempotencyState
    result: ToolResult | None = None


class ToolIdempotencyStore(Protocol):
    async def reserve(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        request_fingerprint: str,
    ) -> IdempotencyReservation: ...

    async def complete(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        result: ToolResult,
    ) -> None: ...

    async def mark_indeterminate(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
    ) -> None: ...


class InMemoryToolIdempotencyStore:
    """Process-local implementation for tests and embedded library callers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: dict[tuple[str, str], tuple[str, str, str, ToolResult | None]] = {}

    async def reserve(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        request_fingerprint: str,
    ) -> IdempotencyReservation:
        key = (tool_name, idempotency_key)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = (
                    IdempotencyState.IN_PROGRESS,
                    owner_token,
                    request_fingerprint,
                    None,
                )
                return IdempotencyReservation(IdempotencyState.RESERVED)
            state, _owner, existing_fingerprint, result = entry
            if existing_fingerprint != request_fingerprint:
                return IdempotencyReservation(IdempotencyState.CONFLICT)
            if state == IdempotencyState.COMPLETED:
                return IdempotencyReservation(IdempotencyState.COMPLETED, result)
            if state == IdempotencyState.INDETERMINATE:
                return IdempotencyReservation(IdempotencyState.INDETERMINATE)
            return IdempotencyReservation(IdempotencyState.IN_PROGRESS)

    async def complete(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        result: ToolResult,
    ) -> None:
        key = (tool_name, idempotency_key)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] != IdempotencyState.IN_PROGRESS or entry[1] != owner_token:
                raise RuntimeError("idempotency reservation ownership was lost")
            self._entries[key] = (
                IdempotencyState.COMPLETED,
                owner_token,
                entry[2],
                result,
            )

    async def mark_indeterminate(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
    ) -> None:
        key = (tool_name, idempotency_key)
        async with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry[0] == IdempotencyState.IN_PROGRESS
                and entry[1] == owner_token
            ):
                self._entries[key] = (
                    IdempotencyState.INDETERMINATE,
                    owner_token,
                    entry[2],
                    None,
                )


class SQLiteToolIdempotencyStore:
    """Durable local ledger for write-tool result replay and fail-closed recovery."""

    def __init__(self, db_path: Path, *, in_progress_ttl_seconds: float) -> None:
        self.db_path = db_path
        self.in_progress_ttl = timedelta(seconds=max(in_progress_ttl_seconds, 1.0))
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False

    async def reserve(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        request_fingerprint: str,
    ) -> IdempotencyReservation:
        await self._ensure_schema()
        now = datetime.now(UTC)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                """
                SELECT status, request_fingerprint, result_json, updated_at
                FROM tool_idempotency
                WHERE tool_name = ? AND idempotency_key = ?
                """,
                (tool_name, idempotency_key),
            )
            row = await cursor.fetchone()
            if row is None:
                await db.execute(
                    """
                    INSERT INTO tool_idempotency(
                        tool_name, idempotency_key, status, owner_token,
                        request_fingerprint, result_json, created_at, updated_at
                    ) VALUES (?, ?, 'in_progress', ?, ?, NULL, ?, ?)
                    """,
                    (
                        tool_name,
                        idempotency_key,
                        owner_token,
                        request_fingerprint,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
                await db.commit()
                return IdempotencyReservation(IdempotencyState.RESERVED)

            status, existing_fingerprint, result_json, updated_at = row
            if existing_fingerprint != request_fingerprint:
                await db.commit()
                return IdempotencyReservation(IdempotencyState.CONFLICT)
            if status == IdempotencyState.COMPLETED:
                await db.commit()
                return IdempotencyReservation(
                    IdempotencyState.COMPLETED,
                    ToolResult.model_validate_json(result_json),
                )
            if status == IdempotencyState.INDETERMINATE:
                await db.commit()
                return IdempotencyReservation(IdempotencyState.INDETERMINATE)
            if _is_expired(updated_at, now, self.in_progress_ttl):
                await db.execute(
                    """
                    UPDATE tool_idempotency
                    SET status = 'indeterminate', updated_at = ?
                    WHERE tool_name = ? AND idempotency_key = ?
                    """,
                    (now.isoformat(), tool_name, idempotency_key),
                )
                await db.commit()
                return IdempotencyReservation(IdempotencyState.INDETERMINATE)
            await db.commit()
            return IdempotencyReservation(IdempotencyState.IN_PROGRESS)

    async def complete(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
        result: ToolResult,
    ) -> None:
        await self._ensure_schema()
        result_json = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE tool_idempotency
                SET status = 'completed', result_json = ?, updated_at = ?
                WHERE tool_name = ? AND idempotency_key = ?
                  AND status = 'in_progress' AND owner_token = ?
                """,
                (
                    result_json,
                    datetime.now(UTC).isoformat(),
                    tool_name,
                    idempotency_key,
                    owner_token,
                ),
            )
            await db.commit()
            if cursor.rowcount != 1:
                raise RuntimeError("idempotency reservation ownership was lost")

    async def mark_indeterminate(
        self,
        tool_name: str,
        idempotency_key: str,
        owner_token: str,
    ) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE tool_idempotency
                SET status = 'indeterminate', updated_at = ?
                WHERE tool_name = ? AND idempotency_key = ?
                  AND status = 'in_progress' AND owner_token = ?
                """,
                (
                    datetime.now(UTC).isoformat(),
                    tool_name,
                    idempotency_key,
                    owner_token,
                ),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_idempotency (
                        tool_name TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        owner_token TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        result_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(tool_name, idempotency_key)
                    )
                    """
                )
                await db.commit()
            self._schema_ready = True


def _is_expired(updated_at: str, now: datetime, ttl: timedelta) -> bool:
    try:
        timestamp = datetime.fromisoformat(updated_at)
    except ValueError:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return now - timestamp.astimezone(UTC) >= ttl
