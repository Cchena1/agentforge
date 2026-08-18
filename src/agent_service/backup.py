from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

BACKUP_SCHEMA_VERSION = 1
APP_VERSION = "0.4.0"


class BackupError(RuntimeError):
    pass


class BackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(pattern=r"^data/[A-Za-z0-9_.-]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    kind: Literal["sqlite"] = "sqlite"


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = BACKUP_SCHEMA_VERSION
    backup_id: str
    created_at: datetime
    app_version: str = APP_VERSION
    vector_backend: Literal["sqlite", "qdrant"]
    source_state_dir: str
    files: list[BackupFile]


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backup_id: str
    valid: bool
    files_verified: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class StateDirectoryLock:
    """Cross-process lock shared by the service and offline backup/restore CLI."""

    def __init__(self, state_dir: Path) -> None:
        resolved = state_dir.expanduser().resolve()
        self.path = resolved.parent / f".{resolved.name}.agentforge.lock"
        self._stream: BinaryIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as exc:
            stream.close()
            raise BackupError(
                f"state directory is in use: {self.path.parent}; stop AgentForge before backup or restore"
            ) from exc
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        stream.close()
        self._stream = None

    def __enter__(self) -> StateDirectoryLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class BackupManager:
    """Verified offline backup and restore for the default SQLite deployment."""

    def __init__(
        self, state_dir: Path, *, vector_backend: Literal["sqlite", "qdrant"] = "sqlite"
    ) -> None:
        self.state_dir = state_dir.expanduser().resolve()
        self.vector_backend = vector_backend

    def create(self, output_dir: Path) -> Path:
        if self.vector_backend != "sqlite":
            raise BackupError(
                "Qdrant backup is fail-closed in v0.4; use Qdrant collection snapshots before backing up app state"
            )
        output_root = output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        backup_id = f"afb_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        final = output_root / backup_id
        if final.exists():
            raise BackupError(f"backup target already exists: {final}")
        with StateDirectoryLock(self.state_dir):
            sqlite_files = sorted(self.state_dir.glob("*.sqlite3"))
            if not sqlite_files:
                raise BackupError(f"no SQLite state files found in {self.state_dir}")
            staging = Path(tempfile.mkdtemp(prefix=f".{backup_id}-", dir=output_root))
            try:
                data_dir = staging / "data"
                data_dir.mkdir()
                records: list[BackupFile] = []
                for source in sqlite_files:
                    if source.is_symlink() or not source.is_file():
                        raise BackupError(f"refusing non-regular state file: {source}")
                    destination = data_dir / source.name
                    _sqlite_backup(source, destination)
                    _assert_sqlite_integrity(destination)
                    records.append(
                        BackupFile(
                            relative_path=f"data/{source.name}",
                            sha256=_sha256_file(destination),
                            size_bytes=destination.stat().st_size,
                        )
                    )
                manifest = BackupManifest(
                    backup_id=backup_id,
                    created_at=datetime.now(UTC),
                    vector_backend=self.vector_backend,
                    source_state_dir=str(self.state_dir),
                    files=records,
                )
                (staging / "backup-manifest.json").write_text(
                    manifest.model_dump_json(indent=2), encoding="utf-8"
                )
                os.rename(staging, final)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            _update_operations_status(
                self.state_dir,
                last_backup_success_unixtime=time.time(),
                last_backup_id=backup_id,
            )
        return final

    @staticmethod
    def verify(backup_dir: Path) -> VerificationResult:
        root = backup_dir.expanduser().resolve()
        errors: list[str] = []
        manifest_path = root / "backup-manifest.json"
        try:
            manifest = BackupManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return VerificationResult(
                backup_id="unknown",
                valid=False,
                files_verified=0,
                errors=[f"manifest_invalid:{type(exc).__name__}"],
            )
        if manifest.schema_version != BACKUP_SCHEMA_VERSION:
            errors.append(f"unsupported_schema_version:{manifest.schema_version}")
        verified = 0
        for record in manifest.files:
            path = (root / record.relative_path).resolve()
            if root not in path.parents or path.is_symlink() or not path.is_file():
                errors.append(f"unsafe_or_missing_file:{record.relative_path}")
                continue
            if path.stat().st_size != record.size_bytes:
                errors.append(f"size_mismatch:{record.relative_path}")
                continue
            if _sha256_file(path) != record.sha256:
                errors.append(f"hash_mismatch:{record.relative_path}")
                continue
            try:
                _assert_sqlite_integrity(path)
            except BackupError:
                errors.append(f"sqlite_integrity_failed:{record.relative_path}")
                continue
            verified += 1
        if not manifest.files:
            errors.append("empty_backup")
        return VerificationResult(
            backup_id=manifest.backup_id,
            valid=not errors and verified == len(manifest.files),
            files_verified=verified,
            errors=errors,
        )

    @staticmethod
    def restore(
        backup_dir: Path, target_state_dir: Path, *, replace_existing: bool = False
    ) -> Path:
        source = backup_dir.expanduser().resolve()
        target = target_state_dir.expanduser().resolve()
        if target.parent == target or source == target or source in target.parents:
            raise BackupError("unsafe restore target")
        result = BackupManager.verify(source)
        if not result.valid:
            raise BackupError(f"backup verification failed: {', '.join(result.errors)}")
        manifest = BackupManifest.model_validate_json(
            (source / "backup-manifest.json").read_text(encoding="utf-8")
        )
        with StateDirectoryLock(target):
            if target.exists() and any(target.iterdir()) and not replace_existing:
                raise BackupError("restore target is not empty; use --replace-existing explicitly")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-restore-", dir=target.parent))
            rollback: Path | None = None
            try:
                for record in manifest.files:
                    destination = staging / Path(record.relative_path).name
                    shutil.copy2(source / record.relative_path, destination)
                    _assert_sqlite_integrity(destination)
                (staging / "restore-manifest.json").write_text(
                    manifest.model_dump_json(indent=2), encoding="utf-8"
                )
                if target.exists():
                    rollback = target.parent / f".{target.name}-rollback-{uuid.uuid4().hex[:8]}"
                    os.rename(target, rollback)
                os.rename(staging, target)
                if rollback is not None:
                    shutil.rmtree(rollback)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                if rollback is not None and rollback.exists() and not target.exists():
                    os.rename(rollback, target)
                raise
            _update_operations_status(
                target,
                last_restore_verification="success",
                last_restore_backup_id=manifest.backup_id,
                last_restore_success_unixtime=time.time(),
            )
        return target


def _sqlite_backup(source: Path, destination: Path) -> None:
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    try:
        with (
            closing(sqlite3.connect(source_uri, uri=True)) as source_db,
            closing(sqlite3.connect(destination)) as dest_db,
        ):
            source_db.backup(dest_db)
    except sqlite3.Error as exc:
        raise BackupError(f"SQLite backup failed for {source.name}: {exc}") from exc


def _assert_sqlite_integrity(path: Path) -> None:
    try:
        with closing(sqlite3.connect(path)) as db:
            rows = [str(row[0]) for row in db.execute("PRAGMA integrity_check").fetchall()]
    except sqlite3.Error as exc:
        raise BackupError(f"SQLite integrity check failed for {path.name}: {exc}") from exc
    if rows != ["ok"]:
        raise BackupError(f"SQLite integrity check failed for {path.name}: {rows[:3]}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_operations_status(state_dir: Path, **values: object) -> None:
    operations = state_dir / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    path = operations / "status.json"
    payload: dict[str, object] = {"schema_version": 1}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(current, dict):
                payload.update(current)
        except (OSError, json.JSONDecodeError):
            pass
    payload.update(values)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
