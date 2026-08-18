# Long-term Memory Schema v2 Migration Guide

## Purpose

Version 0.4.0 introduces an additive long-term Memory migration. The goal is to prevent stale or conflicting memories from being silently returned while preserving existing data and clients.

## Changed persistence fields

The `long_term_memories` table adds nullable columns:

- `memory_key`;
- `valid_from`;
- `expires_at`;
- `superseded_at`;
- `superseded_by`;
- `embedding_profile`.

A `memory_schema_metadata` table records schema version `2`. Existing records are backfilled with `valid_from = created_at`; their embeddings are lazily refreshed when the configured embedding profile or vector dimension differs.

## Public compatibility window

This release is the migration stage:

1. Existing `POST /memory` payloads remain valid.
2. New clients may send `memory_key`, `valid_from`, and `expires_at`.
3. New clients may use namespace-scoped `DELETE /memory/{memory_id}?namespace=...`.
4. Existing rows are not deleted during migration.
5. Removal of legacy semantics requires a separately announced major-version deprecation and deletion stage.

## Required pre-migration procedure

Stop the service and create a verified backup:

```powershell
uv run python -m agent_service.operations backup --state-dir .\state --output-dir E:\agentforge-backups
uv run python -m agent_service.operations verify --backup-dir E:\agentforge-backups\afb_...
```

Retain the verified backup outside the active state directory. The backup workflow uses SQLite Backup API, SHA-256 verification, `PRAGMA integrity_check`, and an isolated restore target.

## Upgrade procedure

```powershell
uv sync --all-groups
uv run uvicorn agent_service.app:create_app --factory --host 127.0.0.1 --port 8000
```

On first Memory access, initialization performs the additive migration in this order:

1. create the base table if absent;
2. inspect existing columns;
3. add missing columns;
4. backfill `valid_from`;
5. create the active-memory lookup index;
6. record schema version `2`.

This order intentionally supports existing v1 databases where the new columns do not yet exist.

## Validation

```powershell
uv run pytest tests/test_memory.py tests/test_api.py -p no:cacheprovider
uv run python -m mypy src/agent_service
uv run ruff check src tests --no-cache
```

Verify these behaviors:

- a repeated `memory_key` supersedes the previous active record;
- expired and future-dated records are not recalled;
- low-relevance queries return no long-term Memory;
- stale embedding profiles are refreshed before scoring;
- deletion requires the matching namespace;
- service metrics expose Memory operation outcomes without storing text.

## Rollback

Do not run an older binary against a state directory already used by v0.4.0. For application rollback:

1. stop all AgentForge processes;
2. verify the pre-migration backup again;
3. restore to a separate directory;
4. run smoke tests against that directory;
5. promote the restored directory only after validation.

```powershell
uv run python -m agent_service.operations restore --backup-dir E:\agentforge-backups\afb_... --target-state-dir .\state-restored
```

The restore command does not overwrite a non-empty target unless `--replace-existing` is explicitly supplied.

## Known limits

- Memory deletion is hard deletion in the local store; enterprise retention, legal hold, export, consent, and PII workflows remain deployment policy.
- Recall is namespace-scoped but not an authentication boundary; trusted middleware must derive the namespace.
- Lazy re-embedding increases the first recall latency after an embedding-profile change.
- Large-scale namespaces require an indexed Vector Store migration backed by measured workload evidence.
