# RAG Versioning and Migration Guide

**Status:** Active migration window
**Introduced:** 2026-08-13
**Applies to:** AgentForge 0.x local SQLite and optional Qdrant RAG backends

## Why this migration exists

The previous ingestion path deleted the currently indexed chunks before the replacement version was fully parsed, embedded, and stored. A parser, embedding, or vector-store failure could therefore make a previously searchable source disappear.

The new design treats document versions as immutable build artifacts and makes a small SQLite registry the visibility authority. Retrieval sees only the active version of a managed source. Activation occurs only after every chunk has been written successfully.

## Persistent-format changes

### SQLite vector store

`document_chunks` gains additive visibility and authorization columns:

```sql
version_id TEXT NULL
tenant_id TEXT NOT NULL DEFAULT 'public'
acl_json TEXT NOT NULL DEFAULT '[]'
```

Startup performs an additive, in-place migration:

1. Create the table if it does not exist.
2. Inspect columns with `PRAGMA table_info(document_chunks)`.
3. Add `version_id`, `tenant_id`, and `acl_json` only when absent.
4. Create version and tenant indexes only after their columns exist.

No existing rows are deleted. SQLite defaults migrate legacy rows to tenant `public` with an empty, tenant-wide ACL. Registry initialization creates composite scopes before copying legacy version rows into `rag_document_versions_v2`, enables foreign-key checks on write connections, and leaves the original tables intact.

### Version registry

A new state file is created at:

```text
<AI_AGENT_STATE_DIR>/rag_registry.sqlite3
```

It owns:

- `rag_source_scopes.active_version_id`: the visibility pointer per `(tenant_id, source_id)`;
- `rag_document_versions_v2`: the current immutable version table, keyed by version ID and constrained to composite `(tenant_id, source_id)` scopes;
- legacy `rag_sources` and `rag_document_versions`: retained for the 0.x migration window. Existing rows migrate to `public`, and new public-tenant records are dual-written so rollback to the prior reader remains possible.

The vector store owns chunk payloads. The registry owns lifecycle and visibility. Neither component silently assumes ownership of the other component's data.

### Qdrant payloads

New points include `version_id`, `tenant_id`, and optional `acl` payload fields. Existing points without tenant/ACL metadata remain readable only under the `public` tenant during the migration window.

## Compatibility stages

### Stage 1 - Additive introduction (current)

Public request changes are additive and default-compatible:

- ingestion adds `tenant_id=public` and `acl=[]`;
- retrieval adds `tenant_id=public` and `principals=[]`.

The following response fields are additive:

- ingestion: `version_id`, `content_sha256`, `idempotent`;
- retrieval: `index_versions`.

Legacy clients that omit the new request fields and ignore unknown response fields continue to work in the public tenant. Existing unversioned chunks remain visible until that `(tenant_id, source_id)` has an active version in the new registry. A `building` candidate does not hide legacy evidence.

### Stage 2 - Migration window (current 0.x line)

Re-ingest each source through `POST /documents/ingest`.

For a migrated source:

1. content and pipeline configuration produce a deterministic version ID;
2. chunks are written under that immutable version;
3. the registry atomically changes the active pointer;
4. old unversioned rows for that same `source_id` become invisible but are not deleted.

This provides rollback evidence and avoids destructive migration. Identical content, pipeline profile, tenant, and ACL return `idempotent=true` without parsing, embedding, or rewriting chunks. An ACL change creates a new immutable version so authorization changes cannot reuse stale chunk policy.

### Stage 3 - Removal (future breaking release only)

Support for unversioned rows may be removed only after:

- all supported deployments have completed re-ingestion;
- migration telemetry or an explicit audit confirms no visible unversioned rows remain;
- release notes announce the removal;
- compatibility tests are updated for the breaking release.

No Stage 3 deletion is implemented in this milestone.

## Registry downgrade/re-upgrade behavior

During the 0.x compatibility window, public-tenant registry writes are dual-written to `rag_document_versions_v2` and the original `rag_document_versions`, and activation updates both `rag_source_scopes` and `rag_sources`. If operators temporarily roll back to the prior public-only reader and that reader advances the legacy active pointer, a later upgrade reconciles newer legacy timestamps and statuses back into v2. Private-tenant records are never written to the global legacy tables because those tables cannot represent composite identity.

This is a migration window, not permanent bidirectional replication. Operators must still back up the registry and vector store together and must not run old and new writers concurrently.

## Authorization migration semantics

- `(tenant_id, source_id)` is the logical document identity. The same source ID may exist independently in different tenants.
- Empty ACL means tenant-wide visibility. A non-empty ACL requires intersection with caller principals.
- SQLite applies tenant and ACL predicates before loading or decoding vectors. Qdrant applies them in the vector query filter.
- Unauthorized active versions are omitted from both search hits and `index_versions`.
- When a restricted version activates, any legacy unversioned rows for that same tenant/source are hidden even from callers that lack the required principal.
- Legacy SQLite rows and registry entries become `public` plus empty ACL through additive defaults. Legacy Qdrant points missing tenant/ACL payload are treated as public only.

The API fields are an authorization-context contract, not proof of identity. Production deployments must populate them from trusted authentication middleware or an API gateway. The model-facing `search_knowledge` tool remains public-tenant only so an LLM cannot choose its own authorization scope.

## Failure and rollback behavior

| Failure point | Visible result |
|---|---|
| Hashing, parsing, chunking, or embedding fails | Existing active version remains visible; no new active pointer is written. |
| Vector upsert fails | Candidate is marked `failed`; existing active version remains visible. |
| Process stops after vector upsert but before activation | Candidate chunks are orphaned but invisible; existing active version remains visible. |
| Activation succeeds | New version becomes visible and the previous active version is marked `retired`. |
| Identical content/profile is submitted again | Existing active version is returned as an idempotent success. |

Rollback procedure:

1. Stop writers.
2. Back up both the vector database and `rag_registry.sqlite3` as one logical recovery set.
3. Restore both files from the same snapshot.
4. Start one service instance and verify `/documents/search` returns the expected `index_versions`.
5. If temporarily downgrading within the 0.x window, run only one legacy public writer; after upgrading again, start one new instance so initialization can reconcile newer legacy public state into v2.
6. Resume normal writers only after the visibility snapshot is correct.

Do not restore only one of the two databases: chunk payloads and active-version pointers must remain aligned.

## Concurrency contract

Within one service process, ingestion uses a lock per `(tenant_id, source_id)`:

- revisions of the same logical source are serialized;
- different sources remain asynchronous and may overlap;
- retrieval is never blocked by a candidate build and continues to use the active pointer.

The lock is process-local. Multi-process or multi-host write coordination is not yet claimed; production deployment should use a single ingestion writer until a distributed lease or queue is added. Read replicas may remain concurrent.

## Verification evidence

The test suite covers:

- in-place migration of a legacy SQLite table without `version_id`;
- preservation and searchability of legacy rows before source migration;
- continued legacy visibility while the first versioned replacement is still building;
- additive API response compatibility;
- idempotent re-ingestion;
- failed replacement retaining the previous active version;
- failed first ingestion exposing no partial version;
- candidate invisibility before activation;
- source mutation detection between hashing and parsed-snapshot acceptance;
- same-source serialization and cross-source overlap;
- active-version filtering in SQLite and Qdrant adapters.

See `docs/rag-implementation-evidence.md` for exact commands and current results.
