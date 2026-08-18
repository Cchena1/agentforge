# ADR-013: P0 Correctness Boundaries for Runtime, Memory, and Write Tools

- Status: Accepted
- Date: 2026-08-17
- Scope: Agent runtime, long-term Memory, model observability, and Tool execution contracts

## Context

An adversarial review raised four P0 concerns: SQLite contention in LangGraph checkpoints, missing write-tool idempotency, long-term Memory staleness/deletion, and missing model cost/error attribution. The first concern was based on an incorrect architectural assumption: the current graph is compiled without a LangGraph checkpointer. SQLite currently owns short-term Memory, long-term Memory, RAG metadata/index state, and ingestion jobs; it does not persist LangGraph checkpoints.

Replacing SQLite with PostgreSQL/Redis/Celery without a measured multi-instance workload would add an unverified distributed runtime and violate YAGNI. The immediate risk is correctness at existing boundaries, not an unproven QPS target.

## Decision

### 1. Keep LangGraph execution in-process and make the boundary explicit

- `AgentGraph` owns bounded per-request control flow only.
- It has no durable checkpointer and cannot resume a graph after process termination.
- Durable ingestion jobs remain a separate state machine; they are not LangGraph checkpoints.
- PostgreSQL/Redis checkpointing is deferred until a multi-instance requirement or measured SQLite lock contention exists.

### 2. Upgrade long-term Memory with an additive schema-v2 migration

The Memory store now records:

- `memory_key` for semantic replacement;
- `valid_from` and `expires_at` for validity windows;
- `superseded_at` and `superseded_by` for update lineage;
- `embedding_profile` for provider/model/dimension compatibility.

Recall excludes superseded, not-yet-valid, and expired records. It combines semantic similarity, lexical overlap, importance, and recency, then abstains below a configured relevance threshold. Records with a stale embedding profile or dimension are lazily re-embedded in a bounded batch before scoring. Deletion is namespace-scoped.

### 3. Coordinate write-capable Tool calls through a durable idempotency ledger

The executor enforces `idempotency_key` as a non-bypassable write invariant before approval or execution. The service composition uses `SQLiteToolIdempotencyStore` in `tool_idempotency.sqlite3` to atomically reserve `(tool_name, idempotency_key)`, reject key reuse with a different argument fingerprint, persist successful `ToolResult` envelopes, and replay a completed result after process restart without invoking the handler again.

A write timeout or exception after dispatch has an unknowable external outcome. The ledger therefore marks it `indeterminate`, returns `TOOL_WRITE_OUTCOME_UNKNOWN` with `retryable=false`, and requires human/domain reconciliation instead of blind retry. An abandoned `in_progress` lease also becomes indeterminate after a bounded TTL. Embedded library callers use a process-local implementation unless they inject a durable store.

This is not a cross-system exactly-once claim. Before a payment, message, ticket, or other externally mutating Tool is admitted, its owner must propagate the same key to a target system that supports idempotency or transactional reconciliation.

### 4. Add bounded model usage and failure attribution

Model responses capture prompt, completion, cached, and reasoning token counts when the provider supplies them. Estimated cost is calculated only from explicitly configured per-route prices. Prometheus metrics and OpenTelemetry spans use bounded route/category labels and never record prompts or model output bodies.

### 5. Clarify RAG activation semantics

“Atomic activation” means the candidate index is prepared and validated before a single active-version pointer is switched inside the version registry. It is not a distributed transaction across SQLite, Qdrant, cloud object storage, and a model provider. A crash can leave invisible orphan chunks; the old active version remains readable.

## Compatibility and migration

- Public API changes are additive: `DELETE /memory/{memory_id}` is added; existing routes are unchanged.
- Memory schema migration is forward and additive. Existing rows remain readable and are backfilled with `valid_from=created_at`.
- New optional request/response fields preserve existing clients.
- `tool_idempotency.sqlite3` is a new additive state file. The existing backup glob includes it automatically once created; older backups remain restorable but contain no write-result history.
- Downgrading to an older binary after schema-v2 use is not supported without restoring a verified pre-migration backup.

## Rejected alternatives

1. **Immediate PostgreSQL/Redis migration** — rejected because the current graph has no checkpointer and no pressure-test evidence identifies SQLite as the active P0 bottleneck.
2. **Process-local cache as the service default** — rejected because it loses completed results on restart. It remains available only for embedded callers and tests; the FastAPI service injects the durable SQLite ledger.
3. **LLM-selected Memory deletion/replacement policy** — rejected because lifecycle decisions require deterministic namespace and validity contracts.
4. **Provider list-price lookup at runtime** — rejected because prices are mutable external configuration and would make accounting non-reproducible.

## Consequences

Positive:

- Corrects false architecture claims before scaling work.
- Makes stale Memory, conflicting facts, deletion, and weak recall fail deterministically.
- Exposes token/cost/failure evidence without leaking content.
- Prevents completed write calls from being re-executed locally and makes uncertain outcomes fail closed.

Limitations:

- The Agent graph is not crash-resumable.
- SQLite remains a single-node persistence baseline.
- Memory recall still scans one namespace in-process and is not a large-scale ANN implementation.
- Cross-system exactly-once behavior remains Tool/domain-owned; the local ledger cannot prove whether an external system committed just before a process or network failure.
