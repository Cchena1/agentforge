# AgentForge Acceptance Evidence

## 1. Evaluation scope

This document maps engineering claims to implementation and tests. The latest verified RAG milestone covers functional behavior, schema contracts, migrations, failure injection, and concurrency semantics.

**Explicit exclusion:** no pressure, throughput, saturation, p95/p99 latency, or sustained-load testing was performed.

## 2. Capability-to-evidence matrix

| Requirement | Implementation evidence | Test evidence |
|---|---|---|
| Retry and fallback | `src/agent_service/llm.py` | `test_retry_recovers_from_transient_timeout`; API fallback test |
| Strict malformed-output handling | `src/agent_service/llm.py`, `schemas.py` | missing-field repair and schema rejection tests |
| Function calling | `src/agent_service/tools/` | tool-description and strict-schema tests |
| Async independent tools | `src/agent_service/tools/base.py` | `test_independent_tools_run_concurrently_without_load_testing` |
| Tool dependency/conflict control | tool DAG, `$ref`, resource locks | dependency, cycle, and shared-lock tests |
| Tool runtime policy | `ToolExecutionScope`, policy, approval, and guard gates | allowlist, write denial, and policy-failure tests |
| Write-tool idempotency | executor invariant plus SQLite result ledger | missing-key rejection, argument-conflict rejection, restart replay, and indeterminate-timeout tests |
| Bounded model context | complete-turn trimming and versioned content-addressed spill | context integrity, artifact hash/schema, and graph integration tests |
| Layered memory | `src/agent_service/memory.py` | short-term isolation/compaction and long-term namespace tests |
| Multi-agent isolation | `src/agent_service/multi_agent.py` | isolated context, compressed result, capability non-escalation, and spawn-denial tests |
| Loop loss prevention | `src/agent_service/graph.py` | repeated-call/max-step termination test |
| RAG parsing and citations | `documents.py`, `rag.py` | semantic chunks, citations, table rows, scan detection |
| Versioned/idempotent ingestion | `rag.py`, `rag_registry.py` | idempotence and active-version tests |
| Failed replacement safety | inactive candidate plus active pointer | failed replacement/first-ingest/building-version tests |
| Source snapshot integrity | hash before and after parsing | source-mutation rejection test |
| Legacy persistence migration | additive SQLite migration, composite-foreign-key v2 registry, public dual-write/reconciliation, and active-only legacy hiding | legacy schema, downgrade/re-upgrade, foreign-key, and first-activation visibility tests |
| Same-source conflict control | per-source `asyncio.Lock` | same-source serialization test |
| Cross-source async behavior | keyed locks, no global ingestion lock | different-source overlap test |
| SQLite/Qdrant visibility | active-version filter in adapters | RAG lifecycle and Qdrant version-filter tests |
| Tenant/ACL retrieval isolation | composite tenant/source identity plus pre-scoring store filters | cross-tenant, ACL, legacy-leak, SQLite pre-decode, Qdrant, and API tests |
| Embedding safety | immutable provider profile and dimension checks | wrong-dimension rejection test |
| API compatibility | additive response fields | API legacy-field/version evidence test |
| Evaluation contract | `scripts/evaluate_rag.py`, `eval/*.jsonl` | deterministic metrics and malformed-record tests |
| Type/dependency discipline | `pyproject.toml`, `uv.lock` | Ruff and strict mypy gates |

## 3. Public API compatibility

The request contracts for `/documents/ingest` and `/documents/search` are extended only with defaulted fields; existing callers remain compatible.

Additive ingestion request fields:

- `tenant_id` (default `public`);
- `acl` (default empty, meaning tenant-wide).

Additive ingestion response fields:

- `version_id`;
- `content_sha256`;
- `idempotent`.

Additive retrieval request fields:

- `tenant_id` (default `public`);
- `principals` (default empty).

Additive retrieval response field:

- `index_versions`.

No existing response field was removed. Persistent migration follows the staged policy in `docs/rag-versioning-migration.md`.

## 4. RAG correctness claims

Verified invariants:

1. A reader sees the old complete source version or the new complete source version, never a partially activated candidate.
2. Retrying identical content under the same pipeline is idempotent.
3. A vector-store failure cannot erase the previously active source.
4. A failed first build exposes no candidate chunks.
5. A file changed while being parsed is rejected before version registration.
6. Retrieval reports the active version snapshot it used.
7. Legacy unversioned rows remain readable until their logical source has a successfully activated version; a building replacement does not create a search outage.
8. Same-tenant/same-source writes do not race in one process, while unrelated tenant/source identities can overlap.
9. Tenant and ACL filters run before candidate scoring; unauthorized sources are absent from hits and `index_versions`.
10. The same `source_id` can exist independently in multiple tenants.
11. Legacy records migrate to the `public` tenant only.

## 5. Verification commands

Run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='<repository>\src;<repository>'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m ruff check --no-cache src tests scripts
.venv\Scripts\python.exe -m mypy --cache-dir '<writable-cache>' src scripts
.venv\Scripts\python.exe scripts\evaluate_rag.py --queries eval\golden_queries.example.jsonl
node --check public\app.js
node --check server.js
```

The root `README.md` records the latest verified counts after final target-repository verification.

## 6. Evaluation assets

- `eval/golden_queries.example.jsonl`: two schema examples, not a representative benchmark.
- `scripts/evaluate_rag.py`: source Recall@k, MRR, nDCG, and empty-result-rate evaluator.
- `docs/rag-evaluation-contract.md`: schema and release-gate rules.
- `docs/rag-implementation-evidence.md`: detailed RAG claim-to-test mapping.

Retrieval-quality improvement is claimed only for the frozen external 200-page Benchmark comparison. It does not establish production-corpus quality or answer-level citation precision.

## 7. Not verified

- production provider function-calling and rate-limit behavior;
- provider-backed embedding quality;
- OCR and complex-table quality on representative documents;
- remote Qdrant persistence and index tuning;
- learned reranking, remote sparse-vector, LLM rewrite, or graph-RAG gains beyond the implemented bounded corrective retrieval;
- authentication middleware and trusted identity-to-authorization-context mapping;
- multi-process/multi-host ingestion writers;
- durable ingestion job restart/cancellation/progress behavior;
- deletion, retention, and orphan-vector garbage collection;
- pressure, throughput, p95/p99, saturation, or long-duration stability.

## 8. Known limitations and release gates

1. The deterministic hash embedding exists for offline reproducibility and must not be presented as a production semantic model.
2. The default reranker is lightweight and deterministic, not a cross-encoder.
3. Per-`(tenant_id, source_id)` locks are process-local.
4. A crash between vector upsert and activation can leave invisible orphan chunks.
5. Parser profile identity is conservative and may cause re-ingestion after optional parser package changes.
6. The Starlette/httpx TestClient deprecation warning remains open.
7. Quality thresholds must be set only after a representative versioned dataset is reviewed.
8. Chat-tool RAG is intentionally public-tenant only until authenticated request context is propagated into the graph/tool runtime.
9. Tool-result artifacts are ephemeral: no read API, TTL, quota, backup, or restore contract is provided.
10. Context token accounting is approximate; the newest complete turn may remain over budget and emits a warning instead of being split.
11. Multi-agent capability enforcement requires each worker to pass `context.tool_scope()` into the executor.

## 9. Evidence index

- `README.md`
- `docs/architecture.md`
- `docs/adr-006-rag-correctness-first.md`
- `docs/adr-007-rag-authorization-boundary.md`
- `docs/adr-011-deepseek-harness-runtime-guardrails.md`
- `docs/rag-engineering-plan.md`
- `docs/rag-versioning-migration.md`
- `docs/rag-evaluation-contract.md`
- `docs/rag-implementation-evidence.md`
- `docs/rag-dependency-capability-inventory.md`
- `tests/`

## 8. 2026-08-16 retrieval-quality acceptance

The third 200-page Benchmark is frozen under `eval/基于OmniDocBench和OHR-Bench的RAG基准测试v3/`.

| Gate | Result |
|---|---:|
| Unique pages | 200 |
| Queries per text variant | 559 |
| Ground Truth Recall@5 | 95.53% |
| Formatting Noise Recall@5 | 95.53% |
| Semantic/OCR Noise Recall@5 | 88.91% |
| Hit-level Citation Validity | 100% |
| Pytest | 97 passed |
| Ruff | passed |
| Strict mypy | 27 source files passed |
| Pressure/load test | not performed |

Citation Validity is restricted to retrieval-hit schema, source/version identity, and quote membership. It is not answer-level semantic citation precision.
