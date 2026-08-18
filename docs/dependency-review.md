# AgentForge Dependency and Reference-Project Review

## 1. Review policy

AgentForge prefers maintained libraries and proven architectural patterns over custom framework code. A dependency is admitted only when the current dependency set cannot supply the required capability and the addition has a typed boundary, a fallback, and measurable value.

The detailed RAG-specific capability inventory is in `docs/rag-dependency-capability-inventory.md`.

## 2. Reference patterns adopted

The research review compared maintained RAG and agent projects and extracted patterns rather than copying entire frameworks:

- structure-aware parsing and hybrid chunking;
- explicit typed pipeline components;
- dense/sparse retrieval and reciprocal-rank fusion;
- named vectors and payload filters;
- retrieval evaluation as a versioned artifact;
- optional graph or hierarchical retrieval rather than a mandatory default.

The source snapshot and URLs are recorded in `docs/research-snapshot.json` and `docs/rag-engineering-plan.md`.

## 3. Why entire repositories were not copied

Copying a full RAG platform would duplicate AgentForge's LangGraph, Pydantic, async execution, memory, and vector abstractions. It would also blur state ownership and introduce a large transitive dependency and security surface.

AgentForge therefore copies only patterns that fit local contracts and verifies them with a proof of concept. Third-party code must not be copied without license and attribution review.

## 4. Current runtime dependency capabilities

| Dependency | Role | Design decision |
|---|---|---|
| FastAPI | Async HTTP service and OpenAPI | Retain |
| Pydantic v2 / pydantic-settings | Strict contracts and environment configuration | Retain; reject extra fields |
| LangGraph | Bounded agent state machine | Retain for agent orchestration |
| OpenAI SDK | Async model and embedding gateway | Retain behind adapters |
| Tenacity | Retry primitives | Retain; retry only classified transient failures |
| aiosqlite | Async short-term memory, vector baseline, and version registry | Retain; WAL and explicit transactions |
| pypdf | Lightweight text-layer PDF parser | Retain |
| Docling optional extra | OCR, layout, and complex document parsing | Keep optional pending representative quality verification |
| Qdrant client optional extra | Indexed vector backend and metadata filtering | Keep optional pending remote deployment verification |
| Uvicorn | ASGI runtime | Retain |

## 5. Development dependencies

| Dependency | Role |
|---|---|
| pytest / pytest-asyncio | Functional, failure-injection, migration, and async tests |
| Ruff | Linting and import/style checks |
| mypy | Strict type checking |
| uv and `uv.lock` | Reproducible environment and dependency locking |

## 6. RAG milestone dependency decision

The versioned-ingestion and tenant/ACL milestones add no runtime package. Canonical SHA-256 identities, tenant/source-keyed `asyncio.Lock`, Pydantic validation, aiosqlite composite scopes and transactions, SQLite JSON predicates, and existing Qdrant payload filters provide the required correctness and authorization primitives.

A separate authentication package was not added because this service does not yet own the identity boundary. Production integration must derive tenant/principal context from the selected gateway or middleware, and that dependency requires its own threat-model and migration review.

This avoids admitting sparse encoders, rerankers, evaluator frameworks, graph stores, or identity libraries before a concrete missing capability and integration boundary are approved.

## 7. Candidate admission gates

| Candidate | Required evidence before adoption |
|---|---|
| Sparse encoder/BM25 package | Source Recall@k gain on a frozen domain dataset and acceptable operational cost |
| Cross-encoder/API reranker | MRR/nDCG gain, bounded latency/cost, license and fallback review |
| Ragas or similar evaluator | A specific metric gap not covered by deterministic native evaluation |
| Full Docling production profile | Representative scan/table fixtures and operator review |
| GraphRAG/LightRAG-style subsystem | Demonstrated multi-hop gap and budgeted indexing/refresh cost |
| Tokenizer package | Proven incompatibility between provider tokenization and existing capabilities |

## 8. Security and maintenance review checklist

Every new package requires:

1. license compatibility;
2. recent release and commit activity;
3. issue responsiveness;
4. security advisory and vulnerability review;
5. transitive dependency and binary/model download impact;
6. supported Python/platform wheels;
7. typed adapter and failure behavior;
8. persisted-format migration and rollback where relevant;
9. tests for unavailable dependency and degraded operation;
10. a removal criterion if the experiment fails.

## 9. Current known dependency concern

The test suite emits one Starlette TestClient deprecation warning related to the httpx transition. It does not fail business logic, but the FastAPI/Starlette/httpx compatibility path should be upgraded in a dedicated dependency change with regression tests.

## 10. Re-review cadence

Dependency health and optional integration decisions should be reviewed before each release that changes runtime packages, model providers, parser profiles, vector backends, or persisted formats. Research popularity alone is not an admission criterion.

## P0 correctness milestone (2026-08-17)

No runtime dependency was added. The implementation reuses existing Pydantic v2, aiosqlite, OpenTelemetry, Prometheus, OpenAI-compatible response metadata, and the existing Tool policy boundary. PostgreSQL, Redis checkpointing, and distributed workers were not introduced because the current LangGraph is compiled without a checkpointer and no measured workload identifies checkpoint contention. Admission of a new persistence service requires a multi-instance requirement, license/security review, a failure-mode PoC, and reproducible workload evidence.
