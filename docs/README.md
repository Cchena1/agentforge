# AgentForge Documentation Map

## Project position

AgentForge is a portfolio-oriented, asynchronous agent engineering project. It demonstrates engineering delivery rather than a single prompt demo: strict contracts, failure handling, async orchestration, layered memory, versioned RAG, migration discipline, and reproducible evidence.

## Source-of-truth order

1. `README.md` - external evaluator entry point and run instructions.
2. `src/agent_service/` - maintained implementation.
3. `tests/` - executable claims and compatibility evidence.
4. ADRs and engineering plans - decisions, rejected alternatives, and staged work.
5. Resume material - concise claims that must remain consistent with implementation evidence.

The top-level legacy demo files are compatibility artifacts, not the architectural source of truth.

## Document index

| Document | Purpose |
|---|---|
| `README.md` | Product summary, architecture, API, run commands, and quality gates |
| `docs/architecture.md` | State ownership, async orchestration, memory, RAG lifecycle, and security boundaries |
| `docs/dependency-review.md` | Dependency policy and reference-project pattern review |
| `docs/rag-dependency-capability-inventory.md` | Existing RAG dependency capabilities and admission gates |
| `docs/rag-engineering-plan.md` | Research-backed target architecture and phased roadmap |
| `docs/adr-006-rag-correctness-first.md` | First-principles and adversarial RAG decision |
| `docs/adr-007-rag-authorization-boundary.md` | Tenant/ACL identity, trust boundary, pre-scoring enforcement, and migration decision |
| `docs/adr-008-document-parser-routing.md` | Quality-gated Docling/PaddleOCR/pypdf parser routing and license boundaries |
| `docs/adr-009-query-orchestration.md` | Bounded query planning, drift protection, async decomposition, fusion, latency, and cost decision |
| `docs/adr-010-observability-backup-recovery.md` | Metrics, OpenTelemetry Trace, alerting, verified backup and isolated restore decision |
| `docs/adr-011-deepseek-harness-runtime-guardrails.md` | Tool middleware, context budget, artifact spill, capability isolation, and migration decision |
| `docs/operations.md` | Operator runbook for health, metrics, traces, alert response, backup and restore |
| `docs/rag-versioning-migration.md` | Persistent-format compatibility, migration, and rollback |
| `docs/rag-evaluation-contract.md` | Versioned golden-query schema and metric rules |
| `docs/rag-implementation-evidence.md` | Claim-to-test evidence and known limits |
| `docs/acceptance-evidence.md` | Whole-project acceptance matrix |
| `docs/resume-project-description.md` | English STAR resume and interview material |
| `docs/research-snapshot.json` | Machine-readable research snapshot |

## Code ownership map

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI composition and lifecycle |
| `settings.py` | Environment configuration and route definitions |
| `schemas.py` | Strict Pydantic contracts |
| `context_runtime.py` | Complete-turn context budgeting and content-addressed tool-result spill |
| `graph.py` | LangGraph state machine and loop limits |
| `llm.py` | Async model gateway, retries, fallbacks, and structured repair |
| `tools/` | Function-calling registry and dependency-aware async executor |
| `memory.py` | Session-scoped short-term and namespaced long-term memory |
| `multi_agent.py` | Isolated parallel workers and compressed results |
| `documents.py` | Legacy parser/chunker compatibility during the migration window |
| `document_models.py` | Parser-neutral canonical blocks, assets, attempts, quality, and chunks |
| `document_pipeline.py` | Parser registry, deterministic quality gate, fallbacks, and parent-child chunking |
| `ingestion_jobs.py` | Durable asynchronous ingestion job state and restart recovery |
| `embeddings.py` | Embedding profiles and validation |
| `query_planning.py` | Typed deterministic query plans, protected anchors, and explicit decomposition |
| `rag.py` | Versioned ingestion, evidence gating, bounded query orchestration, fusion, and citations |
| `rag_registry.py` | Durable lifecycle and active-version pointers |
| `vector_store.py` | SQLite and Qdrant adapters |
| `observability.py` | Bounded-cardinality Prometheus metrics and OpenTelemetry exporters |
| `backup.py` | State lock, verified SQLite backup manifest and isolated restore |
| `operations.py` | Backup/verify/restore CLI |

## Verification path for external reviewers

1. Read the root `README.md` capability matrix.
2. Inspect `docs/adr-006-rag-correctness-first.md` and `docs/adr-008-document-parser-routing.md` for the RAG engineering rationale and parser decision.
3. Inspect `docs/rag-implementation-evidence.md` for executable claims.
4. Run pytest, Ruff, mypy, JavaScript syntax checks, and the RAG contract evaluator.
5. Review known limitations before treating any design-stage feature as implemented.

## Documentation maintenance rule

Any change to user behavior, public interfaces, commands, data sources, model parameters, runtime procedure, artifacts, or conclusions must update the relevant document and tests in the same change. Pure refactoring that preserves behavior may omit a documentation edit if the change summary explicitly says why.
