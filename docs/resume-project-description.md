# AgentForge Resume Project Description

## One-line positioning

Built an asynchronous, schema-first agent service with LangGraph, strict Pydantic contracts, retry/fallback recovery, dependency-aware parallel tools, layered memory, and failure-safe versioned RAG.

## Recommended resume entry - STAR format

### AgentForge - Production-Oriented Async Agent Service

**Situation**
A single-file agent demo could call a model but lacked production controls: malformed JSON could enter business logic, tool calls had no dependency/conflict model, memory ownership was unclear, agent loops were unbounded, and document replacement could erase searchable knowledge when ingestion failed.

**Task**
Redesign the demo into an evaluation-ready Agent engineering project while preserving the public 0.x API, maintaining reproducible local operation, and producing executable evidence instead of unsupported performance claims.

**Action**

- Designed a modular FastAPI/LangGraph architecture with explicit ownership for graph state, session memory, long-term semantic memory, tool DAG state, vector data, and active document versions.
- Implemented a fully asynchronous model and tool path using AsyncOpenAI, `asyncio`, async LangGraph nodes, aiosqlite, semaphores, dependency resolution, `$ref` result wiring, cycle rejection, and resource locks for conflicting mutations.
- Defined schema-first function-calling contracts with Pydantic v2, precise tool-selection boundaries, `extra="forbid"`, bounded structured-output repair, and immediate rejection of missing fields, wrong types, and unknown arguments.
- Added classified retry and fallback behavior: transient failures receive bounded exponential backoff; exhausted routes switch only to configured compatible fallbacks; unsafe recovery returns explicit degradation or human handoff.
- Built layered memory with session-isolated recent context, rolling summaries, user-namespaced long-term vector memory, and compressed multi-agent result envelopes instead of shared raw context.
- Reworked RAG ingestion around immutable content/pipeline/authorization versions, SHA-256 identity, idempotent re-ingestion, a durable SQLite active-version registry, active-only retrieval filters for SQLite/Qdrant, and additive API version evidence.
- Added tenant-scoped source identity and document ACL enforcement: SQLite filters before vector decoding, Qdrant filters in-query, unauthorized sources are omitted from hits and version metadata, and legacy data migrates to public-only semantics.
- Used adversarial failure tests to prove that failed replacement, partial first ingestion, concurrent revisions, embedding-dimension mismatch, source mutation during parsing, and legacy SQLite migration cannot silently corrupt visible evidence.
- Added a Pydantic-validated JSONL evaluation contract and deterministic Recall@k/MRR/nDCG evaluator while explicitly excluding pressure-test and quality-improvement claims without a representative labeled corpus.

**Result**

- Converted the demo into a typed, modular service with reproducible `uv.lock` dependency management and external-evaluator documentation.
- Verified the current milestone with **47 passing tests**, **Ruff passing**, and **strict mypy passing across 17 source files**.
- Preserved existing API fields while adding `version_id`, `content_sha256`, `idempotent`, and retrieval `index_versions` during a documented migration window.
- Demonstrated failure-safe RAG behavior: readers continue using the previous complete version until a replacement is fully stored and activated.
- Produced implementation, migration, ADR, dependency, and acceptance evidence suitable for code review and interview discussion.

## One-page resume bullets

- Engineered an async Agent service with **FastAPI, LangGraph, asyncio, Pydantic v2, AsyncOpenAI, aiosqlite, SQLite/Qdrant, and uv**, replacing a single-file demo with typed module boundaries and explicit state ownership.
- Built **schema-first function calling** and structured-output repair; rejected missing/wrong/extra fields before business logic and implemented classified retry, route fallback, controlled degradation, and human handoff.
- Implemented a **dependency-aware parallel tool DAG** with semaphores, `$ref` data flow, cycle detection, failure propagation, timeouts, and resource locks to prevent conflicting mutations.
- Designed **layered memory and multi-agent isolation** using bounded session context, rolling summaries, user-namespaced vector memory, isolated worker context, and compressed citation-bearing results.
- Delivered **failure-safe versioned RAG** with content hashing, pipeline-profile identity, idempotent ingestion, active-pointer switching, SQLite/Qdrant version filters, legacy-data migration, and source-level concurrency control.
- Established executable evidence with **47 pytest cases, Ruff, strict mypy for 17 source files, migration/failure-injection tests, and a JSONL retrieval evaluator**, without claiming unperformed pressure testing.

## 60-second interview introduction

I started from an agent demo that could call an LLM but had none of the controls needed for reliable delivery. I separated graph state, memory, tools, model routing, and retrieval behind typed interfaces. The runtime is async end to end, and tool calls form a dependency DAG rather than an uncontrolled list. Pydantic validates every model-produced structure, while retry is limited to transient failures and fallback is explicit.

The most important RAG change was not adding a fashionable retriever. I first fixed the evidence lifecycle. Each document build now has a deterministic content-and-pipeline version. Candidate chunks are stored invisibly, and a durable registry activates the new version only after storage succeeds. Failure-injection tests show that a failed replacement keeps the old version searchable, concurrent revisions do not race in one process, and old SQLite data migrates additively. I then added tenant-scoped source identity and ACL filtering at the storage boundary, with adversarial tests for cross-tenant reads, same-ID tenant isolation, legacy leakage, and authorization-before-scoring. I documented the remaining boundary honestly: authentication context still has to come from a trusted gateway, alongside durable ingestion jobs, remote Qdrant verification, and representative retrieval-quality evaluation.

## Interview deep dives

### 1. How do you handle malformed model JSON?

Treat model output as untrusted input. Extract the candidate object, validate it against a strict Pydantic model, run at most one bounded repair attempt with validation errors as feedback, then validate again. If fields remain missing or types remain wrong, return a typed failure or degradation. Never fill business-critical fields with guessed defaults.

### 2. How do you keep tool descriptions from causing wrong selection?

Describe decision boundaries, not just capabilities: when to use, when not to use, side effects, permission scope, input units, and successful output shape. Use enums and bounded fields where the domain is closed, reject extra arguments, and keep one tool responsible for one coherent operation.

### 3. How do parallel tools avoid dependency and mutation conflicts?

Represent calls as a DAG. Only nodes with satisfied dependencies become runnable. Resolve upstream results through typed references, reject cycles, skip downstream nodes after dependency failure, use a semaphore for bounded concurrency, and acquire a resource lock for calls that mutate the same logical object.

### 4. How is memory layered?

Graph state owns the current turn. SQLite short-term memory owns a bounded recent session window and rolling summary. Long-term memory owns user-namespaced semantic facts. The RAG index owns source documents. Multi-agent workers receive isolated context and return compressed facts/citations instead of raw traces.

### 5. Why did you prioritize versioned ingestion over hybrid retrieval?

A better ranker cannot repair missing or partially replaced evidence. The first invariant is that every result comes from a known complete source version. Immutable candidates plus an active pointer provide rollback, reproducibility, and failure safety. Hybrid retrieval and reranking can then be evaluated on a stable corpus.

### 6. What happens when document replacement fails?

The old active pointer remains unchanged. If parsing, embedding, validation, or vector upsert fails, the candidate never activates. If a process stops after vector upsert but before activation, those chunks are invisible orphans rather than visible corruption. Cleanup is a separate retention responsibility.

### 7. How do you handle files changing during ingestion?

Hash before parsing and hash again after parsing. If the values differ, reject the build before registering its version. This prevents parsed content from being attributed to the wrong content identity.

### 8. What are the current limits?

No pressure testing was performed. The golden file is only a schema example. Tenant/ACL filtering is functionally verified, but authentication and trusted identity propagation are external. Durable ingestion jobs, distributed writer coordination, orphan cleanup, production OCR quality, provider embedding quality, learned reranking, and remote Qdrant operation are not yet claimed.

## Evidence index

- `README.md`
- `docs/architecture.md`
- `docs/adr-006-rag-correctness-first.md`
- `docs/adr-007-rag-authorization-boundary.md`
- `docs/rag-implementation-evidence.md`
- `docs/rag-versioning-migration.md`
- `docs/acceptance-evidence.md`
- `tests/test_rag.py`
- `tests/test_api.py`
- `tests/test_tools.py`
- `tests/test_llm.py`
- `tests/test_memory.py`
- `tests/test_multi_agent.py`

## Claims to avoid

Do not claim:

- production throughput, p95/p99 latency, or high-concurrency capacity;
- retrieval-quality improvement without a representative labeled comparison;
- production OCR/table accuracy;
- built-in authentication or trusted identity proof inside the current service;
- distributed or exactly-once ingestion;
- remote Qdrant production readiness;
- complete claim-level citation verification.

These boundaries make the resume narrative credible and provide clear follow-up engineering topics.
