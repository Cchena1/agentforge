# RAG Dependency Capability Inventory

**Review date:** 2026-08-13
**Decision scope:** correctness-first RAG engineering milestone

## Decision summary

This milestone adds no runtime dependency. The existing dependency set already provides the capabilities needed to implement durable version visibility, asynchronous persistence, strict contracts, optional layout-aware parsing, and optional indexed vector search.

Adding retrieval libraries before a labeled evaluation corpus exists would increase operational and security surface without proving a user-visible quality gain. Hybrid retrieval, learned reranking, and advanced query transformation therefore remain profile-gated follow-up decisions.

## Existing dependency capabilities used

| Dependency | Existing capability used | Ownership boundary | Decision |
|---|---|---|---|
| `aiosqlite` | Async SQLite access, WAL mode, explicit transactions, additive schema migration | Version registry and local vector persistence | Reuse; no replacement needed |
| `pydantic` v2 | Strict request/response and evaluation-record validation | Public/API and offline evaluation contracts | Reuse; unknown fields and wrong types fail early |
| `pypdf` | Text-layer PDF extraction and scanned-PDF suspicion signal | Built-in parser path | Reuse for lightweight baseline |
| `docling` optional extra | OCR, layout-aware parsing, and complex document conversion | Parser adapter only | Keep optional because installation is heavy and OCR quality was not verified in this milestone |
| `qdrant-client` optional extra | HNSW-backed vector search, payload filters, in-memory adapter testing | Vector-store adapter | Keep optional; remote service operation and index tuning remain unverified |
| `openai` | Async embedding route through an OpenAI-compatible API | Embedding adapter | Reuse; dimension mismatches now fail rather than mutate runtime state |
| `langgraph` | Agent control flow and bounded loop handling | Agent orchestration, not index lifecycle | Keep; do not force ingestion lifecycle into the chat graph |
| `pytest` / `pytest-asyncio` | Failure injection, async concurrency, migration, and contract tests | Verification only | Reuse |
| `ruff` / `mypy` | Static linting and strict type checks | Repository quality gates | Reuse |

## Newly implemented without a new package

| Capability | Why an existing primitive is sufficient |
|---|---|
| Immutable document versions | Canonical JSON over tenant, source, ACL, content hash, and pipeline profile, then deterministic SHA-256 identity |
| Active-version registry | Composite-key scope table, foreign-key-safe v2 version table, public dual-write migration window, and `BEGIN IMMEDIATE` transactions in SQLite |
| Idempotent ingestion | Tenant + source + normalized ACL + content hash + parser/chunker/embedding profile identity |
| Atomic visibility switch | One active pointer per `(tenant_id, source_id)`; vector writes complete before activation |
| Same-source conflict control | Per-`(tenant_id, source_id)` `asyncio.Lock`; no global serialization |
| Evaluation contract | Pydantic models and JSONL, using existing tooling |

## Deferred candidate dependencies

These are candidates, not approved dependencies:

| Need | Candidate pattern | Admission gate |
|---|---|---|
| Sparse retrieval | Qdrant sparse vectors or a maintained BM25 implementation | A labeled corpus shows source Recall@k gain over the current lexical/vector baseline |
| Learned reranking | Cross-encoder or API reranker behind a typed adapter | nDCG/MRR gain justifies latency, cost, model license, and deployment burden |
| Evaluation framework | Ragas or another maintained evaluator | Native contracts become insufficient for a specific measured metric; judge-model variance is documented |
| Advanced PDF/table parsing | Full Docling production profile | Representative scanned and table-heavy documents pass quality review |
| Graph retrieval | GraphRAG/LightRAG-style optional subsystem | Multi-hop corpus benchmark shows material gain and the extraction/refresh cost is budgeted |

## Dependency admission checklist

Before adding any package, record:

1. exact capability gap that current dependencies cannot satisfy;
2. license compatibility with the repository;
3. release recency and maintenance activity;
4. issue and security-response evidence;
5. transitive dependency and binary/model download impact;
6. typed adapter boundary and fallback behavior;
7. offline and provider-backed test plan;
8. migration and rollback plan for any persisted format;
9. measured quality gain on a versioned golden dataset;
10. removal criteria if the experiment fails.

## Evidence sources

The comparative research snapshot is stored in `docs/research-snapshot.json`, and the paper/project mapping is documented in `docs/rag-engineering-plan.md`. The architectural decision for this milestone is `docs/adr-006-rag-correctness-first.md`.
