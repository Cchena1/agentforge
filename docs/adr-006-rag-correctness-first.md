# ADR-006: Establish RAG Correctness Before Retrieval Sophistication

- **Status:** Accepted
- **Date:** 2026-08-13
- **Decision owners:** AgentForge maintainers
- **Scope:** RAG ingestion, visibility, retrieval evolution, and evaluation

## Context

The current RAG baseline can parse, chunk, embed, store, retrieve, and cite documents, but it replaces a source by deleting its existing chunks before inserting the replacement. A failed write can therefore destroy the last usable version. The system also lacks a durable content identity and cannot prove which immutable source version produced an answer.

External systems demonstrate many useful techniques—hybrid retrieval, contextual chunks, reranking, graph retrieval, recursive summaries, and late interaction—but none repairs this lifecycle invariant. Adding those techniques first would improve ranking on top of a state model that can still expose partial or missing knowledge.

## First-principles analysis

A production RAG answer is trustworthy only if all of the following hold:

1. the indexed evidence corresponds to a known source version;
2. a reader observes either the old complete version or the new complete version, never a partial replacement;
3. retrying the same input is idempotent;
4. retrieval can report the version snapshot used by the answer;
5. failure does not silently widen authorization or return stale, retired evidence;
6. ranking improvements are measured against a fixed corpus rather than assumed from papers or popularity.

The earliest missing invariant is therefore **versioned evidence visibility**, not a more advanced retriever.

## Decision

Implement RAG improvements in this order:

1. content hashing and deterministic pipeline/version identity;
2. durable source/version registry;
3. build new vector data under an inactive version;
4. atomically activate the new version after successful storage;
5. filter retrieval to active versions and expose the version snapshot;
6. establish a frozen evaluation contract and baseline;
7. then implement tokenizer-aware chunks, sparse retrieval, RRF, reranking, corrective retrieval, and claim-level citation validation.

The existing public ingestion and retrieval routes remain available. Version fields are additive during the 0.x compatibility window.

## Adversarial review

### Challenge: "Qdrant upsert is already atomic enough"

**Rebuttal:** an individual upsert does not make a multi-step source replacement atomic. Parsing, embedding, multiple point writes, validation, and activation are separate failure domains. A source-level active pointer is required.

### Challenge: "Keep only one version to reduce storage"

**Rebuttal:** deleting first removes rollback and creates downtime on failure. Retired-version cleanup can happen asynchronously after a bounded retention period; correctness precedes storage optimization.

### Challenge: "A deterministic chunk ID is sufficient"

**Rebuttal:** the current chunk ID depends on source name, position, and text but does not identify parser/chunker/embedding semantics. The same file under a changed pipeline must be a new version.

### Challenge: "Use GraphRAG or Contextual Retrieval now because papers report gains"

**Rebuttal:** external gains do not establish benefit on this corpus. Those techniques add indexing cost and failure modes. They remain strategy-level experiments after the baseline and lifecycle invariants exist.

### Challenge: "SQLite and Qdrant cannot share identical transaction semantics"

**Rebuttal:** the registry defines logical visibility independently of physical storage. Both stores write immutable version-tagged points; retrieval includes only the active version. Physical cleanup may differ without changing the public invariant.

### Challenge: "Legacy unversioned rows will disappear after migration"

**Rebuttal:** compatibility filtering keeps unversioned rows visible until the registry has an active version for that source. A `building` candidate does not hide legacy evidence. Once activation succeeds, only the active version is visible. This permits a migration window without allowing old rows to leak back into results.

## Rejected or deferred alternatives

| Alternative | Decision | Reason |
|---|---|---|
| Delete then insert | Rejected | can lose the active corpus on failure |
| Overwrite points in place | Rejected | cannot reproduce answer evidence or roll back safely |
| Add a distributed queue immediately | Deferred | single-node bounded asyncio worker is sufficient until multi-process ownership is required |
| Adopt a second RAG framework | Rejected | duplicates LangGraph/Pydantic/Docling/Qdrant capabilities and obscures state ownership |
| GraphRAG as default | Deferred | corpus-specific benefit and indexing cost are unproven |
| Contextual Retrieval as default | Deferred | generated metadata and ingestion cost require controlled comparison |
| ColBERT as default | Deferred | multivector storage/inference complexity is unproven for the current corpus |

## Consequences

Positive:

- failed replacements retain the previous active evidence;
- identical re-ingestion becomes a no-op;
- retrieval responses can identify the active version snapshot;
- later hybrid and corrective strategies have a stable data contract;
- migration from legacy unversioned rows can be staged.

Costs and limitations:

- retired vectors temporarily consume storage;
- metadata and vector stores are not a single physical transaction, so orphan inactive points can remain after failure;
- cleanup, asynchronous job APIs, authentication-context propagation, and hybrid retrieval remain follow-up work;
- the current chunker is still character-budgeted and does not yet retain full Docling layout objects.

## Verification

Required tests:

- identical content returns the same version and reports an idempotent ingest;
- a source changed during parsing is rejected before its version is registered;
- failed replacement leaves the previous version active and searchable;
- a building version is invisible until activation;
- legacy rows remain visible while the first versioned replacement is building;
- Qdrant filters to the active version;
- legacy public response fields remain present;
- all tests, Ruff, and strict Mypy pass.
