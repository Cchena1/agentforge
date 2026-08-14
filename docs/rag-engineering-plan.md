# AgentForge RAG Engineering Upgrade Plan

**Status:** living plan; Phase 1 correctness and tenant/ACL isolation are implemented, later retrieval-quality phases remain proposed
**Research snapshot:** August 13, 2026
**Objective:** evolve the current demo into a production-oriented, evidence-grounded RAG subsystem that demonstrates practical Agent engineering capability.
**Scope note:** this plan explicitly excludes load, stress, throughput, and p95/p99 pressure testing.

## 1. Executive decision

AgentForge already has a credible baseline: asynchronous parsing and embedding calls, Docling/PDF fallbacks, structure-aware chunking, SQLite and Qdrant adapters, lexical/vector scoring, deduplication, and structured citations. The right move is not to replace it with a large framework. The upgrade should preserve its typed boundaries and add production lifecycle management, hybrid retrieval, bounded corrective reasoning, citation verification, and regression evaluation.

Recommended target:

1. Docling-backed canonical ingestion for layout, OCR, tables, and stable source locations.
2. Versioned, idempotent indexing with durable ingestion jobs and atomic activation.
3. Dense and sparse retrieval in parallel, reciprocal-rank fusion, and an optional reranker.
4. Bounded adaptive/corrective retrieval inspired by Adaptive-RAG and CRAG.
5. Evidence-aware context packing and claim-level citation verification.
6. A frozen evaluation corpus and release regression gates.
7. Pydantic contracts, asyncio concurrency, and LangGraph orchestration without framework sprawl.

GraphRAG, RAPTOR, late chunking, ColBERT, HyDE, and generated contextual chunks remain optional experiments. They become defaults only after they outperform the simpler baseline on the project's own corpus.


## Implementation checkpoint: August 13, 2026

Implemented from this plan:

- immutable content/pipeline/authorization versions and durable active pointers;
- additive legacy migration for vector and registry state;
- tenant-scoped source identity and ACL filtering before candidate scoring;
- failure-safe activation, idempotency, source-mutation checks, and process-local keyed concurrency;
- strict evaluation contracts and executable migration/isolation evidence.

Still gated on representative corpus evidence or additional platform work: sparse retrieval, learned reranking, corrective RAG, claim-level citation verification, durable jobs, authenticated identity propagation, and remote Qdrant operations.

## 2. Current-state audit

### Existing strengths

- Composite parser with optional Docling and built-in fallback.
- Scanned-PDF detection rather than silently accepting empty extraction.
- Chunking that prefers headings, paragraphs, and table rows over blind fixed-width slicing.
- Async embedding and vector-store protocols.
- Deterministic HashEmbedding for offline tests and an OpenAI-compatible provider.
- SQLite exact cosine retrieval for local use and a Qdrant adapter.
- Strict Pydantic request/response models and structured Citation output.

### Main engineering gaps

| Area | Current behavior | Required upgrade |
|---|---|---|
| Source lifecycle | immutable candidate versions, idempotency, active-pointer switch, and rollback-safe public migration are implemented | durable ingestion job, cancellation/progress, retention, orphan cleanup, and distributed writer lease |
| Document model | Docling result flattened to Markdown-like blocks | preserve hierarchy, page, locator, bbox, table identity, captions, OCR/confidence metadata |
| Chunk budget | character targets | embedding-tokenizer-aware hierarchical chunks and parent-child links |
| Embedding contract | embed plus dimension | model/profile ID, token limit, normalization, batching, cache and migration compatibility |
| Retrieval | dense retrieval plus local lexical score | independent sparse branch, parallel retrieval, RRF, reranker, context builder |
| Corrective path | one retrieval pass | evidence evaluator, bounded rewrite/broaden/retrieve-again branch, abstention |
| Citation | hits contain citation metadata | validate claim coverage, citation existence, authorization, quote/source consistency |
| Observability | end-to-end latency | per-stage trace, model/index versions, candidate counts, fallback/degradation reason |
| Evaluation | unit/integration tests | frozen relevance set, retrieval/grounding/citation metrics, release gates |
| Security | workspace path resolution plus tenant/ACL filtering in SQLite and Qdrant | trusted authentication context, MIME/size limits, untrusted-document prompt boundary |

The original delete-before-insert risk is resolved: the current implementation builds an immutable candidate and activates it only after vector storage succeeds. Remaining lifecycle work is operational durability?job recovery, distributed ownership, retention, and orphan cleanup?not evidence visibility correctness.

## 3. Reusable project patterns

GitHub stars are discovery signals, not selection criteria. Maintenance, license, fit, documentation, and operational complexity decide adoption.

| Project | Reusable pattern | AgentForge decision |
|---|---|---|
| RAGFlow | deep document understanding, heterogeneous ingestion, multiple recall, fused reranking, grounded citations | reuse architecture concepts, not the entire platform |
| Docling | layout/OCR/table parsing, unified document model, hierarchical and hybrid chunking | keep as preferred parser and preserve more structure |
| Qdrant | named dense/sparse vectors, payload filters, RRF, multivectors, collection aliases | production vector backend |
| Haystack | explicit typed components, branches, loops, serializable pipelines | adopt component discipline while retaining LangGraph |
| GraphRAG / LightRAG | graph-enhanced local/global retrieval | optional strategy only for relationship-heavy corpora |
| Ragas | experiment tracking and RAG evaluation | optional evaluation dependency, never runtime-critical |
| RAGatouille | accessible ColBERT integration | reference for a later late-interaction PoC |

## 4. Paper-to-engineering mapping

| Work | Engineering use |
|---|---|
| Self-RAG | lightweight evidence sufficiency and groundedness checks; do not train a Self-RAG model |
| CRAG | retrieval-quality evaluator that triggers a corrective branch |
| Adaptive-RAG | typed policy selecting single, corrective, or bounded multi-hop retrieval |
| RAPTOR | optional summary-tree index for global questions over long corpora |
| Late Chunking | optional long-context embedding profile |
| ColBERTv2 | optional token-level late-interaction profile using multivectors |
| HyDE | low-confidence fallback while retaining the original-query branch |
| IRCoT | bounded retrieve-reason-retrieve state for multi-hop queries |
| BGE-M3 | self-hosted Chinese/English dense+sparse candidate after license/hardware review |
| MTEB | reminder that model choice must be based on a domain test set, not one leaderboard |
| RAGAS / RAGChecker | evaluation ideas and optional offline diagnostic tooling |

Anthropic's Contextual Retrieval is also useful: prepend a short document-aware context to each chunk for embedding/BM25, then rerank. It adds ingestion cost and generated metadata, so it should be an off-by-default EnrichmentStrategy and the original chunk must remain the source of truth.

## 5. Target architecture

~~~mermaid
flowchart LR
  subgraph Ingestion
    A["Register source"] --> B["Versioned job"] --> C["Parser router"]
    C --> D["Canonical document"] --> E["Token-aware hierarchical chunks"]
    E --> F1["Dense batch"]
    E --> F2["Sparse batch"]
    F1 --> G["Inactive version index"]
    F2 --> G
    G --> H["Validate and atomically activate"]
  end
  subgraph Query
    Q["Query and authorization"] --> R["Policy"]
    R --> S1["Dense retrieval"]
    R --> S2["Sparse retrieval"]
    S1 --> T["RRF fusion"]
    S2 --> T
    T --> U["Optional reranker"] --> V["Evidence sufficiency"]
    V -->|sufficient| W["Context builder"] --> X["Grounded generation"] --> Y["Citation verifier"]
    V -->|low confidence and budget| Z["Rewrite or broaden"] --> S1
    V -->|budget exhausted| H1["Abstain or human handoff"]
    Y -->|invalid after one repair| H1
  end
~~~

LangGraph owns bounded workflow state. The RAG core remains callable without LangGraph through typed protocols so it can be unit-tested and used from API routes, tools, and workers.

## 6. Core contracts

Required Pydantic domain records:

- SourceRecord: source ID, tenant, ACL, type, active version, deletion state.
- DocumentVersion: content hash, parser/chunker/embedding/sparse/index versions and lifecycle state.
- IngestionJob: durable state, attempts, timestamps, error classification and cancellation.
- CanonicalBlock: type, text, page, locator, bbox, heading path, table ID and confidence.
- ChunkRecord: immutable chunk ID, version, parent, text, contextual prefix, token count and locators.
- RetrievalCandidate: per-branch ranks/scores and source authorization metadata.
- RetrievalTrace: policy, versions, candidate counts, selected chunks, stage timings and degradation reason.
- GroundedAnswer/GroundedClaim: answer text, claim-to-citation mapping, confidence, insufficiency and handoff state.

Identifier rules:

- `(tenant_id, source_id)` is the stable logical document identity;
- version_id is derived from canonical tenant/source/ACL/content/pipeline identity or protected by an equivalent uniqueness rule;
- chunk_id derives from version, canonical locator and normalized text hash;
- vector point IDs derive deterministically from chunk IDs;
- public citation IDs include source, version and chunk identity.

These rules make retries idempotent and answers reproducible against the exact indexed version.

## 7. Versioned ingestion

### Job lifecycle

Use a durable SQLite-backed job store for the initial single-node deployment:

queued -> parsing -> chunking -> embedding -> indexing -> validating -> active

Terminal states: completed, failed_retriable, failed_permanent, cancelled, superseded.

Run a bounded in-process asyncio.Queue with controlled workers. Define a JobBroker protocol but do not add Redis/Celery until multi-process or distributed execution is a demonstrated requirement.

### Idempotency and activation

1. Resolve and validate the source.
2. Stream SHA-256 while reading.
3. Return the active version if tenant, source, normalized ACL, content hash, and pipeline profile already match.
4. Parse and index into a new inactive namespace.
5. Validate non-empty content, chunk/vector count, dimensions and sample readability.
6. Atomically set the new active version and retire the old one.
7. Delete retired vector data asynchronously after a documented retention window.

For Qdrant, use versioned collections plus aliases, or an equivalent active-version payload contract. Never delete the active index before replacement validation succeeds.

### Parser routing

| Input | Primary | Fallback | Do not retry |
|---|---|---|---|
| Digital PDF | Docling | pypdf | corrupt or empty after both parsers |
| Scanned PDF | Docling OCR | configured OCR provider/process | unsupported input or unusable extraction |
| DOCX/PPTX/XLSX/HTML | Docling | reviewed format-specific fallback only | corrupt or unsupported input |
| TXT/Markdown/code | built-in parser | permitted encoding fallback | policy violation or unusable content |
| CSV/TSV | structured built-in parser | safe dialect fallback | malformed/oversized content |

Retry timeouts, temporary file locks, 429 and transient 5xx errors with capped exponential backoff and jitter. Do not blindly retry schema errors, corrupt files, authentication errors or policy rejection.

### Tables and scans

- Preserve table identity, caption, header hierarchy, page and source locator.
- Create a parent table chunk with caption/schema/summary and child row-group chunks that repeat headers.
- Do not split a row merely to hit a soft target; apply a hard maximum and emit a warning.
- Store OCR origin and confidence. Low-confidence OCR can be searchable but must remain visible to the reranker/verifier.
- Empty scanned extraction is a failure, not a successful zero-chunk ingestion.

## 8. Chunking

Replace character-only sizing with Docling hierarchy plus the tokenizer of the selected embedding profile. Initial configurable values for evaluation, not universal claims:

- target around 450 embedding tokens;
- hard maximum around 700 tokens;
- overlap around 64 tokens across natural sibling boundaries;
- heading path and optional parent summary stored separately;
- table header-preserving row groups;
- code symbol/block boundaries when relevant.

Index children for precision and expand to a bounded parent or neighbor context for generation. Every child records parent_chunk_id.

Optional experiments:

- contextual retrieval: embed generated prefix plus chunk, cite original text, cache by content/prompt version;
- late chunking: only with a compatible long-context embedding model;
- RAPTOR: separate summary-level nodes for global questions.

## 9. Embeddings, caches and index

### Embedding profile contract

Expose profile_id, provider/model, output dimension, maximum input tokens, normalization, languages/modalities, maximum batch size, retry policy and cache namespace. HashEmbedding remains test-only and production configuration must reject it.

Maintain two candidate deployment profiles:

1. managed OpenAI-compatible embedding for low operational burden;
2. reviewed self-hosted multilingual embedding such as BGE-M3 for Chinese/English/private deployment.

Choose on the project's frozen dataset. Public MTEB ranking only shortlists models. Review model license, model card, hardware memory, batch behavior and operational ownership before adoption.

Cache keys:

- parse: content hash + parser profile;
- chunks: canonical document hash + chunker profile;
- embeddings: model profile + normalized text hash;
- query embedding: model profile + normalized query hash;
- retrieval: active index version + authorization scope + query + filters + policy.

A new index version invalidates retrieval caches naturally.

### Qdrant layout

Named vectors:

- dense: HNSW-backed dense embedding;
- sparse: lexical sparse representation;
- optional late: multivector representation for ColBERT experiments.

Payload indexes: tenant_id, acl, source_id, version_id, document_type, language, page, and business date fields where relevant. Authorization filters are mandatory inside every branch query, never applied only after retrieval.

SQLite exact search remains the deterministic development/test backend; it is not positioned as horizontally scalable production storage.

## 10. Retrieval pipeline

### Policy

Begin with deterministic and observable policies:

- single: normal factual lookup;
- corrective: activated only after insufficient first-pass evidence;
- multi_hop: explicit or detected from clear decomposition indicators.

Do not introduce an LLM router before a labeled policy dataset exists. A future router must return a strict schema with confidence and reason codes.

### Parallel candidate generation

Use asyncio.TaskGroup for independent dense, sparse and optional exact metadata/identifier branches. Every branch receives one immutable RetrievalContext containing tenant, ACL, source filters, active index version, timeout budget and trace ID. Branches return values; only the coordinator mutates the merged result.

### Fusion and reranking

1. Retrieve a candidate pool larger than final context size.
2. Fuse ranks with reciprocal-rank fusion rather than adding uncalibrated raw scores.
3. Deduplicate by content hash and parent/source proximity.
4. Optionally rerank the shortlist with a cross-encoder or reviewed hosted reranker.
5. Select diverse evidence under the context token budget.

If reranking fails, return fused ranking with degraded=true. If one retrieval branch fails, use the survivor only when authorization and index consistency remain guaranteed; otherwise abstain.

ColBERT/multivector retrieval stays optional until token-level matching demonstrates sufficient benefit to justify storage and inference complexity.

### Evidence sufficiency

Prefer a calibrated feature-based evaluator before an LLM judge:

- top score and score margin;
- dense/sparse rank agreement;
- independent support count and source diversity;
- query-term/entity coverage;
- duplicate ratio;
- parser/OCR confidence;
- requested metadata coverage.

Tune thresholds on the frozen labeled set. Do not copy thresholds from external projects.

### Corrective branch

If evidence is insufficient and budget remains:

1. normalize spelling and identifiers;
2. create one constrained rewrite;
3. broaden filters only when authorization/business policy permits;
4. optionally add a HyDE branch while retaining the original query;
5. retrieve and rerank once more;
6. stop after at most two total retrieval rounds.

For multi-hop work, cap decomposition at three sub-questions and retrieval at two rounds. Store intermediate facts as structured cited records, not unconstrained chain-of-thought.

### Context builder

The context builder owns the token budget. It groups evidence by source/parent, removes near-duplicates, includes stable citation labels, reserves output capacity, and expands parent/neighbor context only when useful. Retrieved text is untrusted data and cannot provide system/tool instructions.

## 11. Answer and citation integrity

Generation returns strict claims with citation IDs, confidence, insufficient_evidence and handoff_required. Validation checks:

1. every citation ID exists in supplied evidence;
2. every non-trivial factual claim has a citation;
3. quoted snippets match source spans after normalization;
4. source/version/chunk IDs agree;
5. no citation points to filtered or unauthorized evidence;
6. low evidence becomes abstention instead of fluent invention.

Allow one structured repair attempt for malformed output or bad citation references. A second failure returns a typed degraded/handoff response.

## 12. Async, retry, fallback and conflict control

Async rules:

- TaskGroup for independent retrieval branches and document batches;
- semaphores per provider;
- bounded queues for backpressure;
- embedding batches constrained by item and token limits;
- cancellation propagation during shutdown;
- asyncio.to_thread or a process boundary for blocking parsing/OCR;
- no database transaction held across network/model calls.

Failure policy:

| Failure | Retry | Fallback |
|---|---|---|
| timeout, 429, transient 5xx | capped exponential backoff with jitter | compatible provider/profile only |
| invalid credentials/policy rejection | no | fail closed with typed configuration error |
| malformed model JSON | one repair | typed handoff/degraded response |
| corrupt document | no blind retry | supported parser fallback, then permanent failure |
| reranker timeout | only within request budget | fused rank |
| sparse branch unavailable | normally no request retry | dense-only degraded result when policy allows |
| vector store inconsistent/unavailable | bounded transient retry | no silent stale local fallback without an explicit consistency contract |

Important vector fallback rule: models of different dimension or semantics cannot silently write into the same vector field. A fallback model must be declared compatible or use a separate named vector/index version.

State ownership:

- ingestion worker owns job transitions;
- source registry owns active version;
- retrieval coordinator owns merge/fusion;
- context builder owns token budget;
- verifier owns citation validity;
- LangGraph state carries IDs and compressed typed results, not duplicated full documents.

## 13. Evaluation and release gates

### Scope

No pressure testing will be performed. Per-stage durations may be captured during ordinary deterministic tests for troubleshooting, but the project must not claim throughput, maximum concurrency, p95/p99 or production capacity.

### Versioned assets

Add:

- eval/golden_queries.jsonl
- eval/parser_fixtures/
- eval/expected_citations.jsonl
- eval/profiles/*.yaml
- scripts/evaluate_rag.py
- artifacts/rag-eval/<date-profile>/report.json

Each query records query text, tenant/ACL scope, filters, relevant source/chunk or evidence spans, answerability, expected sources, and tags such as table, scan, identifier, multilingual, multi-hop, and prompt injection.

Metrics:

- retrieval: Recall@k, MRR, nDCG@k, source recall, duplicate rate, empty and unauthorized result rates;
- generation: correctness on labeled items, faithfulness/groundedness, citation precision/recall, claim citation coverage, invalid citation rate, abstention accuracy;
- correctness: idempotency, active-version atomicity, fallback classification, dimension mismatch rejection, tenant isolation, loop termination, schema repair exhaustion.

Ragas may be an eval optional dependency. RAGChecker may be used for offline diagnosis. Neither belongs in the production runtime dependency set.

Release process:

1. freeze the current implementation as baseline;
2. run every profile on the same committed dataset and evaluator versions;
3. require improvement on agreed primary metrics with no regression in authorization, citation validity, abstention or migration tests;
4. record thresholds in an ADR after baseline measurement rather than inventing them now;
5. store machine-readable reports and a concise Markdown comparison with model/index/prompt versions.

## 14. Observability and security

Emit a structured retrieval trace with request/session/trace IDs, authorization-scope hash, policy, rewrite count, all pipeline versions, branch candidate counts, stage durations, selected chunk IDs/scores, citation IDs, degradation and abstention reasons.

Do not log full content by default. Use hashes, IDs and short redacted previews; full-content debug must be opt-in and access controlled.

Security requirements:

- validate MIME/content, extension, maximum file size/page count and decompression limits;
- enforce tenant/ACL filters in every store query;
- treat retrieved instructions as untrusted data;
- define deletion semantics for metadata, vectors, caches, traces and exported artifacts;
- exclude secrets from logs and exceptions;
- test adversarial prompt injection and cross-tenant identifiers.

## 15. Public API and data migration

Do not break current public contracts in place.

Stage A, addition:

- POST /v2/documents returns job_id, source_id and candidate version_id;
- GET /v2/ingestion/jobs/{job_id};
- GET /v2/documents/{source_id};
- DELETE /v2/documents/{source_id} with retention semantics;
- POST /v2/retrieval returns hits plus trace_id, policy, degradation and index version.

Stage B, migration window:

- existing endpoints call compatibility adapters;
- documentation marks old routes/fields deprecated;
- integration tests verify both contracts;
- caller usage is confirmed before removal.

Stage C, removal:

- remove legacy routes only in a documented major version;
- publish migration notes and a data migration command;
- retain the previous index for a stated rollback window.

Persistent stores receive explicit schema versions. Qdrant reindexing uses a new version/collection and an atomic alias or active-version switch. SQLite uses forward migrations with backup and rollback documentation.

## 16. Implementation roadmap

Estimates are rough single-developer engineering ranges. They exclude external model provisioning, hardware/deployment approval and pressure testing.

### Phase 0 - baseline and ADRs (2-3 days)

Deliver: evaluation schema and starter corpus, current baseline report, ADRs for canonical documents/Qdrant/embeddings/API migration/evaluation, and an updated dependency capability inventory.

Exit: current behavior is reproducible and no choice depends only on stars or a public leaderboard.

### Phase 1 - versioned ingestion (4-6 days)

Deliver: source/version/job repositories, bounded async worker, hash idempotency, embedding cache, richer Docling conversion, tokenizer-aware hierarchical chunker, build-validate-activate flow and legacy compatibility adapter.

Exit: repeated content is idempotent; failed replacement keeps the active version; scan/table fixtures preserve citations.

### Phase 2 - hybrid retrieval and reranking (4-6 days)

Deliver: dense/sparse protocols, Qdrant named-vector layout, parallel TaskGroup branches, RRF, deduplication, context budgeter, reranker interface/fallback and retrieval trace.

Exit: frozen evaluation beats the dense baseline under the recorded gate; all branches enforce authorization; reranker failure is typed and degraded.

### Phase 3 - adaptive/corrective RAG (3-5 days)

Deliver: deterministic policy, evidence evaluator, one bounded rewrite/broaden path, optional HyDE flag, bounded multi-hop state and abstention/handoff.

Exit: loops always terminate; unanswerable queries abstain at the agreed level; traces explain routing.

### Phase 4 - grounding and hardening (3-5 days)

Deliver: claim/citation schema, citation checks, one repair attempt, injection/isolation fixtures, deletion flow, operator docs and acceptance evidence.

Exit: tests return no invalid citation ID; unsupported answers abstain/handoff; migration documentation is complete.

### Phase 5 - optional PoCs

Each receives its own profile, report, dependency review and rollback path: contextual retrieval, BGE-M3, ColBERT/multivectors, RAPTOR, and LightRAG/GraphRAG. None becomes default without corpus-specific evidence.

## 17. Acceptance evidence

| Requirement | Evidence |
|---|---|
| scans and tables | parser fixtures, canonical snapshots, page/locator/table citations |
| chunking choice | committed profile comparison on retrieval and citation metrics |
| embedding choice | license/hardware review and same-dataset comparison; HashEmbedding blocked in production |
| index choice | ADR covering corpus, filters, updates and SQLite/Qdrant roles |
| hybrid retrieval | trace of dense/sparse branches, RRF and authorization filters |
| reranking | profile comparison and forced-failure fallback test |
| corrective RAG | low-confidence fixture, rewrite trace and maximum-attempt test |
| citations | claim-to-citation tests and invalid-ID rejection |
| async | TaskGroup/semaphore/cancellation/bounded-queue tests |
| retry/fallback | transient/permanent classification and degradation fields |
| Pydantic | missing, wrong-type, extra-field, malformed JSON and repair-exhaustion tests |
| migration | idempotency, rollback, active-version switch and compatibility tests |
| security | cross-tenant and prompt-injection fixtures |
| evaluation | versioned JSON report and Markdown summary; explicitly no pressure-test claim |

## 18. Dependency strategy

Use existing dependencies first: Pydantic, asyncio, Tenacity, LangGraph, Docling, qdrant-client and aiosqlite.

Potential additions require a separate review:

- Ragas as an eval extra;
- a sparse encoder/local embedding package;
- a reranker package or hosted provider client;
- a tokenizer only when the chosen provider cannot expose a compatible tokenizer through current dependencies.

For every addition record license, latest release/commit, issue responsiveness, security advisories, transitive weight, platform/wheel support and the exact missing capability it supplies.

## 19. Key risks

| Risk | Mitigation |
|---|---|
| framework sprawl | keep local typed protocols; copy patterns, not platforms |
| expensive ingestion | hash caches, batching, resumable jobs, enrichments off by default |
| embedding migration corruption | profile/dimension lock, versioned collections, no silent mixing |
| contextual-prefix hallucination | store separately, version prompt, compare to plain baseline |
| graph retrieval overengineering | corpus-specific PoC only after a demonstrated baseline gap |
| uncalibrated confidence | tune on frozen labeled data and record thresholds in an ADR |
| reranker outage | fused-rank fallback with degraded marker |
| stale fallback leaks deleted data | no silent local fallback; explicit consistency/deletion contract |
| async partial activation | immutable versions and one owner for activation state |
| evaluator bias | deterministic metrics plus reviewed labeled samples and versioned judges |

## 20. Sources

Official repositories and documentation:

- https://github.com/infiniflow/ragflow
- https://github.com/docling-project/docling
- https://docling-project.github.io/docling/examples/hybrid_chunking/
- https://github.com/qdrant/qdrant
- https://qdrant.tech/documentation/concepts/hybrid-queries/
- https://github.com/microsoft/graphrag
- https://github.com/HKUDS/LightRAG
- https://github.com/deepset-ai/haystack
- https://github.com/explodinggradients/ragas
- https://github.com/AnswerDotAI/RAGatouille
- https://www.anthropic.com/news/contextual-retrieval

Papers:

- https://arxiv.org/abs/2310.11511 (Self-RAG)
- https://arxiv.org/abs/2401.15884 (CRAG)
- https://arxiv.org/abs/2403.14403 (Adaptive-RAG)
- https://arxiv.org/abs/2401.18059 (RAPTOR)
- https://arxiv.org/abs/2409.04701 (Late Chunking)
- https://arxiv.org/abs/2112.01488 (ColBERTv2)
- https://arxiv.org/abs/2212.10496 (HyDE)
- https://arxiv.org/abs/2212.10509 (IRCoT)
- https://arxiv.org/abs/2402.03216 (BGE-M3)
- https://arxiv.org/abs/2210.07316 (MTEB)
- https://arxiv.org/abs/2309.15217 (RAGAS)
- https://arxiv.org/abs/2408.08067 (RAGChecker)

This is a design proposal, not proof that these features are implemented. Implementation evidence must be added phase by phase to tests, evaluation artifacts, ADRs, API docs and docs/acceptance-evidence.md.
