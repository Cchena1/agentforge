# RAG Dependency Capability Inventory

**Review date:** 2026-08-16
**Decision scope:** correctness-first RAG engineering milestone

## Decision summary

This milestone adds no runtime dependency. Python's bundled SQLite build already provides FTS5 and BM25, while the existing dependency set provides durable version visibility, asynchronous persistence, strict contracts, optional layout-aware parsing, and optional indexed vector search.

The labeled OHR-Bench cohort showed that the former raw-overlap/Hash-dense fusion was the dominant local recall defect. Reusing SQLite FTS5 closed that gap without adding a package, model artifact, service, license, or supply-chain surface. Learned reranking, advanced query transformation, and remote sparse retrieval remain profile-gated follow-up decisions.

## Existing dependency capabilities used

| Dependency | Existing capability used | Ownership boundary | Decision |
|---|---|---|---|
| Python `sqlite3` / SQLite FTS5 | Built-in Unicode full-text index and BM25 scoring | Local lexical side index and deterministic retrieval | Reuse; no new license or package |
| `aiosqlite` | Async SQLite access, WAL mode, explicit transactions, additive schema migration | Version registry and local vector/FTS persistence | Reuse; no replacement needed |
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
| Query planning | Pydantic typed plans plus deterministic regex/token primitives; no new runtime package or model call |
| Query concurrency | `asyncio.TaskGroup`, `Semaphore`, and `timeout`; bounded to two additional queries by default |
| Cross-query fusion | Small in-process RRF keyed by verified Chunk ID; reuses existing retrieval scores and citation contracts |
| Local sparse retrieval | SQLite FTS5 virtual table with built-in BM25, transactionally updated beside canonical chunks and backfilled for older databases |
| Embedding-profile-aware fusion | Non-semantic HashEmbedding rank is excluded; semantic providers retain dense/lexical/metadata fusion |
| Candidate diversification | Deterministic source/page-first selection prevents same-page chunks from exhausting Top-K |

## Deferred candidate dependencies

These are candidates, not approved dependencies:

| Need | Candidate pattern | Admission gate |
|---|---|---|
| Remote sparse retrieval | Qdrant sparse vectors or a maintained BM25 service | Qdrant becomes a production target and the same labeled corpus proves parity/gain over SQLite FTS5 |
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

## 2026-08-14 document parser admission decision

| Dependency/project | License posture | Maintenance/security review | Capability used | Admission state |
|---|---|---|---|---|
| Docling | MIT project; transitive models/dependencies still require lockfile review | Existing `documents` extra is locked; runtime not installed in the verification environment | PDF/DOCX structure, layout provenance, tables, pictures | Approved optional primary parser; real-corpus PoC outstanding |
| pypdf | BSD-style package already in the base dependency set | Existing dependency and tests | Emergency digital-PDF text fallback | Approved with quality gate; not valid for scans/multicolumn/table claims |
| PaddleOCR / PP-StructureV3 | Apache-2.0 project; model packages must be reviewed with the chosen profile | Adapter implemented without adding an unreviewed package to the lockfile | OCR/layout/table fallback | Deferred runtime admission pending isolated PoC |
| MinerU | Repository/model license requires separate legal review | Active project, but not required by the selected runtime | Quality-rule and shadow-benchmark reference | Not admitted |
| RAGFlow | Apache-2.0 | Platform-scale dependency footprint is unnecessary here | Parser registry/job/citation patterns | Pattern reference only |
| RAG-Anything | MIT | Platform integration would exceed current scope | Asset registry and async boundaries | Pattern reference only |
| OmniDocBench | Research benchmark assets | Evaluation-only | Quality dimensions | Evaluation reference only |
| Marker | Code/model licenses vary by component/profile | Challenger only | Shadow parsing | Not admitted |
| olmOCR | Apache-2.0 project; hardware profile is the binding constraint | 6 GB local GPU is not a suitable primary path | Hard-page VLM parser | Not admitted locally |
| OpenDataLoader PDF | Apache-2.0 | Adds a Java runtime not otherwise required | Deterministic parser challenger | Deferred |

### Existing dependency capability checklist

- FastAPI: async API and OpenAPI deprecation metadata.
- Pydantic v2: strict request/response/location validation and cross-field settings validation.
- asyncio: `TaskGroup`, bounded `Semaphore`, cooperative cancellation, executor boundaries.
- aiosqlite: durable job state, version registry, memory, and local exact-search persistence.
- pypdf: emergency PDF text extraction only.
- Docling optional extra: primary PDF/DOCX parser adapter.
- Qdrant optional extra: HNSW and metadata filtering for larger corpora.
- Tenacity: bounded transient model retry; not used to hide deterministic parser quality failures.

No new runtime dependency was added in this milestone. PaddleOCR remains an explicit candidate rather than an undeclared installation requirement.

## Query transformation pattern review (2026-08-15)

| Project/paper | Maintenance or provenance signal | License / reuse decision | Applied pattern |
|---|---|---|---|
| LangChain / LangGraph | Maintained upstream ecosystem already present in this project | MIT; pattern-level reuse | History-aware routing: do not pay rewrite cost when no contextualization is needed |
| LlamaIndex `SubQuestionQueryEngine` | Maintained upstream implementation | MIT; pattern-level reuse | Bounded explicit subqueries and asynchronous execution |
| CRAG | Primary research paper | Research concept | Evidence-quality gate before corrective retrieval |
| Rewrite-Retrieve-Read | Primary research paper | Research concept | Retriever-oriented rewrite objective |
| HyDE / Query2doc | Primary research papers | Deferred profile | Rejected from default path until labeled quality gain offsets latency, cost, and drift |
| NirDiamant/RAG_Techniques | Popular reference repository | Non-commercial terms conflict with MIT/commercial portfolio use | Concepts may be reviewed; source code must not be copied |

No dependency was added. Existing Pydantic, asyncio, LangGraph context, and vector-store contracts cover the selected minimum implementation.

## Observability and recovery dependency review (2026-08-15)

| Dependency/tool | License posture | Maintenance/admission evidence | Capability used | Decision |
|---|---|---|---|---|
| `prometheus-client` 0.26.0 | Apache-2.0 | Mature official Python client; locked by `uv.lock` | Custom registry, Counter/Gauge/Histogram, exposition | Added to base runtime |
| OpenTelemetry API/SDK/exporter 1.44.0 | Apache-2.0 | Official maintained SDK/exporter family; locked by `uv.lock` | TraceProvider, FastAPI instrumentation, local/OTLP export | Added to base runtime |
| `opentelemetry-instrumentation-fastapi` 0.65b0 | Apache-2.0 | Official contrib instrumentation; beta version is pinned and tested | HTTP server spans and trace context | Added; pinned, upgrade requires regression |
| Prometheus `promtool` 3.13.1 | Apache-2.0 | Official release binary, used only from Git ignored `state/tools` | Config/rule validation and rule unit tests | Test tool, not committed |
| Alertmanager `amtool` 0.32.1 | Apache-2.0 | Official release binary, used only from Git ignored `state/tools` | Alertmanager config validation | Test tool, not committed |
| Python `sqlite3` | PSF / SQLite public-domain component | Standard library, already present | Online Backup API and integrity checks | Reused; no new package |
| Qdrant snapshots | Apache-2.0 server/client ecosystem | Existing optional vector dependency | Future remote-vector disaster recovery | Runtime integration deferred; current CLI fail-closed |

Security and cost boundaries:

- No SaaS exporter is enabled by default.
- OTLP endpoint is explicit configuration; API secrets are not emitted.
- Prometheus labels exclude user-controlled identifiers and text.
- Trace files are bounded by size and backup count.
- Official tool archives remain under `state/` and are excluded from Git; benchmark evidence remains small.


## DeepSeek Harness pattern review (2026-08-16)

| Candidate | License / status | Reused capability | Decision |
|---|---|---|---|
| DeepSeek Harness | MIT; official repository labels it Developer Preview | Tool pre-execution policy, approval/guard stages, capability-filtered Subagent, bounded context pattern | Pattern adopted; package not added and code not copied |
| Existing Pydantic | Already pinned in project | Strict argument and runtime contract validation | Reused; no dependency change |
| Existing asyncio / LangGraph | Already pinned in project | Tool DAG concurrency, bounded loops and state ownership | Reused; no dependency change |
| Existing Prometheus client | Already pinned in project | Bounded-cardinality Tool runtime metrics | Reused; no dependency change |
| Model-specific tokenizer | Not present | More accurate context accounting | Rejected for now; add only after model-family and error-budget requirements are defined |

Review conclusion: the improvement is architectural reuse, not a new runtime dependency. This avoids duplicate orchestration stacks and keeps the current `uv.lock` unchanged.

## 2026-08-16 FTS5/BM25 admission review

| Review item | Finding |
|---|---|
| Capability gap | Raw whitespace overlap and non-semantic Hash dense ranking reduced OHR-Bench Recall@5 |
| Existing capability | The project Python runtime ships SQLite 3.53.1 with FTS5/BM25 enabled |
| License | SQLite is already part of the runtime; no new package or model license enters the project |
| Maintenance/security | Uses the runtime SQLite implementation rather than a separately vendored search library |
| Persistence impact | Additive `document_chunks_fts` virtual table; canonical chunk rows remain the source of truth |
| Migration | Initialization backfills when canonical/FTS row counts differ; application writes update both in one transaction |
| Fallback | FTS technical failure falls back to the legacy overlap scorer and records `degraded_retrieval` |
| Evidence | Third 200-page Benchmark: Recall@5 95.53% clean, 95.53% formatting noise, 88.91% semantic/OCR noise |

Official design references used during the review: SQLite FTS5/BM25 documentation and Qdrant Hybrid Queries documentation. The latter is a future adapter reference, not evidence that the current Qdrant path is hybrid.


## 2026-08-18 Graph-aware retrieval admission review

| Review item | Finding |
|---|---|
| Capability gap | Hybrid retrieval did not use existing `parent_id` / `heading_path` relationships for bridge evidence |
| Microsoft GraphRAG | MIT-licensed official project; full indexing includes entity/relationship extraction and community workflows |
| LangGraph Agentic RAG | Official workflow reference for bounded retrieve/grade/rewrite control flow |
| Existing capability | Parent-child chunks, active-version filtering, tenant/ACL filtering, Pydantic contracts and asyncio timeouts already exist |
| Decision | Add plugin protocols and SQLite one-hop structural expansion; do not add a graph database or LLM indexing dependency |
| Persistence impact | None; reuse canonical `metadata_json`; no SQLite schema migration |
| Backend gap | Qdrant structural-neighbor adapter remains unsupported and fails open to verified direct retrieval with a warning |
| Revisit trigger | Failed cases must demonstrate cross-document entity/community reasoning that structural one-hop expansion cannot solve |


## 2026-08-19 GraphRAG / Semantic Embedding 依赖评审

| 依赖或项目 | 用途 | 许可证/维护判断 | 决策 |
|---|---|---|---|
| `sentence-transformers>=5,<6`（锁定解析为 5.7.0） | 本地 Query/Document 语义向量接口 | Apache-2.0；仅作为 `semantic` extra，避免基础安装强制拉取模型运行栈 | 采用 |
| `BAAI/bge-m3` | 中英文及多语言 Dense Embedding，1024 维 | MIT；模型卡声明 8192 Token 与多检索模式；本项目只使用 Dense | 采用默认模型，真实资源占用待验证 |
| Microsoft GraphRAG | 实体关系索引、Local/Global/DRIFT 架构参考 | 官方研究项目；完整索引成本较高 | 借鉴架构，不引入 Runtime |
| Neo4j GraphRAG for Python | KG Builder 和 Retriever 插件边界参考 | 官方持续维护；需要 Neo4j 服务与额外运维 | 不作为默认依赖，保留 Store Adapter 扩展点 |
| `aiosqlite`（已有） | Local-first 版本化图存储 | 已在项目中使用，无新增依赖 | 用于最小 PoC |

新增依赖不复制第三方源码；所有实现基于公开接口与架构模式重新落地。若未来引入 Neo4j Adapter，必须单独评审服务许可证、备份恢复、权限模型和部署成本。
