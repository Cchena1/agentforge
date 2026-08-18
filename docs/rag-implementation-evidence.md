# RAG Correctness Milestone: Implementation Evidence

**Evidence date:** 2026-08-16
**Scope:** functional, contract, migration, tenant/ACL isolation, concurrency, and static verification
**Explicit exclusion:** no pressure, throughput, saturation, p95/p99, or sustained-load testing

## Implemented behavior

1. Content-addressed, immutable document versions.
2. Pipeline identity includes parser, chunker configuration, embedding model/profile, and dimension.
3. Idempotent ingestion for unchanged content and pipeline profiles.
4. Durable SQLite version registry with `building`, `active`, `retired`, and `failed` lifecycle states.
5. Activation only after all candidate vectors are stored.
6. Retrieval visibility restricted to active versions while retaining unversioned compatibility for unmanaged sources.
7. Same-source ingestion serialization with cross-source asynchronous overlap.
8. Additive API fields for version evidence without removing legacy fields.
9. Strict embedding vector-count and dimension validation before persistence.
10. JSONL/Pydantic retrieval evaluation contract and a metric script for labeled ranked-result files.
11. Composite `(tenant_id, source_id)` identity with tenant-wide or principal-intersection ACL semantics.
12. SQLite authorization filtering before vector decoding and Qdrant authorization filtering inside the vector query.
13. Additive migration of legacy SQLite/Qdrant records to public-tenant semantics.
14. Additive SQLite FTS5 side index with BM25 scoring, backfill, transactional update, and delete behavior.
15. Embedding-profile-aware fusion that prevents non-semantic HashEmbedding ranks from outvoting lexical evidence.
16. Source/page-first candidate diversification before duplicate-page fill.
17. Fused-score evidence gating with bounded corrective retrieval.
18. Third 200-page OHR-Bench run with a 150-page longitudinal cohort and 50-page disjoint holdout.

## Claim-to-test evidence

| Claim | Evidence |
|---|---|
| A failed replacement cannot erase the searchable version | `test_failed_replacement_keeps_previous_version_searchable` |
| A failed first build cannot leak partial chunks | `test_failed_first_ingest_does_not_expose_partial_version` |
| Candidate chunks are invisible until activation | `test_building_version_is_invisible_until_activation` |
| Identical ingestion is idempotent | `test_reingesting_identical_content_is_idempotent` |
| Legacy SQLite data migrates in place | `test_existing_unversioned_sqlite_index_migrates_without_data_loss` |
| Registry versions migrate into a composite-foreign-key v2 table | `test_registry_migrates_legacy_rows_to_public_tenant` |
| Composite tenant identity and public dual-write remain foreign-key valid | `test_registry_v2_uses_composite_identity_and_public_dual_write` |
| A downgrade/upgrade cycle reconciles newer legacy public state | `test_registry_reconciles_public_writes_after_legacy_rollback` |
| Legacy evidence remains visible while its first version is building | `test_legacy_rows_remain_visible_until_first_version_activates` |
| Same-source writes do not overlap | `test_same_source_ingests_are_serialized` |
| Different-source writes retain async overlap | `test_different_source_ingests_can_overlap` |
| Wrong embedding dimensions fail before indexing | `test_embedding_dimension_mismatch_is_rejected_before_indexing` |
| A file changed during ingestion is rejected before version registration | `test_source_mutation_during_ingestion_is_rejected` |
| SQLite retrieval returns active versions only | RAG lifecycle tests in `tests/test_rag.py` |
| Qdrant payload filtering returns active versions only | `test_qdrant_search_exposes_only_active_version` |
| Cross-tenant retrieval cannot return another tenant's rows | `test_cross_tenant_retrieval_isolated_before_scoring` |
| Same source ID is independently scoped per tenant | `test_same_source_id_is_isolated_per_tenant` |
| Non-empty ACL requires principal intersection and hides version metadata | `test_acl_requires_principal_intersection_and_hides_index_versions` |
| Restricted activation cannot leak legacy rows | `test_restricted_active_version_suppresses_legacy_rows_for_unauthorized_principal` |
| SQLite authorization precedes vector decoding/scoring | `test_sqlite_authorization_filter_runs_before_vector_decoding` |
| Qdrant applies tenant/ACL filters and public-only legacy semantics | Qdrant tenant/ACL and legacy migration tests in `tests/test_qdrant.py` |
| Public HTTP contracts expose additive tenant/ACL fields | API tenant/ACL and same-source/multi-tenant tests in `tests/test_api.py` |
| Existing API fields remain while version evidence is added | `test_document_ingest_adds_version_fields_without_removing_legacy_fields` |
| Evaluation input rejects malformed records | `tests/test_rag_evaluation.py` |
| Misleading Hash rank cannot outvote BM25 evidence | `test_hash_baseline_does_not_outvote_bm25_relevance` |
| FTS backfill, update, and delete stay consistent | `test_fts_index_backfills_updates_and_deletes_with_chunks` |
| Top-K covers distinct pages before duplicate chunks | `test_retrieval_diversifies_pages_before_filling_duplicate_chunks` |

## Evaluation assets

- `eval/golden_queries.example.jsonl`: schema-valid examples only; not a quality benchmark.
- `scripts/evaluate_rag.py`: validates query records and optionally computes source Recall@k, MRR, nDCG, and empty-result rate from ranked result JSONL.
- `docs/rag-evaluation-contract.md`: dataset, metric, release-gate, and non-goal definitions.

Retrieval-quality claims are now limited to the frozen external Benchmark evidence in `eval/基于OmniDocBench和OHR-Bench的RAG基准测试v3/`. They do not establish production-corpus quality or answer-level citation precision.

## Latest local verification

The final target-repository verification should run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='<repository>\src;<repository>'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m ruff check --no-cache src tests scripts
.venv\Scripts\python.exe -m mypy --cache-dir '<writable-cache>' src scripts
.venv\Scripts\python.exe scripts\evaluate_rag.py --queries eval\golden_queries.example.jsonl
```

Results are updated in the root `README.md` after the target copy is verified.

## Known limits

- Ingestion is request/response based; no durable job queue, worker lease, cancellation API, or progress endpoint exists yet.
- Per-source write locks are process-local. A production multi-writer deployment requires a distributed lease or single ingestion writer.
- Candidate vector rows can remain orphaned after a crash between upsert and activation; they are invisible but need a retention/garbage-collection policy.
- The registry and vector backend cannot share one cross-system transaction when Qdrant is used; visibility correctness is provided by the registry filter rather than atomic storage across systems.
- Authentication is not implemented; production callers must receive trusted tenant/principal context from an identity boundary. The chat tool is public-tenant only.
- Parser profile identity is conservative and can trigger re-ingestion when optional parser package versions change.
- Remote Qdrant operation, production OCR quality, provider embeddings, and learned reranking are not verified.
- The example golden file is not representative enough to set release thresholds.

## Document RAG v0.3 milestone evidence (2026-08-14)

### Implemented

- Parser-neutral Canonical Document Model with PDF/DOCX/text locations, assets, parser attempts, and quality reports.
- Docling primary adapter, optional PaddleOCR adapter, bounded pypdf fallback, and explicit DOC/DOCM rejection.
- Deterministic quality gate; low-quality output raises `DocumentNeedsReviewError` and is not activated.
- Token-aware Parent-Child chunking with page artifact exclusion and table row-group splitting with repeated headers.
- Durable asynchronous ingestion jobs with create/get/cancel APIs, restart recovery, bounded parallelism, and stage events.
- Additive deprecation of the blocking ingestion API.
- Parallel local Vector/FTS5-BM25/Metadata scoring with embedding-profile-aware fusion and deterministic source/page diversification.
- Code-generated citations containing exact evidence quotes, structured locations, content hashes, parser identity, and active-version checks.
- Cross-field Pydantic settings validation and strict citation location validation.

### Claim-to-test map

| Claim | Evidence |
|---|---|
| Fallback selects one authoritative parser and stops after acceptance | `test_quality_gate_uses_one_authoritative_fallback_and_records_attempts` |
| Parser retries are bounded and low-quality evidence is not accepted | `test_parser_attempts_are_bounded_and_low_quality_content_is_not_indexable` |
| DOC/DOCM are rejected explicitly | `test_unsafe_or_legacy_word_formats_are_explicitly_rejected` |
| Headers/page numbers are excluded and table headers survive splits | `test_parent_child_chunking_excludes_page_artifacts_and_preserves_table_headers` |
| Wrong-typed provenance is rejected before business logic | `test_citation_schema_rejects_missing_or_wrong_typed_pdf_provenance` |
| Async job completes and blocking route exposes deprecation | `test_async_ingestion_job_api_completes_and_legacy_route_is_deprecated` |
| Missing jobs are explicit 404s | `test_async_ingestion_job_not_found_is_explicit` |
| Failed replacement preserves the active version | `test_failed_replacement_keeps_previous_version_searchable` |
| Unauthorized content is removed before scoring | `test_sqlite_authorization_filter_runs_before_vector_decoding` |

### Latest local verification

```text
ruff check src main.py tests scripts benchmark-scripts: all checks passed
mypy src scripts: success, no issues found in 27 source files
pytest -q -p no:cacheprovider: 97 passed, 1 Starlette/httpx deprecation warning
benchmark adversarial audit: pass, 200 unique pages and 559 queries per variant
pressure/load testing: not performed by design
```

### Not yet verified

- Real Docling PDF/DOCX extraction in this environment.
- PaddleOCR installation, page-level selective OCR, and real Chinese scan quality.
- Production-corpus Recall@5, answer-level Citation Precision, abstention-rate, and complex-document regression thresholds beyond the frozen external fixtures.
- Remote Qdrant persistence/index tuning and any cloud parser provider.
- `tests/test_rag.py::test_retrieval_degrades_when_one_scoring_branch_fails` proves that an isolated Vector scoring failure preserves Lexical/Metadata retrieval and emits a typed degradation warning.

## 2026-08-15 minimal benchmark remediation

This change intentionally keeps the existing 200-page result files immutable and fixes the next-run execution contract:

1. Docling and PaddleOCR backend objects now reuse one initialized converter/pipeline per backend instance and serialize access through an async lock. This removes per-document model construction while preserving an asynchronous API boundary.
2. A configured cloud parser remains reachable within the three-attempt budget; when all four PDF backends are configured, the final slot is reserved for the explicitly enabled cloud fallback instead of silently truncating it.
3. Parser `TimeoutError` is recorded as a failed attempt and can continue through the bounded fallback sequence.
4. SQLite RRF uses bounded branch candidate lists. Lexical and metadata branches exclude zero-score rows, preventing insertion-order rank credit for non-matches.
5. The benchmark writes the canonical `content_sha256` field and validates it against the complete source-page text hash. Existing reports are historical evidence and are not retroactively changed.

No new runtime dependency was added. Real Docling model loading, OCR, cloud fallback, semantic embedding, learned reranking, and benchmark quality gains remain unverified until the next controlled run.

## 2026-08-15 ten-page failed-PDF smoke evidence

A deterministic smoke set selected one previously failed OmniDocBench page from each of ten strata. With Python UTF-8 mode, a completed Docling model cache, `DOCLING_INFERENCE_COMPILE_TORCH_MODELS=false`, and one reused backend instance, all 10 pages parsed and passed the current structural quality gate. The mean character trigram F1 was only 0.4812, including false acceptances for hard tables and mixed-language notes. The evidence therefore validates parser lifecycle recovery but disproves the assumption that structural acceptance alone establishes extraction fidelity.

Evidence: `eval/基于OmniDocBench和OHR-Bench的RAG基准测试/results/omni_failed_10_smoke_2026-08-15.json`.

## 2026-08-15 minimal PDF content-completeness guard

The same ten-page failed-PDF set was rerun after a bounded safety fix. Parsing remained successful for 10/10 pages, while the deterministic quality gate rejected four sparse visual/table outputs that the baseline had falsely accepted. The gate uses only production-observable signals: indexable characters after removing running artifacts, real page inventory, visual/table assets, and extracted table text. It does not use benchmark ground truth for routing.

Additional protections:

- Docling page and empty-page counts now use the document page inventory instead of only successfully extracted blocks.
- Windows Docling fails fast before model loading when Python UTF-8 mode is absent.
- When Docling torch compilation is enabled but MSVC `cl.exe` is unavailable, the backend disables compilation through Docling settings and emits a typed warning.
- Sparse visual/table output routes to the existing bounded OCR fallback or manual review; no retry or attempt limit changed.

Evidence: `eval/基于OmniDocBench和OHR-Bench的RAG基准测试/results/omni_failed_10_smoke_minimal_fix_2026-08-15.json`.

Residual limitation: two low-F1 pages without a sufficiently strong visual/table sparsity signal remain accepted. Detecting those requires layout-coverage evidence or a challenger parser and is intentionally outside this minimal fix.


## Minimal completeness gate verification (2026-08-15)

The minimal fix closes a concrete false-acceptance path without replacing the parser or adding a dependency:

- running headers, footers, and page numbers no longer count toward indexable text coverage;
- a PDF with visual assets and fewer than 160 indexable characters per page is rejected with `LOW_TEXT_COVERAGE_WITH_VISUAL_ASSETS`;
- a PDF with a table signal and fewer than 160 extracted table characters is rejected with `SPARSE_TABLE_EXTRACTION`;
- rejected pages receive `OCR_REQUIRED` and follow the bounded OCR/manual-review route;
- Docling page inventory, rather than only successfully extracted blocks, determines the page and empty-page counts;
- Windows Docling startup fails fast without UTF-8 mode and disables optional Torch compilation when MSVC `cl.exe` is unavailable.

Verification evidence:

```text
pytest: 65 passed, 1 upstream deprecation warning
ruff: all checks passed for src, main.py, tests, scripts, and benchmark scripts
mypy: success, no issues found in 20 source files
pressure/load test: not performed
```

The frozen same-set regression is `eval/基于OmniDocBench和OHR-Bench的RAG基准测试/results/omni_failed_10_smoke_minimal_fix_2026-08-15.json`. All 10 pages still parsed, while the gate rejected `omni-031`, `omni-016`, `omni-021`, and `omni-036`. Mean character trigram F1 remained 0.4812, so this change is evidence of safer acceptance behavior, not improved extraction fidelity.

## 2026-08-16 BM25 retrieval repair and third 200-page Benchmark

The minimum repair reused SQLite FTS5/BM25 rather than adding a retrieval dependency. On the original 150-page/427-query cohort, Recall@5 changed as follows:

| Variant | Before | After | Delta |
|---|---:|---:|---:|
| Ground Truth | 73.30% | 94.38% | +21.08 pp |
| Formatting Noise | 70.26% | 94.38% | +24.12 pp |
| Semantic/OCR Noise | 57.38% | 86.42% | +29.04 pp |

A disjoint 50-page/132-query holdout reached Recall@5 of 99.24%, 99.24%, and 96.97% for the same variants. The aggregate 200-page/559-query result was 95.53%, 95.53%, and 88.91%. Hit-level Citation Validity was 100%, which means schema/source/quote consistency only; it is not claim-level semantic support.

Verification:

```text
pytest: 97 passed, 1 upstream Starlette/httpx deprecation warning
ruff: all checks passed
mypy strict: success, 27 source files
benchmark adversarial audit: pass, 200 unique pages, 559 queries per variant
pressure/load test: not performed
```

Residual limitations are Finance semantic/OCR retrieval, dense-first Qdrant operation, non-learned reranking, and the absence of answer-level citation entailment evaluation.
