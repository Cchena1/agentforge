# RAG Correctness Milestone: Implementation Evidence

**Evidence date:** 2026-08-14
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

## Evaluation assets

- `eval/golden_queries.example.jsonl`: schema-valid examples only; not a quality benchmark.
- `scripts/evaluate_rag.py`: validates query records and optionally computes source Recall@k, MRR, nDCG, and empty-result rate from ranked result JSONL.
- `docs/rag-evaluation-contract.md`: dataset, metric, release-gate, and non-goal definitions.

No retrieval-quality improvement is claimed because a representative labeled corpus has not yet been created.

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
- Parallel local Vector/Lexical/Metadata scoring with RRF fusion and deterministic rerank.
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
ruff check src tests: all checks passed
mypy src scripts: success, no issues found in 20 source files
pytest -q -p no:cacheprovider: 56 passed, 1 Starlette/httpx deprecation warning
pressure/load testing: not performed by design
```

### Not yet verified

- Real Docling PDF/DOCX extraction in this environment.
- PaddleOCR installation, page-level selective OCR, and real Chinese scan quality.
- Representative Recall@5, Citation Precision, abstention-rate, and complex-document regression thresholds.
- Remote Qdrant persistence/index tuning and any cloud parser provider.
- `tests/test_rag.py::test_retrieval_degrades_when_one_scoring_branch_fails` proves that an isolated Vector scoring failure preserves Lexical/Metadata retrieval and emits a typed degradation warning.
