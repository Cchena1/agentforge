# ADR-008: Quality-Gated Word/PDF Parser Routing

- Status: Accepted
- Date: 2026-08-14
- Owners: document ingestion and RAG modules

## Context

The repository previously flattened Docling output to Markdown and immediately discarded layout, reading order, assets, and parser provenance. That approach could not reliably distinguish digital PDFs from scans, preserve Word structural anchors, or prove that a citation came from the active source version.

## First-principles invariants

1. The original file is the final evidence source; Markdown is derived data.
2. One ingestion produces one authoritative parsed document. Outputs from different parsers are never concatenated into a synthetic body.
3. Parser routing is deterministic and quality-gated; an LLM cannot choose a parser.
4. Every retry, fallback, corrective-retrieval loop, and agent loop is bounded.
5. Low-quality evidence is not silently indexed.
6. Document content is untrusted data and cannot alter system or tool policy.
7. Persistent and public contracts use additive migration before removal.

## Decision

Use the following bounded route:

1. Docling is the primary PDF/DOCX parser and Canonical Document Model reference.
2. PaddleOCR `PPStructureV3` is implemented as an optional OCR/layout adapter. It remains disabled until an isolated environment PoC and dependency review are complete.
3. The built-in pypdf path is an emergency PDF fallback and is accepted only through the same deterministic quality gate.
4. `.doc` and `.docm` are rejected with a `.docx` conversion instruction.
5. Cloud parsing is represented by a provider-neutral protocol. No provider is bound in v0.3; enabling the configuration flag is rejected at startup.
6. A document may make at most three parser attempts. Exhaustion produces `needs_review` rather than an active index.
7. Canonical blocks retain page/section anchors, reading order, bounding boxes when available, asset relationships, parser attempts, and quality metrics.

## Open-source pattern review

| Project | Pattern adopted | Runtime decision |
|---|---|---|
| Docling | Native document structure, layout provenance, tables, pictures, HybridChunker principles | Primary optional runtime |
| PaddleOCR / PP-StructureV3 | OCR, layout, table and chart fallback | Optional adapter; PoC gate required |
| MinerU | Repeating header/footer suppression and reading-order quality criteria | Shadow benchmark only; separate license review required |
| RAGFlow | Parser registry, asynchronous job state, grounded citations | Module-boundary reference only |
| RAG-Anything | Asset registry, asynchronous cache/batch boundaries | Asset ownership pattern adopted |
| OmniDocBench | Text/table/formula/reading-order quality dimensions | Evaluation design reference only |
| Marker | JSON/Markdown challenger parser and optional repair | Not in production route; license/profile review required |
| olmOCR | VLM parsing for hard PDFs | Not a local default for the available 6 GB GPU |
| OpenDataLoader PDF | Deterministic local layout parsing | Deferred to avoid adding a Java runtime for one parser |

No third-party repository was copied wholesale. The implementation uses repository-owned adapters and contracts to avoid license contamination and infrastructure overreach.

## Canonical schema version

The canonical document contract starts at schema version 1. Indexed chunks persist that version and include it in the pipeline profile. Incompatible future revisions require rebuild-and-activate migration; parsers and readers must not guess an in-place conversion.

## Consequences

### Positive

- Parser failures and quality failures are observable and bounded.
- PDF and DOCX provenance survives chunking and citation generation.
- OCR is opt-in instead of an undeclared dependency.
- Index activation remains atomic and prior evidence remains available after replacement failure.

### Costs and limitations

- Current CI does not install Docling or PaddleOCR, so real complex-document quality remains unverified.
- PaddleOCR is currently a document-level fallback adapter; page crop/orientation/resolution retry and selective page replacement remain future work.
- DOCX comments, revisions, text boxes, and footnote fidelity require a real corpus before stronger support claims.

## Revisit triggers

Revisit this ADR only when one of the following occurs:

- a representative corpus proves a common DOCX class cannot be recovered by Docling;
- page-level OCR replacement is implemented and measured against labeled fixtures;
- a candidate parser passes license, maintenance, security, and isolated quality review;
- the deployment hardware or privacy boundary changes enough to justify a VLM/cloud parser.

## Verification

- `tests/test_document_pipeline.py`
- `tests/test_api.py::test_async_ingestion_job_api_completes_and_legacy_route_is_deprecated`
- `tests/test_rag.py` version, tenant, ACL, and activation tests