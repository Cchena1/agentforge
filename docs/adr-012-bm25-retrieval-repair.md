# ADR-012: BM25-first minimal retrieval repair

- **Status:** Accepted
- **Date:** 2026-08-16
- **Scope:** SQLite retrieval, fusion policy, evidence sufficiency, result diversity, benchmark methodology

## Context

The first 200-page benchmark exposed a retrieval failure rather than a parser-only failure: ground-truth text reached Recall@5 73.30%, formatting-noise text 70.26%, and semantic/OCR-noise text 57.38%. Diagnostic replay showed that the default deterministic HashEmbedding was not semantic, yet its cosine score owned 50% of the fusion weight and could outvote the stronger lexical branch. The lexical branch itself used raw whitespace overlap without IDF, document-length normalization, or punctuation-aware tokenization.

The repair must remain local-first, bounded, migration-safe, and small enough to validate before the third benchmark. No load or throughput testing is in scope.

## Mature patterns reviewed

1. SQLite FTS5 provides a maintained in-process full-text index and built-in BM25 ranking, so the local backend does not need a custom BM25 implementation: <https://www.sqlite.org/fts5.html>.
2. Qdrant documents dense+sparse prefetch followed by RRF/DBSF fusion as its production hybrid-search pattern: <https://qdrant.tech/documentation/concepts/hybrid-queries/>. This is the target for a later Qdrant PoC, not something to emulate by scanning all payloads.
3. Existing project dependencies already include SQLite through Python/aiosqlite. FTS5 is available in the locked Windows runtime (SQLite 3.53.1), so no new package or license surface is introduced.

## Decision

### Local SQLite path

- Add an additive 'document_chunks_fts' FTS5 table with 'unicode61' tokenization.
- Rank lexical candidates with built-in 'bm25()'.
- Keep FTS rows transactionally synchronized with chunk upsert and source deletion.
- Backfill the FTS table when an older database is opened and row counts differ. This is an additive migration; existing 'document_chunks' data remains authoritative.
- Preserve the old overlap scorer only as an explicit bounded fallback when FTS5 fails, and emit 'degraded_retrieval:fts5:<error>'.

### Fusion

- Treat HashEmbedding as an offline deterministic baseline, not a semantic signal. Its direct vector weight and vector contribution to RRF are disabled.
- For semantic embedding providers, retain dense+lexical+metadata fusion with balanced weights.
- Continue using rank fusion so score scales from independent branches do not need to be numerically identical.

### Diversity and corrective retrieval

- Prefer distinct '(source_id, page)' evidence groups before filling remaining slots with same-page chunks. This reduces top-k capacity loss without prohibiting multi-chunk evidence when the candidate set is small.
- Evidence sufficiency is based on the fused rerank score; a single non-zero lexical overlap no longer suppresses corrective retrieval.

## Rejected alternatives

- **Custom Python BM25:** rejected because SQLite FTS5 already supplies a mature implementation and index.
- **LLM query rewriting as the first fix:** rejected because it adds latency, drift, and token cost while the baseline ranker is demonstrably misweighted.
- **Cross-encoder reranking now:** deferred until the BM25 repair is benchmarked; it adds a model/runtime dependency and cannot repair missing lexical candidates.
- **Qdrant payload scan for pseudo-BM25:** rejected because it does not scale and would weaken the authorization-before-scoring boundary.

## Adversarial review gates

The repair may proceed to the third 200-page benchmark only if all of the following pass:

- a deliberately misleading HashEmbedding cannot outvote an exact BM25 match;
- FTS backfill, update, and delete remain synchronized with the canonical chunk table;
- tenant and ACL filtering still occurs before vector decoding and before FTS results are returned;
- one scoring branch can fail without failing the entire retrieval request;
- page diversity does not reduce the requested top-k when fallback candidates exist;
- full pytest, Ruff, and strict mypy gates pass.

## Known boundary

The Qdrant backend remains dense-first in this minimal repair. Production Qdrant hybrid retrieval requires a separate sparse-vector capability PoC and schema migration; this benchmark evaluates the default SQLite local path only.
