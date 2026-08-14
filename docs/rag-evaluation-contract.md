# RAG Evaluation Contract

**Status:** contract established; corpus population and baseline measurement are pending.
**Date:** 2026-08-13
**Scope exclusion:** no load, stress, throughput, p95, or p99 testing.

## Purpose

Ranking techniques are accepted only when they improve a frozen, versioned domain dataset without regressing authorization, citation validity, abstention, migration, or failure semantics.

## Query record schema

Each line in `eval/golden_queries.jsonl` must validate against the following logical schema:

```json
{
  "query_id": "stable identifier",
  "query": "user question",
  "tenant_id": "tenant or evaluation namespace",
  "acl": ["allowed principal"],
  "source_ids": ["optional source filter"],
  "relevant_source_ids": ["at least one source for answerable queries"],
  "relevant_chunk_ids": ["optional exact labels"],
  "answerable": true,
  "expected_citation_source_ids": ["source IDs"],
  "tags": ["scan", "table", "identifier", "multilingual", "multi_hop", "prompt_injection"],
  "notes": "reviewer rationale"
}
```

The initial corpus must contain both answerable and unanswerable cases and must include digital text, tables, scanned documents, Chinese/English queries, exact identifiers, conflicting evidence, prompt injection, and authorization boundaries.

## Metrics

Deterministic retrieval metrics:

- Recall@k
- MRR
- nDCG@k
- source-level recall
- duplicate rate
- empty-result rate
- unauthorized-result rate

Answer/citation metrics after generation is added:

- labeled answer correctness
- claim citation coverage
- citation precision and recall
- invalid citation rate
- groundedness/faithfulness
- abstention accuracy

Correctness gates independent of ranking:

- idempotent re-ingestion;
- active-version atomicity;
- parser fallback classification;
- embedding dimension mismatch rejection;
- tenant/ACL isolation;
- bounded corrective-loop termination;
- malformed structured-output fallback.

## Profile comparison rules

1. Commit the corpus and evaluator version.
2. Record parser, chunker, embedding, sparse encoder, vector index, fusion, reranker, prompt, and model profiles.
3. Run the current dense/lexical baseline first.
4. Compare one architectural variable at a time where practical.
5. Do not promote a profile solely because it wins an aggregate score; inspect failures by tag.
6. Any unauthorized result, invalid citation, or non-terminating loop is a release blocker regardless of average quality.
7. Store machine-readable output under `artifacts/rag-eval/<date-profile>/report.json` and a human review summary beside it.

## Initial baseline status

The existing automated suite verifies parser behavior, citations, SQLite retrieval, and Qdrant round trips. It is not a domain relevance benchmark. Baseline measurement remains pending until representative source documents and human-reviewed relevance labels are supplied or created.

## Non-goals

This evaluation contract does not measure or claim:

- throughput;
- maximum concurrent users;
- p50/p95/p99 latency;
- saturation behavior;
- capacity planning.

Normal functional tests may record single-operation duration for diagnosis only.
