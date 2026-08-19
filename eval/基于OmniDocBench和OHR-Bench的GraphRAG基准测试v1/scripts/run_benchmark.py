from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_service.document_models import ParsedBlock, ParsedDocument
from agent_service.document_pipeline import ParentChildChunker
from agent_service.graph_retrieval import (
    FallbackGraphRetriever,
    KnowledgeGraphRetriever,
    StoreBackedStructuralGraphRetriever,
)
from agent_service.knowledge_graph import (
    GraphChunk,
    HeuristicKnowledgeGraphExtractor,
    KnowledgeGraphBuildContext,
    KnowledgeGraphBuilder,
    SQLiteKnowledgeGraphStore,
)
from agent_service.query_planning import DeterministicQueryPlanner
from agent_service.rag import RAGService
from agent_service.rag_reflection import DeterministicEvidenceReflector
from agent_service.rag_registry import InMemoryVersionRegistry
from agent_service.vector_store import SQLiteVectorStore, VectorDocument

VARIANTS = (
    "gt_text",
    "formatting_noise_moderate",
    "semantic_noise_MinerU_moderate",
)
ARMS = ("hybrid_baseline", "graphrag_fallback")
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {
    "academic": 15,
    "administration": 14,
    "finance": 15,
    "law": 14,
    "manual": 14,
    "news": 14,
    "textbook": 14,
}
MODEL_NAME = "BAAI/bge-m3"
MODEL_DIMENSION = 1024
TOP_K = 5
TENANT_ID = "benchmark"


@dataclass(slots=True)
class SourceBundle:
    sample: dict[str, Any]
    chunks: list[Any]
    version_id: str
    content_sha256: str


class BenchmarkParser:
    profile_id = "benchmark-preextracted-pdf-page-v1"

    async def parse(self, path: Path) -> ParsedDocument:
        raise RuntimeError(f"benchmark retrieval does not parse files at query time: {path}")


class CachedEmbeddingProvider:
    dimension = MODEL_DIMENSION
    profile_id = f"sentence-transformers:{MODEL_NAME}:{MODEL_DIMENSION}:benchmark-cache-v1"

    def __init__(self, query_vectors: dict[str, list[float]]) -> None:
        self.query_vectors = query_vectors

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("document vectors must be precomputed in one bounded batch")

    async def embed_query(self, query: str) -> list[float]:
        vector = self.query_vectors.get(query)
        if vector is None:
            raise KeyError(f"query embedding was not precomputed: {query!r}")
        return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 100-page AgentForge GraphRAG benchmark.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_version_id(sample_id: str, variant: str, content_sha256: str) -> str:
    digest = sha256_text(f"{TENANT_ID}\0{sample_id}\0{variant}\0{content_sha256}")[:24]
    return f"rv_{digest}"


def load_source_pool(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    roots = (
        (repo_root / "state" / "benchmark-200p", "longitudinal_150"),
        (repo_root / "state" / "benchmark-200p-v3-holdout", "unseen_holdout_50"),
    )
    samples: list[dict[str, Any]] = []
    qas: dict[str, dict[str, Any]] = {}
    for root, cohort in roots:
        manifest_path = root / "ohr_manifest.json"
        qas_path = root / "ohr_qas_v2.json"
        if not manifest_path.exists() or not qas_path.exists():
            raise FileNotFoundError(
                "Ignored OHR-Bench state is required. Expected "
                f"{manifest_path} and {qas_path}."
            )
        for item in read_json(manifest_path):
            sample = dict(item)
            sample["cohort"] = cohort
            samples.append(sample)
        for item in read_json(qas_path):
            qas[str(item["ID"])] = item
    if len(samples) != 200:
        raise RuntimeError(f"expected 200 source-pool pages, got {len(samples)}")
    return samples, qas


def select_pages(
    repo_root: Path,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], set[str], set[str]]:
    query_pages: dict[str, set[str]] = defaultdict(set)
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    for sample in samples:
        for qa_id in sample["qa_ids"]:
            query_pages[str(qa_id)].add(str(sample["sample_id"]))

    multi_page_ids = {
        source_id
        for relevant in query_pages.values()
        if len(relevant) > 1
        for source_id in relevant
    }
    prior_results = read_json(
        repo_root
        / "eval"
        / "基于OmniDocBench和OHR-Bench的RAG基准测试v3"
        / "results"
        / "ohr_retrieval_semantic_noise_MinerU_moderate.json"
    )
    failed_page_ids = {
        source_id
        for row in prior_results
        if row.get("first_relevant_rank") is None
        for source_id in row["relevant_source_ids"]
    }
    required = multi_page_ids | failed_page_ids
    missing = required.difference(by_id)
    if missing:
        raise RuntimeError(f"required pages missing from source pool: {sorted(missing)}")

    required_docs = {str(by_id[source_id]["doc_name"]) for source_id in required}
    selected: set[str] = set(required)
    for domain, quota in QUOTAS.items():
        domain_required = [item for item in selected if str(by_id[item]["domain"]) == domain]
        if len(domain_required) > quota:
            raise RuntimeError(f"required pages exceed quota for {domain}")
        candidates = [
            sample
            for sample in samples
            if str(sample["domain"]) == domain and str(sample["sample_id"]) not in selected
        ]
        candidates.sort(
            key=lambda item: (
                str(item["doc_name"]) not in required_docs,
                str(item["cohort"]) != "unseen_holdout_50",
                -len(item["qa_ids"]),
                str(item["sample_id"]),
            )
        )
        selected.update(
            str(item["sample_id"]) for item in candidates[: quota - len(domain_required)]
        )

    selected_samples = sorted((by_id[item] for item in selected), key=lambda item: str(item["sample_id"]))
    if len(selected_samples) != 100:
        raise RuntimeError(f"selection must contain exactly 100 pages, got {len(selected_samples)}")
    actual = Counter(str(item["domain"]) for item in selected_samples)
    if dict(actual) != QUOTAS:
        raise RuntimeError(f"domain quotas changed: {dict(actual)}")
    return selected_samples, query_pages, multi_page_ids, failed_page_ids


def build_query_inventory(
    selected_samples: list[dict[str, Any]],
    query_pages: dict[str, set[str]],
    qas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = {str(item["sample_id"]) for item in selected_samples}
    inventory: list[dict[str, Any]] = []
    for qa_id, relevant in sorted(query_pages.items()):
        if not relevant.issubset(selected_ids):
            continue
        qa = qas[qa_id]
        expected_pages = qa["evidence_page_no"]
        expected_count = len(expected_pages) if isinstance(expected_pages, list) else 1
        if expected_count != len(relevant):
            continue
        inventory.append(
            {
                "qa_id": qa_id,
                "query": str(qa["questions"]),
                "domain": str(qa["doc_type"]),
                "answer_form": str(qa["answer_form"]),
                "evidence_source": str(qa["evidence_source"]),
                "relevant_source_ids": sorted(relevant),
                "query_type": "multi_evidence" if len(relevant) > 1 else "single_evidence",
            }
        )
    if len(inventory) != 280:
        raise RuntimeError(f"expected 280 complete-gold queries, got {len(inventory)}")
    if sum(item["query_type"] == "multi_evidence" for item in inventory) != 22:
        raise RuntimeError("expected 22 complete multi-evidence queries")
    return inventory


def public_sample_inventory(
    selected_samples: list[dict[str, Any]],
    multi_page_ids: set[str],
    failed_page_ids: set[str],
) -> dict[str, Any]:
    rows = []
    for sample in selected_samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "benchmark": sample["benchmark"],
                "cohort": sample["cohort"],
                "row_idx": sample["row_idx"],
                "domain": sample["domain"],
                "doc_name": sample["doc_name"],
                "page_idx": sample["page_idx"],
                "qa_count": len(sample["qa_ids"]),
                "selection_reasons": [
                    reason
                    for condition, reason in (
                        (sample["sample_id"] in multi_page_ids, "multi_evidence_coverage"),
                        (sample["sample_id"] in failed_page_ids, "prior_semantic_ocr_failure"),
                    )
                    if condition
                ]
                or ["deterministic_domain_fill"],
                "text_sha256": {variant: sha256_text(str(sample[variant])) for variant in VARIANTS},
            }
        )
    return {
        "schema_version": 1,
        "unique_pages": len(rows),
        "selection_policy": "all multi-evidence pages + all prior semantic/OCR failure pages + deterministic domain fill",
        "domain_quotas": QUOTAS,
        "samples": rows,
    }


def load_model() -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)


def encode_cached(
    model: Any,
    texts: Sequence[str],
    *,
    cache_path: Path,
    kind: str,
    batch_size: int,
) -> list[list[float]]:
    unique_texts = list(dict.fromkeys(texts))
    keys = [sha256_text(f"{kind}\0{text}") for text in unique_texts]
    cached: dict[str, np.ndarray[Any, Any]] = {}
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached = {
            str(key): vector
            for key, vector in zip(payload["keys"].tolist(), payload["vectors"], strict=True)
        }
    missing = [(key, text) for key, text in zip(keys, unique_texts, strict=True) if key not in cached]
    if missing:
        method_name = "encode_query" if kind == "query" else "encode_document"
        method = getattr(model, method_name, None) or model.encode
        raw = method(
            [text for _, text in missing],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != MODEL_DIMENSION:
            raise RuntimeError(f"unexpected embedding shape: {array.shape}")
        for (key, _), vector in zip(missing, array, strict=True):
            cached[key] = vector
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_keys = sorted(cached)
        np.savez_compressed(
            cache_path,
            keys=np.asarray(ordered_keys),
            vectors=np.stack([cached[key] for key in ordered_keys]),
        )
    by_text = {text: cached[key].astype(float).tolist() for text, key in zip(unique_texts, keys, strict=True)}
    return [by_text[text] for text in texts]


async def planned_query_texts(query_inventory: list[dict[str, Any]]) -> list[str]:
    planner = DeterministicQueryPlanner()
    values: list[str] = []
    for item in query_inventory:
        plan = await planner.plan(str(item["query"]), max_variants=3)
        values.extend(variant.query for variant in plan.variants)
    return list(dict.fromkeys(values))


def make_chunks(sample: dict[str, Any], variant: str, chunker: ParentChildChunker) -> list[Any]:
    page_number = int(sample["page_idx"]) + 1
    document = ParsedDocument(
        source_name=f"{sample['doc_name']}.pdf",
        parser="ohr-bench-official-extraction",
        blocks=[
            ParsedBlock(
                text=str(sample[variant]),
                page=page_number,
                kind="paragraph",
                locator=f"page:{page_number}",
                reading_order=0,
                metadata={"domain": sample["domain"], "sample_id": sample["sample_id"]},
            )
        ],
        provenance={
            "benchmark": "OHR-Bench",
            "doc_name": sample["doc_name"],
            "page_idx": sample["page_idx"],
            "variant": variant,
        },
    )
    return chunker.chunk(document)


async def build_variant_index(
    repo_root: Path,
    state_root: Path,
    samples: list[dict[str, Any]],
    variant: str,
    model: Any,
    batch_size: int,
    query_vectors: dict[str, list[float]],
    force_rebuild: bool,
) -> tuple[SQLiteVectorStore, SQLiteKnowledgeGraphStore, InMemoryVersionRegistry, dict[str, SourceBundle]]:
    variant_root = state_root / variant
    vector_db = variant_root / "vectors.sqlite3"
    graph_db = variant_root / "knowledge_graph.sqlite3"
    if force_rebuild:
        for path in (vector_db, graph_db):
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + "-shm").unlink(missing_ok=True)
            path.with_suffix(path.suffix + "-wal").unlink(missing_ok=True)
    provider = CachedEmbeddingProvider(query_vectors)
    store = SQLiteVectorStore(vector_db, provider)
    graph_store = SQLiteKnowledgeGraphStore(graph_db)
    registry = InMemoryVersionRegistry()
    await store.initialize()
    await graph_store.initialize()
    await registry.initialize()

    chunker = ParentChildChunker(target_tokens=500, max_tokens=650, overlap_tokens=60)
    bundles: dict[str, SourceBundle] = {}
    all_chunks: list[Any] = []
    chunk_owners: list[tuple[dict[str, Any], str, str]] = []
    for sample in samples:
        content_sha = sha256_text(str(sample[variant]))
        version_id = stable_version_id(str(sample["sample_id"]), variant, content_sha)
        chunks = make_chunks(sample, variant, chunker)
        bundles[str(sample["sample_id"])] = SourceBundle(sample, chunks, version_id, content_sha)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_owners.append((sample, version_id, content_sha))

    doc_vectors = encode_cached(
        model,
        [str(chunk.text) for chunk in all_chunks],
        cache_path=state_root / "embedding_cache" / f"{variant}_documents.npz",
        kind=f"document:{variant}",
        batch_size=batch_size,
    )
    documents: list[VectorDocument] = []
    graph_by_source: dict[str, list[GraphChunk]] = defaultdict(list)
    for chunk, vector, owner in zip(all_chunks, doc_vectors, chunk_owners, strict=True):
        sample, version_id, content_sha = owner
        source_id = str(sample["sample_id"])
        chunk_id = f"{version_id}:{chunk.chunk_id}"
        metadata = {
            **chunk.metadata,
            "benchmark": "OHR-Bench",
            "domain": sample["domain"],
            "content_sha256": content_sha,
            "pipeline_profile": "graphrag-benchmark-v1",
            "document_schema_version": 1,
            "parser": "ohr-bench-official-extraction",
            "parent_id": chunk.parent_id,
            "location": chunk.location,
            "variant": variant,
        }
        documents.append(
            VectorDocument(
                chunk_id=chunk_id,
                source_id=source_id,
                source_name=f"{sample['doc_name']}.pdf",
                text=str(chunk.text),
                embedding=vector,
                version_id=version_id,
                tenant_id=TENANT_ID,
                page=chunk.page,
                locator=chunk.locator,
                metadata=metadata,
            )
        )
        graph_by_source[source_id].append(
            GraphChunk(
                chunk_id=chunk_id,
                text=str(chunk.text),
                page=chunk.page,
                locator=chunk.locator,
                metadata=metadata,
            )
        )

    await store.upsert(documents)
    builder = KnowledgeGraphBuilder(HeuristicKnowledgeGraphExtractor(max_entities_per_chunk=12))
    for source_id, bundle in bundles.items():
        sample = bundle.sample
        graph_document = await builder.build(
            KnowledgeGraphBuildContext(
                tenant_id=TENANT_ID,
                source_id=source_id,
                source_name=f"{sample['doc_name']}.pdf",
                version_id=bundle.version_id,
                acl=(),
            ),
            graph_by_source[source_id],
        )
        await graph_store.replace_version(graph_document)
        await registry.record_building(
            source_id=source_id,
            version_id=bundle.version_id,
            source_name=f"{sample['doc_name']}.pdf",
            content_sha256=bundle.content_sha256,
            pipeline_profile="graphrag-benchmark-v1",
            parser="ohr-bench-official-extraction",
            chunks_count=len(bundle.chunks),
            warnings=[],
            tenant_id=TENANT_ID,
            acl=[],
        )
        await registry.activate(bundle.version_id)
    return store, graph_store, registry, bundles


def make_service(
    repo_root: Path,
    provider: CachedEmbeddingProvider,
    store: SQLiteVectorStore,
    graph_store: SQLiteKnowledgeGraphStore,
    registry: InMemoryVersionRegistry,
    *,
    graph_enabled: bool,
) -> RAGService:
    graph_retriever = FallbackGraphRetriever(
        KnowledgeGraphRetriever(graph_store),
        StoreBackedStructuralGraphRetriever(store),
    )
    return RAGService(
        repo_root,
        BenchmarkParser(),
        ParentChildChunker(target_tokens=500, max_tokens=650, overlap_tokens=60),
        provider,
        store,
        registry,
        max_corrective_rounds=2,
        query_max_parallel=2,
        query_timeout_seconds=30.0,
        query_min_relevance_score=0.2,
        query_rrf_k=60,
        query_planner=DeterministicQueryPlanner(),
        evidence_reflector=DeterministicEvidenceReflector(),
        graph_retriever=graph_retriever,
        graph_enabled=graph_enabled,
        graph_max_seed_hits=3,
        graph_neighbors_per_seed=2,
        graph_max_neighbors=6,
        graph_max_hops=2,
        graph_timeout_seconds=3.0,
    )


def citation_valid(hit: Any, bundles: dict[str, SourceBundle]) -> bool:
    citation = hit.citation
    bundle = bundles.get(citation.source_id)
    if bundle is None or citation.chunk_id is None or citation.quote is None:
        return False
    if not citation.chunk_id.startswith(f"{bundle.version_id}:"):
        return False
    if citation.content_sha256 != bundle.content_sha256:
        return False
    if citation.parser != "ohr-bench-official-extraction":
        return False
    quote = " ".join(citation.quote.replace("\x00", "").split())
    text = " ".join(hit.text.replace("\x00", "").split())
    return bool(quote) and quote in text


def parse_route(warnings: Sequence[str]) -> tuple[str, int]:
    for warning in warnings:
        if warning.startswith("query_route:"):
            _, route, hops = warning.split(":", 2)
            return route, int(hops)
    return "missing", -1


def parse_max_graph_hops(warnings: Sequence[str]) -> int:
    values = [int(item.rsplit(":", 1)[1]) for item in warnings if item.startswith("graph_expansion:max_hops:")]
    return max(values, default=0)


def parse_corrective_count(warnings: Sequence[str]) -> int:
    values = [int(item.rsplit(":", 1)[1]) for item in warnings if item.startswith("corrective_retrieval:count:")]
    return max(values, default=0)


async def evaluate_arm(
    service: RAGService,
    inventory: list[dict[str, Any]],
    bundles: dict[str, SourceBundle],
    *,
    arm: str,
    variant: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(inventory, start=1):
        response = await service.retrieve(str(item["query"]), TOP_K, tenant_id=TENANT_ID)
        hit_source_ids = [hit.citation.source_id for hit in response.hits]
        relevant = set(item["relevant_source_ids"])
        relevant_ranks = [rank for rank, source_id in enumerate(hit_source_ids, start=1) if source_id in relevant]
        first_rank = min(relevant_ranks, default=None)
        found = relevant.intersection(hit_source_ids)
        route, planned_hops = parse_route(response.warnings)
        records.append(
            {
                "qa_id": item["qa_id"],
                "domain": item["domain"],
                "query_type": item["query_type"],
                "answer_form": item["answer_form"],
                "evidence_source": item["evidence_source"],
                "relevant_source_ids": item["relevant_source_ids"],
                "hit_source_ids": hit_source_ids,
                "first_relevant_rank": first_rank,
                "all_evidence_hit_at_5": relevant.issubset(set(hit_source_ids)),
                "evidence_coverage_at_5": len(found) / len(relevant),
                "citation_valid": all(citation_valid(hit, bundles) for hit in response.hits),
                "latency_ms": response.latency_ms,
                "route": route,
                "planned_graph_hops": planned_hops,
                "observed_graph_hops": parse_max_graph_hops(response.warnings),
                "graph_expansion_used": any(
                    warning.startswith("graph_expansion:knowledge_graph:")
                    or warning.startswith("graph_expansion:structural_fallback:")
                    for warning in response.warnings
                ),
                "corrective_retrieval_count": parse_corrective_count(response.warnings),
                "degraded_retrieval": response.degraded_retrieval,
                "warnings": response.warnings,
            }
        )
        if index % 25 == 0 or index == len(inventory):
            print(f"[{variant}/{arm}] {index}/{len(inventory)} queries", flush=True)
    return records


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(percentile_value * len(ordered)) - 1)]


def rank_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    ranks = [row["first_relevant_rank"] for row in records]
    latencies = [float(row["latency_ms"]) for row in records]
    return {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "mrr_at_5": sum(0.0 if rank is None or rank > 5 else 1.0 / rank for rank in ranks) / count,
        "all_evidence_recall_at_5": sum(bool(row["all_evidence_hit_at_5"]) for row in records) / count,
        "mean_evidence_coverage_at_5": statistics.fmean(float(row["evidence_coverage_at_5"]) for row in records),
        "citation_validity": sum(bool(row["citation_valid"]) for row in records) / count,
        "graph_route_rate": sum(row["route"] == "graph" for row in records) / count,
        "graph_expansion_rate": sum(bool(row["graph_expansion_used"]) for row in records) / count,
        "degraded_retrieval_rate": sum(bool(row["degraded_retrieval"]) for row in records) / count,
        "bounded_hops_rate": sum(int(row["planned_graph_hops"]) <= 2 and int(row["observed_graph_hops"]) <= 2 for row in records) / count,
        "bounded_corrective_rate": sum(int(row["corrective_retrieval_count"]) <= 2 for row in records) / count,
        "median_latency_ms": statistics.median(latencies),
        "p95_single_request_latency_ms": percentile(latencies, 0.95),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_type[str(row["query_type"])].append(row)
        by_domain[str(row["domain"])].append(row)
    return {
        "overall": rank_metrics(records),
        "by_query_type": {name: rank_metrics(rows) for name, rows in sorted(by_type.items())},
        "by_domain": {name: rank_metrics(rows) for name, rows in sorted(by_domain.items())},
    }


def compare_arms(
    baseline: list[dict[str, Any]],
    graph: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row["qa_id"]): row for row in baseline}
    graph_by_id = {str(row["qa_id"]): row for row in graph}
    rows = []
    for qa_id in sorted(baseline_by_id):
        left = baseline_by_id[qa_id]
        right = graph_by_id[qa_id]
        left_hit = left["first_relevant_rank"] is not None
        right_hit = right["first_relevant_rank"] is not None
        left_found = set(left["relevant_source_ids"]).intersection(left["hit_source_ids"])
        right_found = set(right["relevant_source_ids"]).intersection(right["hit_source_ids"])
        rows.append(
            {
                "qa_id": qa_id,
                "query_type": left["query_type"],
                "domain": left["domain"],
                "baseline_hit": left_hit,
                "graphrag_hit": right_hit,
                "graphrag_added_relevant_page": bool(right_found.difference(left_found)),
                "graphrag_lost_relevant_page": bool(left_found.difference(right_found)),
                "baseline_all_evidence": left["all_evidence_hit_at_5"],
                "graphrag_all_evidence": right["all_evidence_hit_at_5"],
                "route": right["route"],
            }
        )
    multi = [row for row in rows if row["query_type"] == "multi_evidence"]
    return {
        "queries": len(rows),
        "graphrag_net_recall5_change": (
            sum(row["graphrag_hit"] for row in rows) - sum(row["baseline_hit"] for row in rows)
        ) / len(rows),
        "graph_added_relevant_page_count": sum(row["graphrag_added_relevant_page"] for row in rows),
        "graph_lost_relevant_page_count": sum(row["graphrag_lost_relevant_page"] for row in rows),
        "multi_evidence_all_hit_gain_count": sum(
            (not row["baseline_all_evidence"]) and row["graphrag_all_evidence"] for row in multi
        ),
        "multi_evidence_all_hit_regression_count": sum(
            row["baseline_all_evidence"] and (not row["graphrag_all_evidence"]) for row in multi
        ),
        "multi_evidence_graph_route_count": sum(row["route"] == "graph" for row in multi),
        "single_evidence_graph_route_count": sum(
            row["route"] == "graph" for row in rows if row["query_type"] == "single_evidence"
        ),
        "cases": rows,
    }


async def main_async(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    benchmark_root = BENCHMARK_ROOT
    results_root = benchmark_root / "results"
    manifests_root = benchmark_root / "manifests"
    state_root = repo_root / "state" / "benchmark-graphrag-100p"
    state_root.mkdir(parents=True, exist_ok=True)

    samples, qas = load_source_pool(repo_root)
    selected, query_pages, multi_page_ids, failed_page_ids = select_pages(repo_root, samples)
    query_inventory = build_query_inventory(selected, query_pages, qas)
    write_json(manifests_root / "sample_inventory.json", public_sample_inventory(selected, multi_page_ids, failed_page_ids))
    write_json(
        manifests_root / "query_inventory.json",
        {
            "schema_version": 1,
            "queries": [
                {key: value for key, value in row.items() if key != "query"}
                | {"query_sha256": sha256_text(str(row["query"]))}
                for row in query_inventory
            ],
        },
    )

    model_started = time.perf_counter()
    model = load_model()
    query_texts = await planned_query_texts(query_inventory)
    query_vector_list = encode_cached(
        model,
        query_texts,
        cache_path=state_root / "embedding_cache" / "queries.npz",
        kind="query",
        batch_size=args.batch_size,
    )
    query_vectors = dict(zip(query_texts, query_vector_list, strict=True))
    model_prepare_seconds = time.perf_counter() - model_started

    environment = {
        "run_date": "2026-08-19",
        "python": sys.version,
        "platform": platform.platform(),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": MODEL_DIMENSION,
        "embedding_device": "cpu",
        "embedding_profile": CachedEmbeddingProvider.profile_id,
        "graph_extractor": HeuristicKnowledgeGraphExtractor().profile_id,
        "query_planner": DeterministicQueryPlanner.profile_id,
        "evidence_reflector": DeterministicEvidenceReflector.profile_id,
        "online_llm_graph_path": "not_run_configured_gateway_returned_http_404_during_preflight",
        "page_budget": 100,
        "query_count": len(query_inventory),
        "multi_evidence_queries": sum(item["query_type"] == "multi_evidence" for item in query_inventory),
        "variants": list(args.variants),
        "arms": list(ARMS),
        "model_and_query_embedding_prepare_seconds": model_prepare_seconds,
        "load_or_pressure_test": False,
    }
    write_json(results_root / "environment.json", environment)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "page_budget": 100,
        "queries_per_arm_variant": len(query_inventory),
        "variants": {},
    }
    for variant in args.variants:
        index_started = time.perf_counter()
        store, graph_store, registry, bundles = await build_variant_index(
            repo_root,
            state_root,
            selected,
            variant,
            model,
            args.batch_size,
            query_vectors,
            args.force_rebuild,
        )
        provider = CachedEmbeddingProvider(query_vectors)
        baseline_service = make_service(
            repo_root, provider, store, graph_store, registry, graph_enabled=False
        )
        graph_service = make_service(
            repo_root, provider, store, graph_store, registry, graph_enabled=True
        )
        index_seconds = time.perf_counter() - index_started

        arm_records: dict[str, list[dict[str, Any]]] = {}
        for arm, service in (
            ("hybrid_baseline", baseline_service),
            ("graphrag_fallback", graph_service),
        ):
            records = await evaluate_arm(
                service,
                query_inventory,
                bundles,
                arm=arm,
                variant=variant,
            )
            arm_records[arm] = records
            write_json(results_root / f"{variant}_{arm}.json", records)

        summary["variants"][variant] = {
            "index_build_seconds": index_seconds,
            "chunks": sum(len(bundle.chunks) for bundle in bundles.values()),
            "hybrid_baseline": summarize_records(arm_records["hybrid_baseline"]),
            "graphrag_fallback": summarize_records(arm_records["graphrag_fallback"]),
            "comparison": compare_arms(
                arm_records["hybrid_baseline"], arm_records["graphrag_fallback"]
            ),
        }
        write_json(results_root / "benchmark_summary.json", summary)

    print(json.dumps({"status": "completed", "summary": str(results_root / "benchmark_summary.json")}, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_size <= 64:
        raise ValueError("batch-size must be between 1 and 64")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
