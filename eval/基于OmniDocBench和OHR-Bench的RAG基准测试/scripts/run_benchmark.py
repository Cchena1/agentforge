from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil

from agent_service.document_models import ParsedBlock, ParsedDocument, ParseRequest
from agent_service.document_pipeline import (
    DoclingParserBackend,
    ParentChildChunker,
    ParseQualityEvaluator,
)
from agent_service.embeddings import HashEmbedding
from agent_service.vector_store import SQLiteVectorStore, VectorDocument

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT / "state" / "benchmark-200p"
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(text: str) -> str:
    return "".join(TOKEN_RE.findall(text.lower()))


def ngrams(text: str, n: int = 3) -> Counter[str]:
    value = normalize(text)
    if len(value) < n:
        return Counter([value]) if value else Counter()
    return Counter(value[i : i + n] for i in range(len(value) - n + 1))


def multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


async def run_parser() -> None:
    manifest = json.loads((ROOT / "omni_manifest.json").read_text(encoding="utf-8"))
    backend = DoclingParserBackend()
    evaluator = ParseQualityEvaluator()
    chunker = ParentChildChunker(target_tokens=500, max_tokens=650, overlap_tokens=60)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(manifest, 1):
        pdf_path = ROOT / item["pdf_path"]
        request = ParseRequest(pdf_path, "application/pdf")
        started = time.perf_counter()
        try:
            document = await asyncio.wait_for(backend.parse(request), timeout=180)
            duration_ms = (time.perf_counter() - started) * 1000
            report = evaluator.evaluate(document)
            chunks = chunker.chunk(document)
            extracted = "\n\n".join(block.text for block in document.blocks if block.text.strip())
            kinds = Counter(block.kind for block in document.blocks)
            result = {
                "sample_id": item["sample_id"],
                "stratum": item["stratum"],
                "status": "parsed",
                "duration_ms": duration_ms,
                "parser": document.parser,
                "quality_accepted": report.accepted,
                "quality_score": report.score,
                "quality_issues": report.issues,
                "warnings": document.warnings,
                "block_count": len(document.blocks),
                "block_kinds": kinds,
                "asset_count": len(document.assets),
                "chunk_count": len(chunks),
                "extracted_chars": len(extracted),
                "ground_truth_chars": item["ground_truth_chars"],
                "char_ratio": len(normalize(extracted))
                / max(1, len(normalize(item["ground_truth_text"]))),
                "char_trigram_f1": multiset_f1(
                    ngrams(extracted), ngrams(item["ground_truth_text"])
                ),
            }
        # Third-party parser backends expose heterogeneous exception types; record all failures.
        except Exception as exc:  # noqa: BLE001
            result = {
                "sample_id": item["sample_id"],
                "stratum": item["stratum"],
                "status": "failed",
                "duration_ms": (time.perf_counter() - started) * 1000,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        write_json(ROOT / "omni_parser_results.json", results)
        print(
            f"Parser {index:02d}/50 {item['sample_id']} {result['status']} {result['duration_ms'] / 1000:.1f}s",
            flush=True,
        )

    parsed = [item for item in results if item["status"] == "parsed"]
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({item["stratum"] for item in results}):
        subset = [item for item in results if item["stratum"] == stratum]
        successful = [item for item in subset if item["status"] == "parsed"]
        by_stratum[stratum] = {
            "pages": len(subset),
            "parsed": len(successful),
            "quality_accepted": sum(bool(item.get("quality_accepted")) for item in successful),
            "mean_trigram_f1": statistics.fmean(item["char_trigram_f1"] for item in successful)
            if successful
            else 0.0,
            "mean_quality_score": statistics.fmean(item["quality_score"] for item in successful)
            if successful
            else 0.0,
        }
    summary = {
        "pages": len(results),
        "parsed": len(parsed),
        "failed": len(results) - len(parsed),
        "quality_accepted": sum(bool(item["quality_accepted"]) for item in parsed),
        "quality_acceptance_rate": sum(bool(item["quality_accepted"]) for item in parsed)
        / max(1, len(results)),
        "mean_quality_score": statistics.fmean(item["quality_score"] for item in parsed)
        if parsed
        else 0.0,
        "mean_char_trigram_f1": statistics.fmean(item["char_trigram_f1"] for item in parsed)
        if parsed
        else 0.0,
        "median_char_trigram_f1": statistics.median(item["char_trigram_f1"] for item in parsed)
        if parsed
        else 0.0,
        "median_duration_ms": statistics.median(item["duration_ms"] for item in results),
        "p95_duration_ms": percentile([item["duration_ms"] for item in results], 0.95),
        "by_stratum": by_stratum,
    }
    write_json(ROOT / "omni_parser_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def qa_page_map(
    qas: list[dict[str, Any]], selected_pairs: dict[tuple[str, int], str]
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    query_by_id: dict[str, dict[str, Any]] = {}
    relevant: dict[str, set[str]] = defaultdict(set)
    for qa in qas:
        raw_pages = qa.get("evidence_page_no", [])
        pages = raw_pages if isinstance(raw_pages, list) else [raw_pages]
        for raw_page in pages:
            pair = (qa["doc_name"], int(raw_page))
            if pair in selected_pairs:
                query_by_id.setdefault(qa["ID"], qa)
                relevant[qa["ID"]].add(selected_pairs[pair])
    return list(query_by_id.values()), relevant


async def run_retrieval_variant(
    variant: str, manifest: list[dict[str, Any]], qas: list[dict[str, Any]]
) -> dict[str, Any]:
    db_path = ROOT / f"ohr_{variant}.sqlite3"
    if db_path.exists():
        db_path.unlink()
    embedding = HashEmbedding(384)
    store = SQLiteVectorStore(db_path, embedding)
    chunker = ParentChildChunker(target_tokens=500, max_tokens=650, overlap_tokens=60)
    documents: list[VectorDocument] = []
    selected_pairs: dict[tuple[str, int], str] = {}
    source_texts: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for item in manifest:
        source_id = item["sample_id"]
        selected_pairs[(item["doc_name"], int(item["page_idx"]))] = source_id
        text = item[variant] or ""
        source_texts[source_id] = text
        source_hashes[source_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed = ParsedDocument(
            source_name=f"{item['doc_name']}#page={item['page_idx']}",
            parser=f"ohr-{variant}",
            blocks=[
                ParsedBlock(
                    text=text,
                    page=int(item["page_idx"]) + 1,
                    kind="paragraph",
                    locator=f"page:{int(item['page_idx']) + 1}:block:1",
                )
            ],
            provenance={"page_count": 1, "empty_pages": 0},
        )
        for chunk in chunker.chunk(parsed):
            vector = (await embedding.embed([chunk.text]))[0]
            documents.append(
                VectorDocument(
                    chunk_id=f"{source_id}:{chunk.chunk_id}",
                    source_id=source_id,
                    source_name=parsed.source_name,
                    text=chunk.text,
                    embedding=vector,
                    version_id=variant,
                    page=int(item["page_idx"]) + 1,
                    locator=f"page:{int(item['page_idx']) + 1}:block:1",
                    metadata={
                        "content_sha256": source_hashes[source_id],
                        "parser": f"ohr-{variant}",
                        "document_schema_version": 1,
                        "reading_order": 1,
                        "parent_id": chunk.parent_id,
                    },
                )
            )
    await store.upsert(documents)
    queries, relevant = qa_page_map(qas, selected_pairs)
    records: list[dict[str, Any]] = []
    for index, qa in enumerate(sorted(queries, key=lambda x: x["ID"]), 1):
        hits, latency_ms = await store.search(qa["questions"], top_k=5)
        hit_sources = [hit.citation.source_id for hit in hits]
        targets = relevant[qa["ID"]]
        ranks = [rank for rank, source_id in enumerate(hit_sources, 1) if source_id in targets]
        first_rank = min(ranks) if ranks else None
        citation_valid = all(
            hit.citation.quote in hit.text
            and hit.citation.source_id in source_texts
            and hit.text in source_texts[hit.citation.source_id]
            and hit.citation.content_sha256 == source_hashes[hit.citation.source_id]
            for hit in hits
        )
        records.append(
            {
                "qa_id": qa["ID"],
                "domain": qa["doc_type"],
                "evidence_source": qa["evidence_source"],
                "answer_form": qa["answer_form"],
                "relevant_source_ids": sorted(targets),
                "hit_source_ids": hit_sources,
                "first_relevant_rank": first_rank,
                "latency_ms": latency_ms,
                "citation_valid": citation_valid,
            }
        )
        if index % 25 == 0 or index == len(queries):
            print(f"Retrieval {variant} {index}/{len(queries)}", flush=True)
    write_json(ROOT / f"ohr_retrieval_{variant}.json", records)
    by_domain: dict[str, Any] = {}
    for domain in sorted({record["domain"] for record in records}):
        subset = [record for record in records if record["domain"] == domain]
        by_domain[domain] = {
            "queries": len(subset),
            "recall_at_1": sum(record["first_relevant_rank"] == 1 for record in subset)
            / len(subset),
            "recall_at_5": sum(record["first_relevant_rank"] is not None for record in subset)
            / len(subset),
            "mrr_at_5": statistics.fmean(
                1 / record["first_relevant_rank"] if record["first_relevant_rank"] else 0
                for record in subset
            ),
        }
    return {
        "variant": variant,
        "pages": len(manifest),
        "chunks": len(documents),
        "queries": len(records),
        "recall_at_1": sum(record["first_relevant_rank"] == 1 for record in records)
        / max(1, len(records)),
        "recall_at_5": sum(record["first_relevant_rank"] is not None for record in records)
        / max(1, len(records)),
        "mrr_at_5": statistics.fmean(
            1 / record["first_relevant_rank"] if record["first_relevant_rank"] else 0
            for record in records
        )
        if records
        else 0.0,
        "citation_validity": sum(record["citation_valid"] for record in records)
        / max(1, len(records)),
        "median_latency_ms": statistics.median(record["latency_ms"] for record in records)
        if records
        else 0.0,
        "p95_latency_ms": percentile([record["latency_ms"] for record in records], 0.95),
        "by_domain": by_domain,
    }


async def run_retrieval() -> None:
    manifest = json.loads((ROOT / "ohr_manifest.json").read_text(encoding="utf-8"))
    qas = json.loads((ROOT / "ohr_qas_v2.json").read_text(encoding="utf-8"))
    summaries = []
    for variant in ["gt_text", "formatting_noise_moderate", "semantic_noise_MinerU_moderate"]:
        summary = await run_retrieval_variant(variant, manifest, qas)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    write_json(ROOT / "ohr_retrieval_summary.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["parser", "retrieval", "environment"])
    args = parser.parse_args()
    if args.phase == "parser":
        asyncio.run(run_parser())
    elif args.phase == "retrieval":
        asyncio.run(run_retrieval())
    else:
        import docling
        import torch

        environment = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "memory_gib": round(psutil.virtual_memory().total / 1024**3, 2),
            "docling": docling.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "embedding": HashEmbedding(384).profile_id,
        }
        write_json(ROOT / "environment.json", environment)
        print(json.dumps(environment, ensure_ascii=False, indent=2))
