from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_service.document_models import ParseRequest
from agent_service.document_pipeline import (
    DoclingParserBackend,
    ParentChildChunker,
    ParseQualityEvaluator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = PROJECT_ROOT / "state" / "benchmark-200p"
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def normalize(text: str) -> str:
    return "".join(TOKEN_RE.findall(text.lower()))


def ngrams(text: str, n: int = 3) -> Counter[str]:
    value = normalize(text)
    if len(value) < n:
        return Counter([value]) if value else Counter()
    return Counter(value[index : index + n] for index in range(len(value) - n + 1))


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(
    manifest: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    failed_ids = {item["sample_id"] for item in failures}
    selected: dict[str, dict[str, Any]] = {}
    for item in sorted(manifest, key=lambda value: value["sample_id"]):
        if item["sample_id"] in failed_ids:
            selected.setdefault(item["stratum"], item)
    return [selected[key] for key in sorted(selected)]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(output: Path) -> None:
    if not sys.flags.utf8_mode:
        raise RuntimeError("Windows runs must use Python -X utf8")
    if os.getenv("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "").lower() not in {
        "0",
        "false",
        "no",
    }:
        raise RuntimeError(
            "Set DOCLING_INFERENCE_COMPILE_TORCH_MODELS=false to avoid requiring MSVC cl.exe"
        )

    manifest = json.loads((STATE_ROOT / "omni_manifest.json").read_text(encoding="utf-8"))
    failures = json.loads(
        (BENCHMARK_ROOT / "results" / "omni_parser_failures_sanitized.json").read_text(
            encoding="utf-8"
        )
    )
    previous = {item["sample_id"]: item for item in failures}
    selected = select_samples(manifest, failures)
    if len(selected) != 10:
        raise RuntimeError(
            f"Expected one failed sample from each of 10 strata, got {len(selected)}"
        )

    backend = DoclingParserBackend()
    evaluator = ParseQualityEvaluator()
    chunker = ParentChildChunker(target_tokens=500, max_tokens=650, overlap_tokens=60)
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "title": "Ten-page failed PDF parser smoke test",
        "selection_rule": (
            "First previously failed OmniDocBench sample by sample_id in each of 10 strata"
        ),
        "environment_contract": {
            "python_utf8_mode": bool(sys.flags.utf8_mode),
            "docling_compile_torch_models": os.environ["DOCLING_INFERENCE_COMPILE_TORCH_MODELS"],
            "shared_backend_instance": True,
            "parallelism": 1,
            "pressure_test": False,
        },
        "parser_profile": backend.profile_id,
        "selected_pages": len(selected),
        "results": results,
        "summary": None,
    }
    write_json(output, payload)

    for index, item in enumerate(selected, 1):
        pdf_path = STATE_ROOT / item["pdf_path"]
        started = time.perf_counter()
        try:
            document = await backend.parse(ParseRequest(pdf_path, "application/pdf"))
            duration_ms = (time.perf_counter() - started) * 1000
            report = evaluator.evaluate(document)
            chunks = chunker.chunk(document)
            extracted = "\n\n".join(block.text for block in document.blocks if block.text.strip())
            result = {
                "sample_id": item["sample_id"],
                "stratum": item["stratum"],
                "previous_status": previous[item["sample_id"]]["status"],
                "previous_failure_fingerprint": previous[item["sample_id"]]["failure_fingerprint"],
                "status": "parsed",
                "quality_accepted": report.accepted,
                "quality_score": report.score,
                "quality_issues": report.issues,
                "duration_ms": duration_ms,
                "parser": document.parser,
                "block_count": len(document.blocks),
                "asset_count": len(document.assets),
                "chunk_count": len(chunks),
                "extracted_chars": len(extracted),
                "ground_truth_chars": item["ground_truth_chars"],
                "char_ratio": len(normalize(extracted))
                / max(1, len(normalize(item["ground_truth_text"]))),
                "char_trigram_f1": multiset_f1(
                    ngrams(extracted), ngrams(item["ground_truth_text"])
                ),
                "pdf_sha256_matches_manifest": sha256_file(pdf_path) == item["pdf_sha256"],
            }
        except Exception as exc:  # noqa: BLE001 - preserve third-party parser failures
            message = str(exc).lower()
            if "gbk" in message:
                fingerprint = "docling_layout_model_gbk_decode_error"
            elif "cl is not found" in message:
                fingerprint = "torch_compile_requires_msvc"
            elif "locate the files on the hub" in message:
                fingerprint = "docling_model_artifact_missing"
            else:
                fingerprint = f"{type(exc).__name__}:unclassified"
            result = {
                "sample_id": item["sample_id"],
                "stratum": item["stratum"],
                "previous_status": previous[item["sample_id"]]["status"],
                "previous_failure_fingerprint": previous[item["sample_id"]]["failure_fingerprint"],
                "status": "failed",
                "duration_ms": (time.perf_counter() - started) * 1000,
                "error_type": type(exc).__name__,
                "failure_fingerprint": fingerprint,
                "pdf_sha256_matches_manifest": sha256_file(pdf_path) == item["pdf_sha256"],
            }
        results.append(result)
        payload["results"] = results
        write_json(output, payload)
        print(
            f"{index:02d}/10 {item['sample_id']} {item['stratum']} {result['status']} "
            f"accepted={result.get('quality_accepted')} "
            f"duration={result['duration_ms'] / 1000:.1f}s",
            flush=True,
        )

    parsed = [item for item in results if item["status"] == "parsed"]
    accepted = [item for item in parsed if item.get("quality_accepted")]
    durations = [float(item["duration_ms"]) for item in results]
    content_guard_rejections = [
        item
        for item in parsed
        if not item.get("quality_accepted")
        and any(
            issue
            in {
                "LOW_TEXT_COVERAGE_WITH_VISUAL_ASSETS",
                "SPARSE_TABLE_EXTRACTION",
            }
            for issue in item.get("quality_issues", [])
        )
    ]
    payload["summary"] = {
        "pages": len(results),
        "previously_failed": sum(item["previous_status"] == "failed" for item in results),
        "parsed": len(parsed),
        "quality_accepted": len(accepted),
        "parse_success_rate": len(parsed) / max(1, len(results)),
        "quality_acceptance_rate": len(accepted) / max(1, len(results)),
        "content_guard_rejections": len(content_guard_rejections),
        "recovery_rate": len(accepted) / max(1, len(results)),
        "mean_quality_score": statistics.fmean(item["quality_score"] for item in parsed)
        if parsed
        else 0.0,
        "mean_char_trigram_f1": statistics.fmean(item["char_trigram_f1"] for item in parsed)
        if parsed
        else 0.0,
        "median_duration_ms": statistics.median(durations) if durations else 0.0,
        "p95_duration_ms": percentile(durations, 0.95),
        "all_pdf_hashes_match": all(item["pdf_sha256_matches_manifest"] for item in results),
    }
    write_json(output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=STATE_ROOT / "omni_failed_10_smoke_latest.json",
    )
    args = parser.parse_args()
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
