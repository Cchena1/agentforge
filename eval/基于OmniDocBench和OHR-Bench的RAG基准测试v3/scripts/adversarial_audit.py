from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

VARIANTS = ["gt_text", "formatting_noise_moderate", "semantic_noise_MinerU_moderate"]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    ranks = [row.get("first_relevant_rank") for row in records]
    latencies = sorted(float(row["latency_ms"]) for row in records)
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        domains[str(row["domain"])].append(row)

    def rank_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        local = [row.get("first_relevant_rank") for row in rows]
        count = len(rows)
        return {
            "queries": count,
            "recall_at_1": sum(rank == 1 for rank in local) / count,
            "recall_at_5": sum(rank is not None and rank <= 5 for rank in local) / count,
            "mrr_at_5": sum(0.0 if rank is None or rank > 5 else 1.0 / rank for rank in local) / count,
        }

    return {
        **rank_metrics(records),
        "citation_validity": sum(bool(row["citation_valid"]) for row in records) / n,
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": latencies[max(0, math.ceil(0.95 * n) - 1)],
        "by_domain": {name: rank_metrics(rows) for name, rows in sorted(domains.items())},
    }


def main() -> int:
    benchmark_dir = Path(__file__).resolve().parents[1]
    summary = load(benchmark_dir / "results" / "ohr_retrieval_summary.json")
    inventory = load(benchmark_dir / "manifests" / "sample_inventory.json")
    assert inventory["unique_pages"] == 200
    assert len(inventory["samples"]) == 200
    assert all("gt_text" not in sample for sample in inventory["samples"])
    assert len({(sample["row_idx"], sample["doc_name"], sample["page_idx"]) for sample in inventory["samples"]}) == 200
    for expected, variant in zip(summary, VARIANTS, strict=True):
        records = load(benchmark_dir / "results" / f"ohr_retrieval_{variant}.json")
        assert len(records) == 559
        actual = metrics(records)
        for key in ("recall_at_1", "recall_at_5", "mrr_at_5", "citation_validity"):
            assert abs(float(expected[key]) - float(actual[key])) < 1e-12, (variant, key)
        assert all(row["citation_valid"] for row in records)
    total = sum(path.stat().st_size for path in benchmark_dir.rglob("*") if path.is_file())
    assert total < 1_000_000_000
    print(json.dumps({"status": "pass", "unique_pages": 200, "queries_per_variant": 559, "bytes": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
