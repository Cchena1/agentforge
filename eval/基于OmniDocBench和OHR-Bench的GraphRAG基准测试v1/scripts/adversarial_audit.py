from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VARIANTS = (
    "gt_text",
    "formatting_noise_moderate",
    "semantic_noise_MinerU_moderate",
)
ARMS = ("hybrid_baseline", "graphrag_fallback")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    inventory = load(root / "manifests" / "sample_inventory.json")
    queries = load(root / "manifests" / "query_inventory.json")
    summary = load(root / "results" / "benchmark_summary.json")

    integrity_checks: dict[str, bool] = {
        "exactly_100_unique_pages": inventory["unique_pages"] == 100
        and len(inventory["samples"]) == 100
        and len({sample["sample_id"] for sample in inventory["samples"]}) == 100,
        "exactly_280_queries": len(queries["queries"]) == 280,
        "exactly_22_multi_evidence_queries": sum(
            row["query_type"] == "multi_evidence" for row in queries["queries"]
        )
        == 22,
        "manifest_contains_no_raw_text": all(
            not any(key in sample for key in ("gt_text", "formatting_noise_moderate", "semantic_noise_MinerU_moderate"))
            for sample in inventory["samples"]
        ),
        "query_manifest_contains_no_raw_query_or_answer": all(
            "query" not in row and "answers" not in row and "evidence_context" not in row
            for row in queries["queries"]
        ),
    }

    quality_checks: dict[str, bool] = {}
    observations: dict[str, Any] = {}
    for variant in VARIANTS:
        baseline = load(root / "results" / f"{variant}_hybrid_baseline.json")
        graph = load(root / "results" / f"{variant}_graphrag_fallback.json")
        integrity_checks[f"{variant}:record_counts"] = len(baseline) == len(graph) == 280
        integrity_checks[f"{variant}:query_identity"] = {
            row["qa_id"] for row in baseline
        } == {row["qa_id"] for row in graph}
        integrity_checks[f"{variant}:citations_valid"] = all(
            row["citation_valid"] for row in (*baseline, *graph)
        )
        integrity_checks[f"{variant}:bounded_graph_hops"] = all(
            row["planned_graph_hops"] <= 2 and row["observed_graph_hops"] <= 2
            for row in graph
        )
        integrity_checks[f"{variant}:bounded_corrective_retrieval"] = all(
            row["corrective_retrieval_count"] <= 2 for row in (*baseline, *graph)
        )

        variant_summary = summary["variants"][variant]
        baseline_metrics = variant_summary["hybrid_baseline"]["overall"]
        graph_metrics = variant_summary["graphrag_fallback"]["overall"]
        baseline_multi = variant_summary["hybrid_baseline"]["by_query_type"]["multi_evidence"]
        graph_multi = variant_summary["graphrag_fallback"]["by_query_type"]["multi_evidence"]
        quality_checks[f"{variant}:overall_recall5_regression_le_2pp"] = (
            graph_metrics["recall_at_5"] + 0.02 >= baseline_metrics["recall_at_5"]
        )
        quality_checks[f"{variant}:multi_evidence_all_recall_not_worse"] = (
            graph_multi["all_evidence_recall_at_5"] >= baseline_multi["all_evidence_recall_at_5"]
        )
        quality_checks[f"{variant}:graph_degraded_rate_le_1pct"] = (
            graph_metrics["degraded_retrieval_rate"] <= 0.01
        )
        quality_checks[f"{variant}:multi_evidence_graph_route_rate_ge_80pct"] = (
            graph_multi["graph_route_rate"] >= 0.80
        )
        observations[variant] = {
            "baseline_recall_at_5": baseline_metrics["recall_at_5"],
            "graphrag_recall_at_5": graph_metrics["recall_at_5"],
            "baseline_multi_all_evidence_recall_at_5": baseline_multi["all_evidence_recall_at_5"],
            "graphrag_multi_all_evidence_recall_at_5": graph_multi["all_evidence_recall_at_5"],
            "graph_degraded_retrieval_rate": graph_metrics["degraded_retrieval_rate"],
            "multi_evidence_graph_route_rate": graph_multi["graph_route_rate"],
            "comparison": {
                key: value
                for key, value in variant_summary["comparison"].items()
                if key != "cases"
            },
        }

    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    integrity_checks["artifact_size_below_1gb"] = total_bytes < 1_000_000_000
    integrity_checks["no_large_raw_formats"] = not any(
        path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".sqlite", ".sqlite3", ".npz"}
        for path in root.rglob("*")
        if path.is_file()
    )

    result = {
        "schema_version": 1,
        "integrity_status": "pass" if all(integrity_checks.values()) else "fail",
        "quality_gate_status": "pass" if all(quality_checks.values()) else "fail",
        "integrity_checks": integrity_checks,
        "quality_checks": quality_checks,
        "observations": observations,
        "artifact_bytes": total_bytes,
        "note": "Quality-gate failure is a benchmark finding, not evidence corruption.",
    }
    target = root / "results" / "adversarial_gate.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if all(integrity_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
