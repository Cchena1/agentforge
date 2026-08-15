from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    ".gitignore": "defense-in-depth ignore rules",
    "README.md": "benchmark entry point",
    "DATA_SOURCES.md": "official sources and reproducibility boundary",
    "EVIDENCE.md": "evidence policy and adversarial review",
    "report.md": "formal benchmark report",
    "manifests/sample_inventory.json": "sanitized 200-page sample inventory",
    "results/environment.json": "runtime environment evidence",
    "results/sample_summary.json": "sample allocation and source hashes",
    "results/omni_parser_summary.json": "parser aggregate evidence",
    "results/omni_failed_10_smoke_2026-08-15.json": ("ten-page failed-PDF smoke evidence"),
    "results/omni_failed_10_smoke_minimal_fix_2026-08-15.json": (
        "ten-page minimal-fix regression evidence"
    ),
    "results/omni_parser_failures_sanitized.json": ("sanitized per-page parser failures"),
    "results/omni_unavailable.json": "parser unavailability marker",
    "results/ohr_retrieval_summary.json": "retrieval aggregate evidence",
    "results/ohr_retrieval_gt_text.json": ("ground-truth per-query ranking evidence"),
    "results/ohr_retrieval_formatting_noise_moderate.json": (
        "formatting-noise per-query ranking evidence"
    ),
    "results/ohr_retrieval_semantic_noise_MinerU_moderate.json": (
        "semantic-noise per-query ranking evidence"
    ),
    "scripts/adversarial_audit.py": "adversarial evidence audit",
    "scripts/generate_evidence_manifest.py": "evidence hash manifest generator",
    "scripts/prepare_samples.py": "deterministic sample preparation",
    "scripts/run_benchmark.py": "benchmark runner",
    "scripts/run_failed_pdf_smoke.py": "failed-PDF smoke regression runner",
}

files = []
for path in sorted(
    item for item in ROOT.rglob("*") if item.is_file() and item.name != "evidence_manifest.json"
):
    relative = path.relative_to(ROOT).as_posix()
    data = path.read_bytes()
    files.append(
        {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "role": ROLES.get(relative, "supporting evidence"),
        }
    )
payload = {
    "schema_version": 1,
    "generated_on": "2026-08-15",
    "hash_algorithm": "sha256",
    "manifest_self_excluded": True,
    "file_count": len(files),
    "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
    "files": files,
}
(ROOT / "evidence_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "file_count": payload["file_count"],
            "total_bytes": payload["total_bytes_excluding_manifest"],
            "largest": max(files, key=lambda item: item["bytes"]),
        },
        ensure_ascii=False,
        indent=2,
    )
)
