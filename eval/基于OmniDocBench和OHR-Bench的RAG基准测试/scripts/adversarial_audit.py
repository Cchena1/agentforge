from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
ONE_GIB = 1024**3
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pt",
    ".bin",
}
ALLOWED_QUERY_KEYS = {
    "qa_id",
    "domain",
    "evidence_source",
    "answer_form",
    "relevant_source_ids",
    "hit_source_ids",
    "first_relevant_rank",
    "latency_ms",
    "citation_valid",
}
RAW_TEXT_KEYS = {
    "gt_text",
    "text",
    "question",
    "answer",
    "formatting_noise_moderate",
    "semantic_noise_mineru_moderate",
}
SMOKE_RESULT_KEYS = {
    "sample_id",
    "stratum",
    "previous_status",
    "previous_failure_fingerprint",
    "status",
    "quality_accepted",
    "quality_score",
    "quality_issues",
    "duration_ms",
    "parser",
    "block_count",
    "asset_count",
    "chunk_count",
    "extracted_chars",
    "ground_truth_chars",
    "char_ratio",
    "char_trigram_f1",
    "pdf_sha256_matches_manifest",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


errors: list[str] = []
notes: list[str] = []
files = [path for path in ROOT.rglob("*") if path.is_file()]
total_bytes = sum(path.stat().st_size for path in files)
if total_bytes >= ONE_GIB:
    errors.append(f"benchmark folder is {total_bytes} bytes, exceeding 1 GiB")

for path in files:
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden binary/dataset artifact: {path}")
    if path.suffix == ".json":
        try:
            load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {path}: {exc}")
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"invalid Python {path}: {exc}")

inventory = load_json(ROOT / "manifests" / "sample_inventory.json")
samples = inventory["samples"]
if inventory["unique_pages"] != 200 or len(samples) != 200:
    errors.append("sample inventory does not contain exactly 200 pages")
ids = [sample["sample_id"] for sample in samples]
if len(ids) != len(set(ids)):
    errors.append("duplicate sample_id in inventory")
counts = Counter(sample["benchmark"] for sample in samples)
if counts != Counter({"OmniDocBench": 50, "OHR-Bench": 150}):
    errors.append(f"unexpected benchmark allocation: {dict(counts)}")
for index, sample in enumerate(samples):
    overlap = {key.lower() for key in sample} & RAW_TEXT_KEYS
    if overlap:
        errors.append(f"raw text field in inventory sample {index}: {sorted(overlap)}")

summary = load_json(ROOT / "results" / "ohr_retrieval_summary.json")
summary_by_variant = {item["variant"]: item for item in summary}
query_files = sorted((ROOT / "results").glob("ohr_retrieval_*.json"))
query_files = [path for path in query_files if path.name != "ohr_retrieval_summary.json"]
if len(query_files) != 3:
    errors.append(f"expected 3 detailed retrieval files, found {len(query_files)}")
for path in query_files:
    rows = load_json(path)
    if len(rows) != 427:
        errors.append(f"{path.name}: expected 427 rows, found {len(rows)}")
    for index, row in enumerate(rows):
        keys = set(row)
        if keys != ALLOWED_QUERY_KEYS:
            errors.append(
                f"{path.name}[{index}] unexpected keys: {sorted(keys ^ ALLOWED_QUERY_KEYS)}"
            )
            break
        for value in row.values():
            if isinstance(value, str) and len(value) > 512:
                errors.append(f"{path.name}[{index}] contains oversized text value")
                break
    ranks = [row["first_relevant_rank"] for row in rows]
    calc = {
        "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr_at_5": sum((1 / rank) if rank is not None and rank <= 5 else 0 for rank in ranks)
        / len(ranks),
        "citation_validity": sum(bool(row["citation_valid"]) for row in rows) / len(rows),
    }
    variant = path.stem.removeprefix("ohr_retrieval_")
    expected = summary_by_variant.get(variant)
    if expected is None:
        errors.append(f"missing summary variant for {variant}")
    else:
        for metric, actual in calc.items():
            if not math.isclose(actual, expected[metric], rel_tol=0, abs_tol=1e-12):
                errors.append(f"{variant} {metric}: detail={actual}, summary={expected[metric]}")

failures = load_json(ROOT / "results" / "omni_parser_failures_sanitized.json")
if len(failures) != 50:
    errors.append(f"expected 50 sanitized parser failures, found {len(failures)}")
serialized_failures = json.dumps(failures, ensure_ascii=False)
if re.search(r"(?:[A-Za-z]:\\|/Users/|\\Users\\)", serialized_failures):
    errors.append("parser failure evidence contains an absolute/user path")


def validate_smoke_evidence(
    filename: str, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(ROOT / "results" / filename)
    rows = payload.get("results", [])
    summary = payload.get("summary", {})
    if len(rows) != 10 or payload.get("selected_pages") != 10:
        errors.append(f"{label} evidence does not contain exactly 10 pages")
    if len({row.get("sample_id") for row in rows}) != 10:
        errors.append(f"{label} evidence has duplicate sample IDs")
    if len({row.get("stratum") for row in rows}) != 10:
        errors.append(f"{label} evidence does not cover 10 unique strata")
    for index, row in enumerate(rows):
        missing_or_extra = set(row) ^ SMOKE_RESULT_KEYS
        if missing_or_extra:
            errors.append(
                f"{label} result {index} has unexpected schema fields: {sorted(missing_or_extra)}"
            )
        if row.get("previous_status") != "failed":
            errors.append(f"{label} result {index} was not selected from prior failures")
        if row.get("status") != "parsed":
            errors.append(f"{label} result {index} did not parse successfully")
        if not row.get("pdf_sha256_matches_manifest"):
            errors.append(f"{label} result {index} PDF hash does not match the manifest")
        if not 0 <= float(row.get("char_trigram_f1", -1)) <= 1:
            errors.append(f"{label} result {index} has invalid character trigram F1")

    parsed_rows = [row for row in rows if row.get("status") == "parsed"]
    accepted_rows = [row for row in parsed_rows if row.get("quality_accepted")]
    durations = [float(row["duration_ms"]) for row in rows]
    calculated = {
        "pages": len(rows),
        "previously_failed": sum(row.get("previous_status") == "failed" for row in rows),
        "parsed": len(parsed_rows),
        "quality_accepted": len(accepted_rows),
        "recovery_rate": len(accepted_rows) / max(1, len(rows)),
        "mean_quality_score": (
            statistics.fmean(float(row["quality_score"]) for row in parsed_rows)
            if parsed_rows
            else 0.0
        ),
        "mean_char_trigram_f1": (
            statistics.fmean(float(row["char_trigram_f1"]) for row in parsed_rows)
            if parsed_rows
            else 0.0
        ),
        "median_duration_ms": statistics.median(durations) if durations else 0.0,
        "p95_duration_ms": percentile(durations, 0.95),
        "all_pdf_hashes_match": all(bool(row.get("pdf_sha256_matches_manifest")) for row in rows),
    }
    for key, actual in calculated.items():
        expected = summary.get(key)
        if isinstance(actual, float):
            if expected is None or not math.isclose(
                actual, float(expected), rel_tol=0, abs_tol=1e-9
            ):
                errors.append(f"{label} summary {key}: detail={actual}, summary={expected}")
        elif actual != expected:
            errors.append(f"{label} summary {key}: detail={actual}, summary={expected}")
    return rows, summary


smoke_rows, smoke_summary = validate_smoke_evidence(
    "omni_failed_10_smoke_2026-08-15.json", "baseline smoke"
)
fixed_rows, fixed_summary = validate_smoke_evidence(
    "omni_failed_10_smoke_minimal_fix_2026-08-15.json", "minimal-fix smoke"
)

baseline_by_id = {row["sample_id"]: row for row in smoke_rows}
fixed_by_id = {row["sample_id"]: row for row in fixed_rows}
if set(baseline_by_id) != set(fixed_by_id):
    errors.append("minimal-fix smoke does not use the same ten samples as the baseline")
for sample_id, baseline_row in baseline_by_id.items():
    fixed_row = fixed_by_id.get(sample_id)
    if fixed_row is None:
        continue
    if fixed_row.get("stratum") != baseline_row.get("stratum"):
        errors.append(f"minimal-fix smoke changed stratum for {sample_id}")
    if not math.isclose(
        float(fixed_row["char_trigram_f1"]),
        float(baseline_row["char_trigram_f1"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        errors.append(f"minimal-fix smoke changed extracted-content F1 for {sample_id}")

expected_rejections = {"omni-031", "omni-016", "omni-021", "omni-036"}
actual_rejections = {row["sample_id"] for row in fixed_rows if not row.get("quality_accepted")}
if actual_rejections != expected_rejections:
    errors.append(
        "minimal-fix smoke rejection set differs from the reviewed four-page set: "
        f"{sorted(actual_rejections)}"
    )
for row in fixed_rows:
    if row["sample_id"] not in expected_rejections:
        continue
    issues = set(row.get("quality_issues", []))
    if "OCR_REQUIRED" not in issues:
        errors.append(f"minimal-fix smoke did not route {row['sample_id']} to OCR")
    if not issues.intersection({"LOW_TEXT_COVERAGE_WITH_VISUAL_ASSETS", "SPARSE_TABLE_EXTRACTION"}):
        errors.append(f"minimal-fix smoke lacks a content guard issue for {row['sample_id']}")

expected_fixed_summary = {
    "parse_success_rate": 1.0,
    "quality_acceptance_rate": 0.6,
    "content_guard_rejections": 4,
}
for key, expected in expected_fixed_summary.items():
    actual = fixed_summary.get(key)
    if isinstance(expected, float):
        if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
            errors.append(f"minimal-fix summary {key}: expected={expected}, actual={actual}")
    elif actual != expected:
        errors.append(f"minimal-fix summary {key}: expected={expected}, actual={actual}")

scan_paths = [*files, PROJECT_ROOT / "README.md", PROJECT_ROOT / "report.md"]
secret_patterns = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
]
machine_path_pattern = re.compile(
    r"(?:[A-Za-z]:\\Users\\(?!<)[^\\\s]+|[A-Za-z]:\\COMSOL62(?:\\|$)|/Users/)",
    re.IGNORECASE,
)
for path in scan_paths:
    if path.suffix.lower() not in {".md", ".json", ".py", ".gitignore"}:
        continue
    if path.name in {"evidence_manifest.json", "adversarial_audit.py"}:
        continue
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if machine_path_pattern.search(text):
        errors.append(f"machine-specific path leaked in {path}")
    for pattern in secret_patterns:
        if pattern.search(text):
            errors.append(f"possible secret in {path}")

manifest = load_json(ROOT / "evidence_manifest.json")
if manifest["file_count"] != len(files) - 1:
    errors.append("evidence manifest file_count does not match folder")
expected_bytes = sum(path.stat().st_size for path in files if path.name != "evidence_manifest.json")
if manifest["total_bytes_excluding_manifest"] != expected_bytes:
    errors.append("evidence manifest total size is stale")
manifest_by_path = {item["path"]: item for item in manifest["files"]}
for path in files:
    if path.name == "evidence_manifest.json":
        continue
    relative = path.relative_to(ROOT).as_posix()
    item = manifest_by_path.get(relative)
    if item is None:
        errors.append(f"evidence manifest is missing {relative}")
    elif item["bytes"] != path.stat().st_size:
        errors.append(f"evidence manifest size is stale for {relative}")
    elif item["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        errors.append(f"evidence manifest hash is stale for {relative}")

largest = max(files, key=lambda path: path.stat().st_size)
notes.extend(
    [
        f"files={len(files)}",
        f"total_bytes={total_bytes}",
        f"largest={largest.relative_to(ROOT).as_posix()}:{largest.stat().st_size}",
        f"sample_counts={dict(counts)}",
        "retrieval_detail_rows=3x427",
        "parser_failure_rows=50",
        f"failed_pdf_smoke_rows={len(smoke_rows)}",
        f"failed_pdf_smoke_mean_f1={float(smoke_summary['mean_char_trigram_f1']):.6f}",
        f"minimal_fix_accepted={int(fixed_summary['quality_accepted'])}/10",
        f"minimal_fix_rejected={len(actual_rejections)}/10",
    ]
)
if errors:
    print("ADVERSARIAL AUDIT: FAILED")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)
print("ADVERSARIAL AUDIT: PASSED")
for note in notes:
    print(f"- {note}")
