from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT / "eval" / "基于OmniDocBench和OHR-Bench的RAG基准测试"
ONE_GIB = 1024**3
forbidden_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".sqlite", ".sqlite3", ".db", ".pt", ".bin"}
allowed_query_keys = {
    "qa_id", "domain", "evidence_source", "answer_form", "relevant_source_ids",
    "hit_source_ids", "first_relevant_rank", "latency_ms", "citation_valid",
}
errors: list[str] = []
notes: list[str] = []
files = [p for p in ROOT.rglob("*") if p.is_file()]
total_bytes = sum(p.stat().st_size for p in files)
if total_bytes >= ONE_GIB:
    errors.append(f"benchmark folder is {total_bytes} bytes, exceeding 1 GiB")
for path in files:
    if path.suffix.lower() in forbidden_suffixes:
        errors.append(f"forbidden binary/dataset artifact: {path}")
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {path}: {exc}")
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"invalid Python {path}: {exc}")

inventory = json.loads((ROOT / "manifests/sample_inventory.json").read_text(encoding="utf-8"))
samples = inventory["samples"]
if inventory["unique_pages"] != 200 or len(samples) != 200:
    errors.append("sample inventory does not contain exactly 200 pages")
ids = [s["sample_id"] for s in samples]
if len(ids) != len(set(ids)):
    errors.append("duplicate sample_id in inventory")
counts = Counter(s["benchmark"] for s in samples)
if counts != Counter({"OmniDocBench": 50, "OHR-Bench": 150}):
    errors.append(f"unexpected benchmark allocation: {dict(counts)}")
raw_text_keys = {"gt_text", "text", "question", "answer", "formatting_noise_moderate", "semantic_noise_mineru_moderate"}
for index, sample in enumerate(samples):
    overlap = {k.lower() for k in sample} & raw_text_keys
    if overlap:
        errors.append(f"raw text field in inventory sample {index}: {sorted(overlap)}")

summary = json.loads((ROOT / "results/ohr_retrieval_summary.json").read_text(encoding="utf-8"))
summary_by_variant = {item["variant"]: item for item in summary}
query_files = sorted((ROOT / "results").glob("ohr_retrieval_*.json"))
query_files = [p for p in query_files if p.name != "ohr_retrieval_summary.json"]
if len(query_files) != 3:
    errors.append(f"expected 3 detailed retrieval files, found {len(query_files)}")
for path in query_files:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 427:
        errors.append(f"{path.name}: expected 427 rows, found {len(rows)}")
    for idx, row in enumerate(rows):
        keys = set(row)
        if keys != allowed_query_keys:
            errors.append(f"{path.name}[{idx}] unexpected keys: {sorted(keys ^ allowed_query_keys)}")
            break
        for value in row.values():
            if isinstance(value, str) and len(value) > 512:
                errors.append(f"{path.name}[{idx}] contains oversized text value")
                break
    ranks = [row["first_relevant_rank"] for row in rows]
    calc = {
        "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr_at_5": sum((1 / rank) if rank is not None and rank <= 5 else 0 for rank in ranks) / len(ranks),
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

failures = json.loads((ROOT / "results/omni_parser_failures_sanitized.json").read_text(encoding="utf-8"))
if len(failures) != 50:
    errors.append(f"expected 50 sanitized parser failures, found {len(failures)}")
serialized_failures = json.dumps(failures, ensure_ascii=False)
if re.search(r"[A-Za-z]:\\|/Users/|\\Users\\|陈楚凡", serialized_failures):
    errors.append("parser failure evidence contains an absolute/user path")

scan_paths = [*files, PROJECT_ROOT / "README.md", PROJECT_ROOT / "report.md"]
secret_patterns = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
]
for path in scan_paths:
    if path.suffix.lower() not in {".md", ".json", ".py", ".gitignore"} and path.name != ".gitignore":
        continue
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if "E:\\COMSOL62" in text or "C:\\Users\\陈楚凡" in text:
        errors.append(f"machine-specific path leaked in {path}")
    for pattern in secret_patterns:
        if pattern.search(text):
            errors.append(f"possible secret in {path}")

manifest = json.loads((ROOT / "evidence_manifest.json").read_text(encoding="utf-8"))
if manifest["file_count"] != len(files) - 1:
    errors.append("evidence manifest file_count does not match folder")
if manifest["total_bytes_excluding_manifest"] != sum(p.stat().st_size for p in files if p.name != "evidence_manifest.json"):
    errors.append("evidence manifest total size is stale")

largest = max(files, key=lambda p: p.stat().st_size)
notes.extend([
    f"files={len(files)}",
    f"total_bytes={total_bytes}",
    f"largest={largest.relative_to(ROOT).as_posix()}:{largest.stat().st_size}",
    f"sample_counts={dict(counts)}",
    "retrieval_detail_rows=3x427",
    "parser_failure_rows=50",
])
if errors:
    print("ADVERSARIAL AUDIT: FAILED")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)
print("ADVERSARIAL AUDIT: PASSED")
for note in notes:
    print(f"- {note}")
