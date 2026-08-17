from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
records = json.loads((BENCH / "results" / "cases.json").read_text(encoding="utf-8"))
summary = json.loads((BENCH / "results" / "summary.json").read_text(encoding="utf-8"))
inventory = json.loads((BENCH / "manifests" / "sample_inventory.json").read_text(encoding="utf-8"))
assert len(records) == 70 and len(inventory) == 50
assert Counter(r["source"] for r in records) == Counter({"LongMemEval": 50, "AgentForge": 20})
assert Counter(r["category"] for r in records if r["source"] == "LongMemEval") == Counter(
    {
        "information_extraction": 10,
        "multi_session": 10,
        "knowledge_update": 10,
        "temporal_reasoning": 10,
        "abstention": 10,
    }
)
assert Counter(r["category"] for r in records if r["source"] == "AgentForge") == Counter(
    {"tenant_isolation": 5, "update": 3, "delete": 2, "rolling_summary": 5, "spill": 5}
)
assert summary["pressure_test"] is False
assert abs(summary["overall_pass_rate"] - sum(r["passed"] for r in records) / 70) < 1e-12
assert all(set(row) == {"question_id", "category", "question_sha256"} for row in inventory)
serialized = json.dumps(records, ensure_ascii=False).lower()
assert "haystack_sessions" not in serialized and "api_key" not in serialized
assert sum(p.stat().st_size for p in BENCH.rglob("*") if p.is_file()) < 1_000_000_000
print("Memory benchmark adversarial audit: PASS")
