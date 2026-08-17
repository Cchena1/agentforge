from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
records = json.loads((BENCH / "results" / "cases.json").read_text(encoding="utf-8"))
summary = json.loads((BENCH / "results" / "summary.json").read_text(encoding="utf-8"))
assert len(records) == 60
assert Counter(r["source"] for r in records) == Counter(
    {"BFCL": 30, "ToolSandbox projection": 15, "AgentForge": 15}
)
assert Counter(r["category"] for r in records if r["source"] == "BFCL") == Counter(
    {"simple_python": 8, "multiple": 6, "parallel": 6, "irrelevance": 6, "multi_turn": 4}
)
assert Counter(
    r["category"] for r in records if r["source"] == "ToolSandbox projection"
) == Counter({"stateful": 5, "canonicalization": 5, "insufficient": 5})
assert summary["pressure_test"] is False
assert abs(summary["overall_pass_rate"] - sum(r["passed"] for r in records) / 60) < 1e-12
serialized = json.dumps(records, ensure_ascii=False).lower()
assert "api_key" not in serialized and "bearer " not in serialized
assert sum(p.stat().st_size for p in BENCH.rglob("*") if p.is_file()) < 1_000_000_000
print("Tool benchmark adversarial audit: PASS")
