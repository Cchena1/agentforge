from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_service.context_runtime import ToolResultSpillStore
from agent_service.embeddings import HashEmbedding
from agent_service.memory import MemoryStore
from agent_service.schemas import ChatMessage, MemoryWrite, ToolResult, ToolStatus

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
DATA = ROOT / "state" / "benchmark-sources" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"


def stream_array(path: Path) -> Iterator[dict[str, Any]]:
    dec = json.JSONDecoder()
    buf = ""
    pos = 0
    with path.open(encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk and pos >= len(buf):
                break
            buf = buf[pos:] + chunk
            pos = 0
            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] in "[,"):
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = end
            if not chunk:
                break


def cat(x: dict[str, Any]) -> str:
    if str(x["question_id"]).endswith("_abs"):
        return "abstention"
    q = x["question_type"]
    if q in {"single-session-user", "single-session-assistant", "single-session-preference"}:
        return "information_extraction"
    if q == "multi-session":
        return "multi_session"
    if q == "knowledge-update":
        return "knowledge_update"
    return "temporal_reasoning"


def select() -> list[dict[str, Any]]:
    buckets = {
        k: []
        for k in [
            "information_extraction",
            "multi_session",
            "knowledge_update",
            "temporal_reasoning",
            "abstention",
        ]
    }
    for x in stream_array(DATA):
        c = cat(x)
        if len(buckets[c]) < 10:
            buckets[c].append(x)
        if all(len(v) == 10 for v in buckets.values()):
            break
    return [x for v in buckets.values() for x in v]


async def official(cases: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    store = MemoryStore(
        root / "longmem.sqlite3", HashEmbedding(), message_limit=8, char_budget=6000
    )
    out = []
    for x in cases:
        ns = f"benchmark:{x['question_id']}"
        for sid, date, sess in zip(
            x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"], strict=True
        ):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in sess)

            for part_index, start in enumerate(range(0, len(text), 18000)):
                await store.remember(
                    MemoryWrite(
                        namespace=ns,
                        text=text[start : start + 18000],
                        importance=0.5,
                        metadata={"session_id": sid, "date": date, "part_index": part_index},
                    )
                )
        t = time.perf_counter()
        hits = await store.recall(ns, x["question"], top_k=5)
        ms = (time.perf_counter() - t) * 1000
        got = [h.metadata.get("session_id") for h in hits]
        expected = set(x.get("answer_session_ids", []))
        c = cat(x)
        if c == "abstention":
            passed = len(got) == 0
            recall = 1.0 if passed else 0.0
        else:
            recall = len(expected.intersection(got)) / len(expected) if expected else 0.0
            passed = recall == 1.0
        out.append(
            {
                "source": "LongMemEval",
                "category": c,
                "case_id": x["question_id"],
                "question_sha256": hashlib.sha256(x["question"].encode()).hexdigest(),
                "expected_evidence_count": len(expected),
                "retrieved_session_ids": got,
                "evidence_recall_at_5": recall,
                "passed": passed,
                "latency_ms": ms,
            }
        )
    return out


async def internal(root: Path) -> list[dict[str, Any]]:
    out = []
    # 5 namespace isolation cases
    for i in range(5):
        s = MemoryStore(
            root / f"iso{i}.sqlite3", HashEmbedding(), message_limit=4, char_budget=1000
        )
        a = s.namespace(f"u{i}", "s")
        b = s.namespace(f"other{i}", "s")
        await s.remember(MemoryWrite(namespace=a, text=f"alpha secret {i}", importance=1))
        await s.remember(MemoryWrite(namespace=b, text=f"beta secret {i}", importance=1))
        h = await s.recall(a, "alpha secret", 5)
        ok = all(x.namespace == a and "beta" not in x.text for x in h)
        out.append(
            {
                "source": "AgentForge",
                "category": "tenant_isolation",
                "case_id": f"iso_{i}",
                "passed": ok,
            }
        )
    # 3 update preference + 2 deletion capability tests
    for i in range(3):
        s = MemoryStore(
            root / f"upd{i}.sqlite3", HashEmbedding(), message_limit=4, char_budget=1000
        )
        ns = f"user:update{i}"
        await s.remember(MemoryWrite(namespace=ns, text="Preferred city is Paris", importance=0.2))
        await s.remember(
            MemoryWrite(namespace=ns, text="Preferred city is Tokyo (latest)", importance=1)
        )
        h = await s.recall(ns, "latest preferred city", 1)
        ok = bool(h and "Tokyo" in h[0].text)
        out.append(
            {"source": "AgentForge", "category": "update", "case_id": f"update_{i}", "passed": ok}
        )
    for i in range(2):
        out.append(
            {
                "source": "AgentForge",
                "category": "delete",
                "case_id": f"delete_{i}",
                "passed": False,
                "failure_code": "PUBLIC_DELETE_API_MISSING",
            }
        )
    for i in range(5):
        s = MemoryStore(
            root / f"sum{i}.sqlite3", HashEmbedding(), message_limit=4, char_budget=2000
        )
        ns = s.namespace(f"u{i}", "sum")
        msgs = [ChatMessage(role="user", content=f"critical fact {i}")] + [
            ChatMessage(role="assistant", content=f"filler {j}") for j in range(7)
        ]
        await s.append_messages(ns, msgs)
        summary, ctx = await s.load_context(ns)
        ok = f"critical fact {i}" in summary and len(ctx) <= 4
        out.append(
            {
                "source": "AgentForge",
                "category": "rolling_summary",
                "case_id": f"summary_{i}",
                "passed": ok,
                "summary_chars": len(summary),
            }
        )
    for i in range(5):
        spill = ToolResultSpillStore(root / f"spill{i}", inline_token_limit=128, preview_chars=256)
        r = ToolResult(
            call_id=str(i),
            tool_name="large",
            status=ToolStatus.SUCCESS,
            data={"content": "evidence" * 3000},
            summary="large",
            latency_ms=1,
        )
        rendered = await spill.render(r)
        ok = rendered.spilled and spill.path_for(rendered.artifact).exists()
        out.append(
            {
                "source": "AgentForge",
                "category": "spill",
                "case_id": f"spill_{i}",
                "passed": ok,
                "spilled": rendered.spilled,
            }
        )
    return out


async def main() -> None:
    cases = select()
    assert len(cases) == 50
    with tempfile.TemporaryDirectory(prefix="agentforge-memory-bench-") as td:
        root = Path(td)
        o = await official(cases, root)
        i = await internal(root)
    records = o + i
    lat = [x["latency_ms"] for x in o]
    summary = {
        "schema_version": 1,
        "total_cases": len(records),
        "official_cases": 50,
        "internal_cases": 20,
        "overall_pass_rate": sum(x["passed"] for x in records) / len(records),
        "longmemeval_recall_at_5": sum(x["evidence_recall_at_5"] for x in o) / len(o),
        "median_retrieval_latency_ms": statistics.median(lat),
        "by_category": {},
        "pressure_test": False,
    }
    for c in sorted({x["category"] for x in records}):
        r = [x for x in records if x["category"] == c]
        summary["by_category"][c] = {
            "cases": len(r),
            "passed": sum(x["passed"] for x in r),
            "pass_rate": sum(x["passed"] for x in r) / len(r),
        }
    (BENCH / "results").mkdir(parents=True, exist_ok=True)
    (BENCH / "results" / "cases.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BENCH / "results" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BENCH / "manifests").mkdir(exist_ok=True)
    (BENCH / "manifests" / "sample_inventory.json").write_text(
        json.dumps(
            [
                {
                    "question_id": x["question_id"],
                    "category": cat(x),
                    "question_sha256": hashlib.sha256(x["question"].encode()).hexdigest(),
                }
                for x in cases
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
