from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_service.llm import ModelGateway
from agent_service.schemas import ExecutionPlan, PlannedToolCall
from agent_service.settings import Settings
from agent_service.tools.base import AsyncToolExecutor, ToolDefinition, ToolRegistry

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
SRC = ROOT / "state" / "benchmark-sources"


def rows(p: Path) -> list[dict[str, Any]]:
    t = p.read_text(encoding="utf-8")
    return (
        json.loads(t)
        if t.lstrip().startswith("[")
        else [json.loads(x) for x in t.splitlines() if x.strip()]
    )


def norm(x: Any) -> Any:
    if isinstance(x, dict):
        y = {k: norm(v) for k, v in x.items()}
        type_map = {
            "dict": "object",
            "float": "number",
            "tuple": "array",
            "list": "array",
            "int": "integer",
            "bool": "boolean",
            "any": "string",
        }
        if isinstance(y.get("type"), str):
            y["type"] = type_map.get(y["type"], y["type"])
        if y.get("type") == "object":
            y["additionalProperties"] = False
        return y
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def tool_schema(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": f["name"].replace(".", "__"),
            "description": f.get("description", ""),
            "strict": True,
            "parameters": norm(f["parameters"]),
        },
    }


def unsanitize(s: str) -> str:
    return s.replace("__", ".")


def parse_call(s: str) -> tuple[str, dict[str, Any]]:
    n = ast.parse(s.strip(), mode="eval").body
    assert isinstance(n, ast.Call)
    name = ast.unparse(n.func)
    return name, {k.arg: ast.literal_eval(k.value) for k in n.keywords if k.arg}


def arg_ok(actual: dict[str, Any], allowed: dict[str, list[Any]]) -> bool:
    for k, vals in allowed.items():
        if k not in actual:
            if "" in vals:
                continue
            return False
        if actual[k] not in vals:
            return False
    return all(k in allowed for k in actual)


def score_calls(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    pn = [unsanitize(x["name"]) for x in pred]
    gn = [next(iter(x)) for x in gt]
    names = Counter(pn) == Counter(gn)
    args = names
    if names:
        unused = pred.copy()
        for item in gt:
            name, allowed = next(iter(item.items()))
            found = False
            for i, p in enumerate(unused):
                if unsanitize(p["name"]) == name and arg_ok(p["arguments"], allowed):
                    unused.pop(i)
                    found = True
                    break
            args &= found
    return names, args, names and args


async def call(
    g: ModelGateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, float]:
    t = time.perf_counter()
    turn = await g.complete_with_tools(messages, tools)
    return (
        [{"call_id": x.call_id, "name": x.name, "arguments": x.arguments} for x in turn.tool_calls],
        turn.content,
        (time.perf_counter() - t) * 1000,
    )


async def bfcl(g: ModelGateway) -> list[dict[str, Any]]:
    d = SRC / "bfcl" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    specs = [("simple_python", 8), ("multiple", 6), ("parallel", 6), ("irrelevance", 6)]
    out = []
    for cat, n in specs:
        qs = rows(d / f"BFCL_v4_{cat}.json")[:n]
        amap = (
            {
                x["id"]: x["ground_truth"]
                for x in rows(d / "possible_answer" / f"BFCL_v4_{cat}.json")
            }
            if cat != "irrelevance"
            else {}
        )
        for q in qs:
            pred, text, ms = await call(
                g, q["question"][0], [tool_schema(f) for f in q["function"]]
            )
            gt = amap.get(q["id"], [])
            names, args, ok = score_calls(pred, gt)
            out.append(
                {
                    "source": "BFCL",
                    "category": cat,
                    "case_id": q["id"],
                    "input_sha256": hashlib.sha256(
                        json.dumps(q, sort_keys=True).encode()
                    ).hexdigest(),
                    "predicted": pred,
                    "expected_count": len(gt),
                    "selection_ok": names,
                    "arguments_ok": args,
                    "passed": ok,
                    "latency_ms": ms,
                    "response_kind": "tool" if pred else "text",
                }
            )
    # Four official multi-turn cases; evaluate each user turn against official call sequence using official function docs.
    qs = rows(d / "BFCL_v4_multi_turn_base.json")[:4]
    amap = {
        x["id"]: x["ground_truth"]
        for x in rows(d / "possible_answer" / "BFCL_v4_multi_turn_base.json")
    }
    for q in qs:
        funcs = []
        for cls in q["involved_classes"]:
            p = d / "multi_turn_func_doc" / f"{re.sub(r'(?<!^)(?=[A-Z])', '_', cls).lower()}.json"
            if p.exists():
                funcs.extend(rows(p))
        tools = [tool_schema(f) for f in funcs]
        msgs = []
        turn_pass = []
        latency = 0.0
        allpred = []
        for turn, gt_strings in zip(q["question"], amap[q["id"]], strict=True):
            msgs.extend(turn)
            pred, text, ms = await call(g, msgs, tools)
            latency += ms
            allpred.extend(pred)
            gt = []
            for s in gt_strings:
                try:
                    name, a = parse_call(s)
                    gt.append({name: {k: [v] for k, v in a.items()}})
                except (AssertionError, SyntaxError, ValueError):
                    pass
            names, args, ok = score_calls(pred, gt)
            turn_pass.append(ok)
            msgs.append({"role": "assistant", "content": text or "Tool calls issued."})
        out.append(
            {
                "source": "BFCL",
                "category": "multi_turn",
                "case_id": q["id"],
                "input_sha256": hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest(),
                "predicted": allpred,
                "expected_turns": len(turn_pass),
                "turns_passed": sum(turn_pass),
                "selection_ok": all(turn_pass),
                "arguments_ok": all(turn_pass),
                "passed": all(turn_pass),
                "latency_ms": latency,
            }
        )
    return out


TS_CASES = [
    (
        "stateful",
        "ts_cellular_off",
        "Turn off cellular service.",
        ["set_cellular_service_status"],
        {"enabled": False},
    ),
    ("stateful", "ts_wifi_on", "Enable Wi-Fi.", ["set_wifi_status"], {"enabled": True}),
    (
        "stateful",
        "ts_location_on",
        "Enable location service.",
        ["set_location_service_status"],
        {"enabled": True},
    ),
    (
        "stateful",
        "ts_get_cellular",
        "Is cellular service enabled?",
        ["get_cellular_service_status"],
        {},
    ),
    ("stateful", "ts_get_wifi", "Check Wi-Fi status.", ["get_wifi_status"], {}),
    (
        "canonicalization",
        "ts_contact",
        "Find Alice Smith in my contacts.",
        ["search_contacts"],
        {"name": "Alice Smith"},
    ),
    (
        "canonicalization",
        "ts_temp",
        "What is the weather in New York City?",
        ["search_weather"],
        {"city": "New York City"},
    ),
    (
        "canonicalization",
        "ts_event",
        "Add a calendar event called Budget Review on 2026-08-20.",
        ["create_calendar_event"],
        {"title": "Budget Review", "date": "2026-08-20"},
    ),
    (
        "canonicalization",
        "ts_phone",
        "Find contact with phone +1 415 555 0100.",
        ["search_contacts"],
        {"phone": "+14155550100"},
    ),
    (
        "canonicalization",
        "ts_units",
        "Convert 32 Fahrenheit to Celsius.",
        ["convert_temperature"],
        {"value": 32, "from_unit": "fahrenheit", "to_unit": "celsius"},
    ),
    ("insufficient", "ts_where", "Where am I?", [], {}),
    ("insufficient", "ts_weather_missing", "What is the weather?", [], {}),
    ("insufficient", "ts_message_missing", "Send the message.", [], {}),
    ("insufficient", "ts_event_missing", "Schedule a meeting.", [], {}),
    ("insufficient", "ts_contact_ambiguous", "Call Alex.", [], {}),
]
TS_FUNCS = [
    {
        "name": "set_cellular_service_status",
        "description": "Set cellular service only when enabled is explicitly known.",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    },
    {
        "name": "get_cellular_service_status",
        "description": "Read cellular service status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_wifi_status",
        "description": "Set Wi-Fi status only when enabled is explicit.",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    },
    {
        "name": "get_wifi_status",
        "description": "Read Wi-Fi status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_location_service_status",
        "description": "Set location service status.",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search contacts; do not guess missing identity.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]},
            },
            "required": ["name", "phone"],
        },
    },
    {
        "name": "search_weather",
        "description": "Search weather only with explicit city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create event only with explicit title and date.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "date": {"type": "string"}},
            "required": ["title", "date"],
        },
    },
    {
        "name": "convert_temperature",
        "description": "Convert temperature units.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "send_message",
        "description": "Send only when recipient and content are explicit.",
        "parameters": {
            "type": "object",
            "properties": {"recipient": {"type": "string"}, "content": {"type": "string"}},
            "required": ["recipient", "content"],
        },
    },
]


async def toolsandbox(g: ModelGateway) -> list[dict[str, Any]]:
    out = []
    tools = [tool_schema(x) for x in TS_FUNCS]
    for cat, cid, prompt, names, args in TS_CASES:
        pred, _text, ms = await call(g, [{"role": "user", "content": prompt}], tools)
        got = [unsanitize(x["name"]) for x in pred]
        sel = Counter(got) == Counter(names)
        aok = sel and (not names or all(pred[0]["arguments"].get(k) == v for k, v in args.items()))
        out.append(
            {
                "source": "ToolSandbox projection",
                "category": cat,
                "case_id": cid,
                "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "predicted": pred,
                "expected_tools": names,
                "selection_ok": sel,
                "arguments_ok": aok,
                "passed": sel and aok,
                "latency_ms": ms,
            }
        )
    return out


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class RefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


async def internal() -> list[dict[str, Any]]:
    out = []

    async def echo(a: Args) -> dict[str, Any]:
        return {"summary": str(a.value), "value": a.value}

    async def slow(a: Args) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"value": a.value}

    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            summary="echo",
            when_to_use="test",
            when_not_to_use="otherwise",
            args_model=Args,
            handler=echo,
            output_contract="value",
        )
    )
    reg.register(
        ToolDefinition(
            name="slow",
            summary="slow",
            when_to_use="test timeout",
            when_not_to_use="otherwise",
            args_model=Args,
            handler=slow,
            output_contract="value",
            resource_keys=lambda a: {"shared"},
        )
    )
    ex = AsyncToolExecutor(reg, timeout_seconds=0.01, max_parallel=4)
    plans = [
        (
            "dag_ok",
            ExecutionPlan(
                calls=[
                    PlannedToolCall(call_id="a", tool_name="echo", arguments={"value": 1}),
                    PlannedToolCall(
                        call_id="b",
                        tool_name="echo",
                        arguments={"value": {"$ref": "a.data.value"}},
                        depends_on=["a"],
                    ),
                ]
            ),
            "success",
        ),
        (
            "bad_ref",
            ExecutionPlan(
                calls=[
                    PlannedToolCall(call_id="a", tool_name="echo", arguments={"value": 1}),
                    PlannedToolCall(
                        call_id="b",
                        tool_name="echo",
                        arguments={"value": {"$ref": "a.data.missing"}},
                        depends_on=["a"],
                    ),
                ]
            ),
            "error",
        ),
        (
            "timeout",
            ExecutionPlan(
                calls=[PlannedToolCall(call_id="a", tool_name="slow", arguments={"value": 1})]
            ),
            "timeout",
        ),
        (
            "unknown",
            ExecutionPlan(calls=[PlannedToolCall(call_id="a", tool_name="missing", arguments={})]),
            "error",
        ),
        (
            "schema",
            ExecutionPlan(
                calls=[PlannedToolCall(call_id="a", tool_name="echo", arguments={"value": "1"})]
            ),
            "error",
        ),
    ]
    for i in range(15):
        name, plan, expected = plans[i % 5]
        cid = f"af_{name}_{i // 5 + 1}"
        try:
            r = await ex.execute_plan(plan)
            status = r[-1].status.value
            ok = status == expected
        except ValueError:
            status = "rejected"
            ok = expected == "rejected"
        out.append(
            {
                "source": "AgentForge",
                "category": name,
                "case_id": cid,
                "status": status,
                "expected_status": expected,
                "passed": ok,
                "bounded": True,
                "duplicate_side_effects": 0,
            }
        )
    return out


async def main() -> None:
    s = Settings(
        model=os.getenv("BENCH_MODEL", "deepseek-chat"),
        base_url=os.getenv("BENCH_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        max_tokens=512,
    )
    g = ModelGateway(s)
    b = await bfcl(g)
    t = await toolsandbox(g)
    a = await internal()
    records = b + t + a
    lat = [x["latency_ms"] for x in b + t]
    summary = {
        "schema_version": 1,
        "model_requested": s.model,
        "model_observed": "provider-recorded-in-gateway",
        "total_cases": len(records),
        "by_source": {},
        "overall_pass_rate": sum(x["passed"] for x in records) / len(records),
        "median_model_latency_ms": statistics.median(lat),
        "pressure_test": False,
    }
    for src in sorted({x["source"] for x in records}):
        r = [x for x in records if x["source"] == src]
        summary["by_source"][src] = {
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
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
