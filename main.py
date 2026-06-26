from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from config import settings


logger = logging.getLogger("ai_agent_backend")
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an AI Agent backend service.
Answer briefly and clearly.
If the user asks to read a file or inspect a text file, call read_file_tool first.
After reading a file, summarize its content with a concise abstract and key points.
Maintain the conversation context provided by the caller.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file_tool",
            "description": "Read a text file from the server and return a concise text summary for the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the text file to read. Can be absolute or relative to the workspace.",
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    }
]

SESSION_LOCK = Lock()
SESSION_STORE: dict[str, list[dict[str, str]]] = {}


class ChatMessage(BaseModel):
    role: str = Field(..., description="user, assistant, system, or tool")
    content: str = Field(default="")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]]


def _safe_preview(text: str, limit: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _split_text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _ensure_demo_file() -> Path:
    demo_path = Path(tempfile.gettempdir()) / settings.demo_file_name
    if not demo_path.exists():
        demo_path.write_text(settings.demo_file_content, encoding="utf-8")
    return demo_path


def _candidate_paths(file_path: str) -> list[Path]:
    raw = file_path.strip().strip('"').strip("'")
    path_obj = Path(raw)
    candidates: list[Path] = [
        Path.cwd() / path_obj.name,
        settings.workspace_root / path_obj.name,
    ]

    if raw.startswith("/tmp/"):
        suffix = raw[len("/tmp/") :]
        candidates.extend(
            [
                Path(tempfile.gettempdir()) / suffix,
                settings.workspace_root / suffix,
            ]
        )

    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.extend([settings.workspace_root / path_obj, Path.cwd() / path_obj])

    if path_obj.name == settings.demo_file_name:
        candidates.extend(
            [
                Path(tempfile.gettempdir()) / settings.demo_file_name,
                settings.workspace_root / settings.demo_file_name,
            ]
        )

    seen: set[str] = set()
    unique: list[Path] = []
    for item in candidates:
        resolved = str(item.resolve() if item.exists() else item)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)
    return unique


def resolve_file_path(file_path: str) -> Path:
    for candidate in _candidate_paths(file_path):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"File not found: {file_path}")


def _read_text_file(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16", "utf-16-le", "utf-16-be")
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        logger.debug("Falling back to utf-8 replacement for %s: %s", path, last_error)
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _summarize_document(text: str, max_points: int = 3) -> str:
    lines = _split_text_lines(text)
    if not lines:
        return "empty file"

    keywords = ("因此", "所以", "主要", "目的", "结论", "总结", "结果", "要求", "问题", "变化", "影响")
    key_lines: list[str] = []
    for line in lines:
        if any(keyword in line for keyword in keywords):
            key_lines.append(line)
        if len(key_lines) >= max_points:
            break

    if not key_lines:
        key_lines = lines[:max_points]

    return "；".join(_safe_preview(line, 120) for line in key_lines[:max_points])


def summarize_text(text: str, path_label: str) -> str:
    lines = _split_text_lines(text)
    if not lines:
        return f"{path_label}: empty file"

    preview = _safe_preview(" ".join(lines[:20]), 300)
    return f"{path_label}: {len(lines)} non-empty lines. Summary: {preview}"


def read_file_tool(file_path: str) -> str:
    resolved = resolve_file_path(file_path)
    text, encoding = _read_text_file(resolved)
    summary = summarize_text(text, str(resolved))
    doc_summary = _summarize_document(text)
    return (
        f"Path: {resolved}\n"
        f"Encoding: {encoding}\n"
        f"Characters: {len(text)}\n"
        f"Lines: {len(text.splitlines())}\n"
        f"Summary: {summary}\n"
        f"Document summary: {doc_summary}\n"
        f"Preview: {_safe_preview(text, 800)}"
    )


def _normalize_history(history: list[ChatMessage] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if item.role not in {"user", "assistant", "system", "tool"}:
            continue
        normalized.append({"role": item.role, "content": item.content or ""})
    return normalized


def _merge_current_message(history: list[dict[str, str]], message: str) -> list[dict[str, str]]:
    current = {"role": "user", "content": message}
    if not history:
        return [current]
    last = history[-1]
    if last.get("role") == "user" and last.get("content") == message:
        return history
    return [*history, current]


def _extract_path_hint(text: str) -> str | None:
    patterns = [
        r"(/tmp/[^\s'\"`]+\.txt)",
        r"([A-Za-z]:[\\/][^\s'\"`]+\.txt)",
        r"([A-Za-z0-9_.-]+\.txt)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _offline_reply(message: str, history: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    tool_calls: list[dict[str, Any]] = []
    normalized_message = message.strip().lower()
    continuation_words = ["继续", "接着", "上一个", "follow up", "continue"]
    last_assistant = next((item["content"] for item in reversed(history) if item["role"] == "assistant"), "")

    if last_assistant and (
        any(word in normalized_message for word in continuation_words)
        or any(word in message for word in continuation_words)
        or (len(message.strip()) <= 6 and not _extract_path_hint(message))
    ):
        return (
            f"我记得上一轮已经处理了这个文件。上一次的核心摘要是：{_safe_preview(last_assistant, 180)}。"
            " 如果你要，我可以继续提炼要点或者帮你写成总结文件。",
            tool_calls,
        )

    hint = _extract_path_hint(message) or _extract_path_hint(" ".join(item["content"] for item in history[-4:]))
    if hint:
        try:
            tool_output = read_file_tool(hint)
            summary_line = next(
                (line for line in tool_output.splitlines() if line.startswith("Document summary: ")),
                "",
            )
            reply = f"已读取文件。{summary_line or _safe_preview(tool_output, 240)}"
            tool_calls.append(
                {
                    "name": "read_file_tool",
                    "arguments": {"file_path": hint},
                    "result": _safe_preview(tool_output, 500),
                }
            )
            return reply, tool_calls
        except Exception as exc:  # noqa: BLE001
            return f"读取文件失败：{exc}", tool_calls

    if any(word in message for word in ["你好", "hello", "hi"]):
        return "你好！有什么可以帮助你？", tool_calls

    if history:
        last_assistant = next((item["content"] for item in reversed(history) if item["role"] == "assistant"), "")
        if last_assistant:
            return f"我记得上一轮的上下文是：{_safe_preview(last_assistant, 120)}。你可以继续让我处理文件或者总结内容。", tool_calls

    return "我可以帮你读取文本文件、总结内容，并保持多轮对话上下文。", tool_calls


def _build_client() -> OpenAI | None:
    if not settings.llm_api_key.strip():
        return None
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )


def _call_model_with_tools(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    client = _build_client()
    if client is None:
        raise RuntimeError("LLM API key is not configured")

    tool_records: list[dict[str, Any]] = []
    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    if settings.debug:
        logger.debug(
            "LLM request -> model=%s base_url=%s messages=%s",
            settings.llm_model,
            settings.llm_base_url,
            request_messages,
        )

    first_response = client.chat.completions.create(
        model=settings.llm_model,
        messages=request_messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    first_choice = first_response.choices[0].message
    if settings.debug:
        logger.debug("LLM first response -> %s", first_response.model_dump())

    if first_choice.tool_calls:
        assistant_message = {
            "role": "assistant",
            "content": first_choice.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in first_choice.tool_calls
            ],
        }
        followup_messages: list[dict[str, Any]] = [*request_messages, assistant_message]

        for tool_call in first_choice.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            if tool_name != "read_file_tool":
                result_text = f"Unsupported tool: {tool_name}"
            else:
                file_path = str(tool_args.get("file_path", "")).strip()
                result_text = read_file_tool(file_path)

            tool_records.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": _safe_preview(result_text, 500),
                }
            )
            followup_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

        if settings.debug:
            logger.debug("LLM tool records -> %s", tool_records)

        final_response = client.chat.completions.create(
            model=settings.llm_model,
            messages=followup_messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        final_choice = final_response.choices[0].message
        if settings.debug:
            logger.debug("LLM final response -> %s", final_response.model_dump())
        return final_choice.content or "", tool_records

    return first_choice.content or "", tool_records


def _store_session(session_id: str, messages: list[dict[str, str]]) -> None:
    with SESSION_LOCK:
        SESSION_STORE[session_id] = messages


def _load_session(session_id: str) -> list[dict[str, str]]:
    with SESSION_LOCK:
        return [dict(item) for item in SESSION_STORE.get(session_id, [])]


@app.on_event("startup")
def _startup() -> None:
    _ensure_demo_file()
    logger.info("Demo file ready at %s/test.txt", tempfile.gettempdir())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, Any]:
    return {
        "debug": settings.debug,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "llm_max_tokens": settings.llm_max_tokens,
        "cors_origins": settings.cors_origins,
        "workspace_root": str(settings.workspace_root),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.session_id:
        history = request.history if request.history is not None else [ChatMessage(**item) for item in _load_session(request.session_id)]
    else:
        history = request.history or []

    normalized_history = _normalize_history(history)
    messages = _merge_current_message(normalized_history, request.message)

    if settings.debug:
        logger.debug("Incoming /chat request -> %s", request.model_dump())
        logger.debug("Composed messages -> %s", messages)

    try:
        reply, tool_calls = _call_model_with_tools(messages)
    except Exception as exc:  # noqa: BLE001
        if settings.llm_api_key.strip():
            logger.exception("LLM request failed, falling back to offline mode")
        else:
            logger.info("LLM unavailable, using offline mode: %s", exc)
        reply, tool_calls = _offline_reply(request.message, messages)

    assistant_message = {"role": "assistant", "content": reply}
    if request.session_id:
        _store_session(request.session_id, [*messages, assistant_message])

    response = ChatResponse(reply=reply, tool_calls=tool_calls)
    if settings.debug:
        logger.debug("Response -> %s", response.model_dump())
    return response
