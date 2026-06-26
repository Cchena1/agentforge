from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_core import WorkspaceAgent
from llm_client import OpenAIChatConfig, OpenAICompatibleChatClient


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class AgentRequestHandler(BaseHTTPRequestHandler):
    server_version = "AIAgent1/1.0"

    @property
    def agent(self) -> WorkspaceAgent:
        return self.server.agent  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            return self._send_json({"ok": True, "service": "python-backend"})
        if path == "/api/llm/config":
            return self._send_json(
                {
                    "ok": True,
                    "defaults": {
                        "baseUrl": self.server.llm_defaults["baseUrl"],  # type: ignore[attr-defined]
                        "model": self.server.llm_defaults["model"],  # type: ignore[attr-defined]
                    },
                    "apiKeyConfigured": bool(self.server.llm_defaults["apiKey"]),  # type: ignore[attr-defined]
                    "endpoint": "/api/llm/chat",
                    "notes": "OpenAI-compatible chat/completions API",
                }
            )
        if path == "/api/state":
            return self._send_json(
                {
                    "workspaceRoot": str(self.agent.workspace_root),
                    "projectRoot": str(self.agent.project_root),
                    "history": self.agent.history[-40:],
                    "memory": self.agent.memory,
                    "textFiles": self.agent.list_text_files(limit=20),
                }
            )
        if path == "/api/file":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            try:
                info = self.agent.read_file(raw_path)
                return self._send_json({"ok": True, "file": info})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _llm_config_from_body(self, body: dict) -> OpenAIChatConfig:
        defaults = self.server.llm_defaults  # type: ignore[attr-defined]
        base_url = str(body.get("baseUrl") or defaults["baseUrl"])
        api_key = str(body.get("apiKey") or defaults["apiKey"])
        model = str(body.get("model") or defaults["model"])
        temperature = float(body.get("temperature", defaults["temperature"]))
        max_tokens_value = body.get("maxTokens", defaults["maxTokens"])
        max_tokens = None if max_tokens_value in (None, "", "null") else int(max_tokens_value)
        timeout = int(body.get("timeout", defaults["timeout"]))
        return OpenAIChatConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            return self._send_json({"ok": False, "error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)

        if path == "/api/chat":
            message = str(body.get("message", ""))
            result = self.agent.handle_message(message)
            return self._send_json({"ok": True, **result})
        if path == "/api/file":
            raw_path = str(body.get("path", ""))
            content = str(body.get("content", ""))
            try:
                result = self.agent.write_file(raw_path, content)
                return self._send_json({"ok": True, "file": result})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if path == "/api/sample-summary":
            raw_paths = body.get("paths")
            try:
                result = self.agent.build_sample_summary(list(raw_paths) if raw_paths else None)
                return self._send_json({"ok": True, "summary": result})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if path == "/api/reset":
            self.agent.history = []
            self.agent.memory = {}
            self.agent._save_state()
            return self._send_json({"ok": True})
        if path == "/api/llm/chat":
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send_json({"ok": False, "error": "`messages` must be a non-empty array"}, status=HTTPStatus.BAD_REQUEST)
            try:
                config = self._llm_config_from_body(body)
                client = OpenAICompatibleChatClient(config)
                parsed_messages = [
                    {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                    for item in messages
                    if isinstance(item, dict)
                ]
                result = client.chat(parsed_messages)
                return self._send_json(
                    {
                        "ok": True,
                        "reply": result["reply"],
                        "model": result["model"],
                        "usage": result["usage"],
                        "raw": result["raw"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    agent = WorkspaceAgent(workspace_root=WORKSPACE_ROOT, project_root=PROJECT_ROOT)
    server = ThreadingHTTPServer((host, port), AgentRequestHandler)
    server.agent = agent  # type: ignore[attr-defined]
    server.llm_defaults = {  # type: ignore[attr-defined]
        "baseUrl": os.getenv("AI_AGENT_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        "apiKey": os.getenv("AI_AGENT_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "model": os.getenv("AI_AGENT_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        "temperature": float(os.getenv("AI_AGENT_LLM_TEMPERATURE", "0.2")),
        "maxTokens": int(os.getenv("AI_AGENT_LLM_MAX_TOKENS", "0")) or None,
        "timeout": int(os.getenv("AI_AGENT_LLM_TIMEOUT", "60")),
    }
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="AI agent-1 local workspace assistant")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    print(f"AI agent-1 running at http://{args.host}:{args.port}")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
