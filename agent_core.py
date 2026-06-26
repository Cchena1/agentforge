from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _short_preview(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


@dataclass
class WorkspaceAgent:
    workspace_root: Path
    project_root: Path
    state_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    state_file: Path = field(init=False)
    history: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()
        self.project_root = self.project_root.resolve()
        self.state_dir = self.project_root / "state"
        self.output_dir = self.project_root / "output"
        self.state_file = self.state_dir / "conversation.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(_read_text_with_fallback(self.state_file))
        except json.JSONDecodeError:
            return
        self.history = list(data.get("history", []))
        self.memory = dict(data.get("memory", {}))

    def _save_state(self) -> None:
        payload = {
            "workspace_root": str(self.workspace_root),
            "project_root": str(self.project_root),
            "history": self.history[-200:],
            "memory": self.memory,
            "saved_at": _now_iso(),
        }
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path.strip().strip('"').strip("'"))
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError("Path escapes workspace root")
        return resolved

    def list_text_files(self, limit: int = 20) -> list[str]:
        results: list[str] = []
        for path in sorted(self.workspace_root.rglob("*.txt")):
            try:
                rel = path.relative_to(self.workspace_root)
            except ValueError:
                continue
            if not rel.parts:
                continue
            if rel.parts[0] == self.project_root.name:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            if path.name.startswith("~$"):
                continue
            results.append(str(path))
            if len(results) >= limit:
                break
        return results

    def read_file(self, raw_path: str) -> dict[str, Any]:
        path = self._resolve_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = _read_text_with_fallback(path)
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "content": text,
            "preview": _short_preview(text),
        }

    def write_file(self, raw_path: str, content: str) -> dict[str, Any]:
        path = self._resolve_path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "size": len(content.encode("utf-8"))}

    def _summarize_text(self, path: Path, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return f"- {path.name}: 空文件或只有空白内容。"

        header = lines[:5]
        body = _short_preview(" ".join(lines[5:25]), 220)
        if "CorrTest for Windows" in text or "Open Circuit Potential" in text:
            ocp = re.search(r"Open Circuit Potential \(V\):\s*([-\d.]+)", text)
            temp = re.search(r"Temperature\(℃\):\s*([-\d.]+)", text)
            details = []
            if ocp:
                details.append(f"OCP 约 {ocp.group(1)} V")
            if temp:
                details.append(f"温度 {temp.group(1)} ℃")
            if re.search(r"^E\(V\)\s+i\(A/cm2\)\s+T\(s\)", text, re.M):
                details.append("包含 E(V) / i(A/cm2) / T(s) 三列数据")
            detail_text = "，".join(details) if details else "包含电化学测试元数据和原始曲线数据"
            return f"- {path.name}: 电化学测试日志，{detail_text}。首行摘要：{_short_preview(' | '.join(header), 180)}"

        if re.match(r"^[+-]?\d", lines[0]) and len(lines) > 5:
            return (
                f"- {path.name}: 数值型数据表，前几行显示成对数值，"
                f"适合做曲线或趋势分析。首行摘要：{_short_preview(' | '.join(header), 180)}"
            )

        return (
            f"- {path.name}: 文本说明文件，首段内容概览为“{_short_preview(' '.join(header), 180)}”。"
            f"{body and f'后续摘录：{body}' or ''}"
        )

    def build_sample_summary(self, raw_paths: list[str] | None = None) -> dict[str, Any]:
        if raw_paths:
            selected = [self._resolve_path(path) for path in raw_paths]
        else:
            selected = [Path(path) for path in self.list_text_files(limit=3)]

        summaries: list[str] = []
        details: list[dict[str, Any]] = []
        for path in selected:
            text = _read_text_with_fallback(path)
            summaries.append(self._summarize_text(path, text))
            details.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "preview": _short_preview(text, 300),
                }
            )

        report_path = self.output_dir / "sample_task_summary.md"
        report_lines = [
            "# 文本文档总结",
            "",
            f"- 生成时间: {_now_iso()}",
            f"- 分析文件数: {len(selected)}",
            f"- 工作目录: {self.workspace_root}",
            "",
            "## 文件摘要",
            *summaries,
            "",
            "## 处理说明",
            "本次示例任务由本地 agent 直接读取 workspace 中的文本文件并写出总结文件。",
        ]
        report_text = "\n".join(report_lines) + "\n"
        report_path.write_text(report_text, encoding="utf-8")
        self.memory["last_summary_path"] = str(report_path)
        self.memory["last_summary_files"] = [str(path) for path in selected]
        self.memory["last_action"] = "build_sample_summary"
        self._save_state()
        return {"path": str(report_path), "text": report_text, "details": details}

    def handle_message(self, message: str) -> dict[str, Any]:
        user_message = message.strip()
        self.history.append({"role": "user", "content": user_message, "time": _now_iso()})
        lower = user_message.lower()
        reply = ""
        action: dict[str, Any] = {}

        if self.memory.get("last_summary_path") and len(user_message) <= 4 and not any(
            token in lower for token in ("txt", "summary", "list", "read", "write", "总结", "归纳", "汇总", "列出", "清单")
        ):
            last_path = self.memory.get("last_summary_path")
            if last_path:
                reply = f"我记得上一次输出在 {last_path}。你可以让我继续扩展它，或者让我处理新的文本文件。"
            else:
                reply = "我还没有生成过总结文件。你可以让我先跑一次示例任务。"
        elif any(word in user_message for word in ("总结", "归纳", "汇总")) or "summary" in lower:
            if any(word in user_message for word in ("文本文档", "txt", "文本")):
                action = self.build_sample_summary()
                reply = f"已读取 3 个文本文件，并写入总结文件：{action['path']}"
            else:
                reply = "我可以先帮你列出 workspace 里的文本文件，或者直接生成总结文件。"
        elif any(word in user_message for word in ("列出", "清单", "list")):
            files = self.list_text_files(limit=8)
            reply = "可用的文本文件如下：\n" + "\n".join(f"- {path}" for path in files) if files else "当前 workspace 没有找到可用的 .txt 文件。"
            action = {"files": files}
        else:
            path_match = re.search(r"([A-Za-z]:[\\/][^\n]+?\.txt|[^\n]+?\.txt)", user_message)
            if path_match:
                try:
                    info = self.read_file(path_match.group(1))
                    reply = f"已读取 {info['path']}，前几行概览：{_short_preview(info['content'], 260)}"
                    action = {"read": info["path"]}
                    self.memory["last_read_path"] = info["path"]
                except Exception as exc:  # noqa: BLE001
                    reply = f"读取失败：{exc}"
            else:
                reply = (
                    "我可以读取和写入 workspace 内的文件，也能维护对话上下文。"
                    " 你可以直接让我列出文本文件、读取某个 .txt，或者生成总结文件。"
                )

        self.history.append({"role": "assistant", "content": reply, "time": _now_iso()})
        self.memory["last_message"] = user_message
        if action:
            self.memory["last_action"] = action.get("path") or action.get("read") or action.get("files") or "chat"
        self._save_state()
        return {"reply": reply, "history": self.history[-40:], "memory": self.memory, "action": action}
