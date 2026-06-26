from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)
    load_dotenv(override=False)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_base_url(value: str, default: str) -> str:
    cleaned = value.strip().rstrip("/")
    if not cleaned:
        return default
    if "deepseek.com" in cleaned:
        return "https://api.deepseek.com"
    return cleaned


@dataclass(slots=True)
class Settings:
    app_name: str = "AI Agent Backend Service"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int | None = None
    llm_timeout: int = 60

    demo_file_name: str = "test.txt"
    demo_file_content: str = (
        "This is a demo file for AI Agent testing.\n"
        "It is used to verify file reading and summary generation.\n"
        "You can also replace it with your own content.\n"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        workspace_root = Path(os.getenv("AI_AGENT_WORKSPACE_ROOT", os.getcwd())).resolve()
        llm_max_tokens_raw = os.getenv("AI_AGENT_LLM_MAX_TOKENS", "").strip()
        max_tokens = int(llm_max_tokens_raw) if llm_max_tokens_raw else None

        llm_base_url = _normalize_base_url(
            os.getenv(
                "AI_AGENT_BASE_URL",
                os.getenv("DEEPSEEK_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            ),
            "https://api.openai.com/v1",
        )

        explicit_model = os.getenv("AI_AGENT_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL")
        default_model = "deepseek-v4-flash" if "deepseek.com" in llm_base_url else "gpt-4o-mini"

        return cls(
            debug=_truthy(os.getenv("DEBUG"), False),
            host=os.getenv("AI_AGENT_HOST", "0.0.0.0"),
            port=int(os.getenv("AI_AGENT_PORT", "8000")),
            workspace_root=workspace_root,
            cors_origins=_csv(
                os.getenv("AI_AGENT_CORS_ORIGINS"),
                ["http://localhost:3000", "http://127.0.0.1:3000"],
            ),
            llm_base_url=llm_base_url,
            llm_api_key=os.getenv(
                "AI_AGENT_API_KEY",
                os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            ),
            llm_model=explicit_model or default_model,
            llm_temperature=float(os.getenv("AI_AGENT_TEMPERATURE", "0.2")),
            llm_max_tokens=max_tokens,
            llm_timeout=int(os.getenv("AI_AGENT_TIMEOUT", "60")),
        )


settings = Settings.from_env()
