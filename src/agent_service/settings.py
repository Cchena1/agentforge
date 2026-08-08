from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoute(BaseModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""))
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_attempts: int = Field(default=3, ge=1, le=8)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_AGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AgentForge"
    debug: bool = Field(
        default=False, validation_alias=AliasChoices("AI_AGENT_DEBUG", "DEBUG")
    )
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workspace_root: Path = Field(default_factory=Path.cwd)
    state_dir: Path = Field(default=Path("state"))
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = Field(default=SecretStr(""))
    fallback_routes_json: str = "[]"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)

    max_agent_steps: int = Field(default=8, ge=1, le=64)
    max_repeated_tool_calls: int = Field(default=2, ge=1, le=8)
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_parallel_tools: int = Field(default=8, ge=1, le=64)
    short_term_message_limit: int = Field(default=24, ge=4, le=200)
    short_term_char_budget: int = Field(default=24000, ge=2000)
    rag_top_k: int = Field(default=6, ge=1, le=30)
    embedding_provider: Literal["hash", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    vector_backend: Literal["sqlite", "qdrant"] = "sqlite"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_documents"

    @field_validator("workspace_root", mode="after")
    @classmethod
    def resolve_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def resolved_state_dir(self) -> Path:
        value = self.state_dir
        if not value.is_absolute():
            value = self.workspace_root / value
        return value.resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def model_routes(self) -> list[ModelRoute]:
        routes = [
            ModelRoute(
                name="primary",
                model=self.model,
                base_url=self.base_url.rstrip("/"),
                api_key=self.api_key,
            )
        ]
        try:
            raw = json.loads(self.fallback_routes_json or "[]")
            routes.extend(ModelRoute.model_validate(item) for item in raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("AI_AGENT_FALLBACK_ROUTES_JSON must be a JSON array of model routes") from exc
        return routes


settings = Settings()
