from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoute(BaseModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""))
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_attempts: int = Field(default=3, ge=1, le=8)
    input_cost_per_million_tokens: float = Field(default=0.0, ge=0)
    output_cost_per_million_tokens: float = Field(default=0.0, ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_AGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AgentForge"
    debug: bool = Field(default=False, validation_alias=AliasChoices("AI_AGENT_DEBUG", "DEBUG"))
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workspace_root: Path = Field(default_factory=Path.cwd)
    state_dir: Path = Field(default=Path("state"))
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    environment: str = Field(default="local", min_length=1, max_length=64)
    metrics_enabled: bool = True
    trace_jsonl_enabled: bool = True
    trace_jsonl_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1_048_576, le=1_073_741_824)
    trace_jsonl_backup_count: int = Field(default=5, ge=1, le=20)
    otel_exporter_otlp_endpoint: str | None = None
    backup_freshness_seconds: int = Field(default=86_400, ge=300, le=2_592_000)

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = Field(default=SecretStr(""))
    fallback_routes_json: str = "[]"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    model_input_cost_per_million_tokens: float = Field(default=0.0, ge=0)
    model_output_cost_per_million_tokens: float = Field(default=0.0, ge=0)

    max_agent_steps: int = Field(default=8, ge=1, le=64)
    max_repeated_tool_calls: int = Field(default=2, ge=1, le=8)
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_parallel_tools: int = Field(default=8, ge=1, le=64)
    tool_write_approval_mode: Literal["deny", "allow"] = "allow"
    tool_result_inline_token_limit: int = Field(default=2500, ge=128, le=100_000)
    tool_result_preview_chars: int = Field(default=4000, ge=256, le=100_000)
    context_token_budget: int = Field(default=64_000, ge=1024, le=2_000_000)
    reserved_output_tokens: int = Field(default=4096, ge=0, le=200_000)
    short_term_message_limit: int = Field(default=24, ge=4, le=200)
    short_term_char_budget: int = Field(default=24000, ge=2000)
    memory_min_relevance_score: float = Field(default=0.35, ge=0, le=1)
    memory_recency_half_life_days: float = Field(default=180.0, gt=0, le=3650)
    rag_top_k: int = Field(default=6, ge=1, le=30)
    rag_corrective_max_rounds: int = Field(default=2, ge=0, le=2)
    rag_query_max_parallel: int = Field(default=2, ge=1, le=4)
    rag_query_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    rag_query_min_relevance_score: float = Field(default=0.2, ge=0, le=1)
    rag_query_rrf_k: int = Field(default=60, ge=1, le=1000)
    rag_ocr_enabled: bool = False
    rag_cloud_fallback_enabled: bool = False
    rag_parser_max_attempts: int = Field(default=3, ge=1, le=3)
    rag_ingestion_parallelism: int = Field(default=2, ge=1, le=16)
    rag_chunk_target_tokens: int = Field(default=500, ge=64, le=4096)
    rag_chunk_max_tokens: int = Field(default=650, ge=128, le=8192)
    rag_chunk_overlap_tokens: int = Field(default=60, ge=0, le=1024)
    embedding_provider: Literal["hash", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    vector_backend: Literal["sqlite", "qdrant"] = "sqlite"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_documents"
    qdrant_api_key: SecretStr = Field(default=SecretStr(""))

    @field_validator("workspace_root", mode="after")
    @classmethod
    def resolve_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def validate_rag_chunk_budget(self) -> Settings:
        if not (
            0
            <= self.rag_chunk_overlap_tokens
            < self.rag_chunk_target_tokens
            <= self.rag_chunk_max_tokens
        ):
            raise ValueError("RAG chunk budgets must satisfy overlap < target <= max tokens")
        if self.rag_cloud_fallback_enabled:
            raise ValueError(
                "Cloud document fallback has no configured provider in v0.4; keep it disabled"
            )
        if self.reserved_output_tokens >= self.context_token_budget:
            raise ValueError("reserved_output_tokens must be smaller than context_token_budget")
        return self

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
                input_cost_per_million_tokens=self.model_input_cost_per_million_tokens,
                output_cost_per_million_tokens=self.model_output_cost_per_million_tokens,
            )
        ]
        try:
            raw = json.loads(self.fallback_routes_json or "[]")
            routes.extend(ModelRoute.model_validate(item) for item in raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                "AI_AGENT_FALLBACK_ROUTES_JSON must be a JSON array of model routes"
            ) from exc
        return routes


settings = Settings()
