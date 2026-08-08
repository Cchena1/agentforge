from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from .documents import CompositeDocumentParser, SemanticChunker
from .embeddings import EmbeddingProvider, HashEmbedding, OpenAIEmbedding
from .graph import AgentGraph
from .llm import ModelGateway
from .memory import MemoryStore
from .rag import RAGService
from .schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentIngestRequest,
    DocumentIngestResponse,
    MemoryRecord,
    MemoryWrite,
    RetrievalRequest,
    RetrievalResponse,
    ToolCallRecord,
)
from .settings import Settings, settings
from .tools import AsyncToolExecutor, build_registry
from .vector_store import QdrantVectorStore, SQLiteVectorStore, VectorStore

logger = logging.getLogger(__name__)


class Services:
    def __init__(self, config: Settings) -> None:
        self.settings = config
        config.resolved_state_dir.mkdir(parents=True, exist_ok=True)
        if config.embedding_provider == "openai" and config.api_key.get_secret_value():
            embedding_client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                max_retries=0,
            )
            self.embeddings: EmbeddingProvider = OpenAIEmbedding(embedding_client, config.embedding_model)
        else:
            self.embeddings = HashEmbedding()
        self.vector_store: VectorStore
        if config.vector_backend == "qdrant":
            self.vector_store = QdrantVectorStore(
                config.qdrant_url, config.qdrant_collection, self.embeddings
            )
        else:
            self.vector_store = SQLiteVectorStore(
                config.resolved_state_dir / "rag.sqlite3", self.embeddings
            )
        self.rag = RAGService(
            config.workspace_root,
            CompositeDocumentParser(prefer_docling=True),
            SemanticChunker(),
            self.embeddings,
            self.vector_store,
        )
        self.memory = MemoryStore(
            config.resolved_state_dir / "memory.sqlite3",
            self.embeddings,
            message_limit=config.short_term_message_limit,
            char_budget=config.short_term_char_budget,
        )
        self.gateway = ModelGateway(config)
        self.registry = build_registry(config.workspace_root, self.rag)
        self.executor = AsyncToolExecutor(
            self.registry,
            timeout_seconds=config.tool_timeout_seconds,
            max_parallel=config.max_parallel_tools,
        )
        self.graph = AgentGraph(
            self.gateway,
            self.registry,
            self.executor,
            max_steps=config.max_agent_steps,
            max_repeated_calls=config.max_repeated_tool_calls,
        )

    async def initialize(self) -> None:
        await self.vector_store.initialize()
        await self.memory.initialize()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    services = Services(settings)
    await services.initialize()
    app.state.services = services
    yield


def create_app(config: Settings = settings) -> FastAPI:
    @asynccontextmanager
    async def configured_lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = Services(config)
        await services.initialize()
        app.state.services = services
        yield

    app = FastAPI(title=config.app_name, version="0.2.0", lifespan=configured_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origin_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def services() -> Services:
        return cast(Services, app.state.services)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        svc = services()
        return {
            "status": "ok",
            "version": "0.2.0",
            "model_routes": [route.name for route in svc.gateway.routes],
            "online_model": svc.gateway.online,
            "memory": "sqlite-wal",
            "vector_backend": config.vector_backend,
        }

    @app.get("/config")
    async def public_config() -> dict[str, Any]:
        return {
            "app_name": config.app_name,
            "workspace_root": str(config.workspace_root),
            "model": config.model,
            "base_url": config.base_url,
            "api_key_configured": bool(config.api_key.get_secret_value()),
            "max_agent_steps": config.max_agent_steps,
            "embedding_provider": config.embedding_provider,
            "vector_backend": config.vector_backend,
            "deprecated_fields": [],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        svc = services()
        session_id = request.session_id or str(uuid.uuid4())
        namespace = svc.memory.namespace(request.user_id, session_id)
        summary, stored_messages = await svc.memory.load_context(namespace)
        history = request.history if request.history is not None else stored_messages
        messages = list(history)
        if not messages or messages[-1].role != "user" or messages[-1].content != request.message:
            messages.append(ChatMessage(role="user", content=request.message))
        long_term = await svc.memory.recall(f"user:{request.user_id}", request.message, top_k=4)
        try:
            result = await svc.graph.run(
                messages,
                memory_summary=summary,
                long_term_memories=[item.text for item in long_term],
            )
        except Exception as exc:
            logger.exception("Agent graph failed")
            raise HTTPException(status_code=500, detail=f"agent execution failed: {type(exc).__name__}") from exc

        reply = result["final_reply"] or "任务执行结束，但模型未返回可展示文本。"
        citations = [Citation.model_validate(item) for item in result["citations"]]
        reply = _ensure_citation_footer(reply, citations)
        await svc.memory.append_messages(
            namespace,
            [ChatMessage(role="user", content=request.message), ChatMessage(role="assistant", content=reply)],
        )
        return ChatResponse(
            reply=reply,
            tool_calls=[ToolCallRecord.model_validate(item) for item in result["tool_records"]],
            citations=citations,
            session_id=session_id,
            degraded=result["degraded"],
            handoff_required=result["handoff_required"],
            warnings=result["warnings"],
        )

    @app.post("/documents/ingest", response_model=DocumentIngestResponse)
    async def ingest_document(request: DocumentIngestRequest) -> DocumentIngestResponse:
        try:
            return await services().rag.ingest(request)
        except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/documents/search", response_model=RetrievalResponse)
    async def search_documents(request: RetrievalRequest) -> RetrievalResponse:
        return await services().rag.retrieve(request.query, request.top_k, request.source_ids)

    @app.post("/memory", response_model=MemoryRecord)
    async def write_memory(request: MemoryWrite) -> MemoryRecord:
        return await services().memory.remember(request)

    return app


def _ensure_citation_footer(reply: str, citations: list[Citation]) -> str:
    if not citations:
        return reply
    footer = []
    for citation in citations:
        locator = f"p.{citation.page}" if citation.page else citation.locator or citation.chunk_id or "source"
        footer.append(f"- [{citation.source_name} — {locator}]")
    heading = "\n\nSources:\n"
    if "Sources:" in reply or "来源：" in reply:
        return reply
    return reply.rstrip() + heading + "\n".join(footer)
