from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .backup import StateDirectoryLock
from .document_pipeline import (
    DocumentNeedsReviewError,
    ParentChildChunker,
    QualityGatedDocumentParser,
)
from .embeddings import EmbeddingProvider, HashEmbedding, OpenAIEmbedding
from .graph import AgentGraph
from .ingestion_jobs import IngestionJobManager
from .llm import ModelGateway
from .memory import MemoryStore
from .observability import Observability, current_trace_id
from .rag import RAGService
from .rag_registry import SQLiteVersionRegistry
from .schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentIngestRequest,
    DocumentIngestResponse,
    IngestionJobResponse,
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
    def __init__(self, config: Settings, observability: Observability | None = None) -> None:
        self.settings = config
        config.resolved_state_dir.mkdir(parents=True, exist_ok=True)
        self.state_lock = StateDirectoryLock(config.resolved_state_dir)
        self.state_lock.acquire()
        try:
            self.observability = observability or Observability(
                service_name="agentforge-service",
                service_version="0.3.0",
                environment=config.environment,
                state_dir=config.resolved_state_dir,
                metrics_enabled=config.metrics_enabled,
                trace_jsonl_enabled=config.trace_jsonl_enabled,
                trace_jsonl_max_bytes=config.trace_jsonl_max_bytes,
                trace_jsonl_backup_count=config.trace_jsonl_backup_count,
                otlp_endpoint=config.otel_exporter_otlp_endpoint,
            )
            if config.embedding_provider == "openai" and config.api_key.get_secret_value():
                embedding_client = AsyncOpenAI(
                    api_key=config.api_key.get_secret_value(),
                    base_url=config.base_url,
                    max_retries=0,
                )
                self.embeddings: EmbeddingProvider = OpenAIEmbedding(
                    embedding_client, config.embedding_model
                )
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
            parser = QualityGatedDocumentParser(
                prefer_docling=True,
                enable_ocr_fallback=config.rag_ocr_enabled,
                max_attempts=config.rag_parser_max_attempts,
            )
            chunker = ParentChildChunker(
                target_tokens=config.rag_chunk_target_tokens,
                max_tokens=config.rag_chunk_max_tokens,
                overlap_tokens=config.rag_chunk_overlap_tokens,
            )
            self.rag = RAGService(
                config.workspace_root,
                parser,
                chunker,
                self.embeddings,
                self.vector_store,
                SQLiteVersionRegistry(config.resolved_state_dir / "rag_registry.sqlite3"),
                max_corrective_rounds=config.rag_corrective_max_rounds,
                query_max_parallel=config.rag_query_max_parallel,
                query_timeout_seconds=config.rag_query_timeout_seconds,
                query_min_relevance_score=config.rag_query_min_relevance_score,
                query_rrf_k=config.rag_query_rrf_k,
                observability=self.observability,
            )
            self.ingestion_jobs = IngestionJobManager(
                config.resolved_state_dir / "rag_ingestion_jobs.sqlite3",
                self.rag,
                max_parallel=config.rag_ingestion_parallelism,
                observability=self.observability,
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

        except Exception:
            self.state_lock.release()
            raise

    async def initialize(self) -> None:
        await self.rag.initialize()
        await self.memory.initialize()
        await self.ingestion_jobs.initialize()

    async def close(self) -> None:
        try:
            await self.ingestion_jobs.close()
            await self.observability.shutdown()
        finally:
            self.state_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    services = Services(settings)
    try:
        await services.initialize()
        app.state.services = services
        yield
    finally:
        await services.close()


def create_app(config: Settings = settings) -> FastAPI:
    app_observability = Observability(
        service_name="agentforge-service",
        service_version="0.3.0",
        environment=config.environment,
        state_dir=config.resolved_state_dir,
        metrics_enabled=config.metrics_enabled,
        trace_jsonl_enabled=config.trace_jsonl_enabled,
        trace_jsonl_max_bytes=config.trace_jsonl_max_bytes,
        trace_jsonl_backup_count=config.trace_jsonl_backup_count,
        otlp_endpoint=config.otel_exporter_otlp_endpoint,
    )

    @asynccontextmanager
    async def configured_lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = Services(config, observability=app_observability)
        try:
            await services.initialize()
            app.state.services = services
            yield
        finally:
            await services.close()

    app = FastAPI(title=config.app_name, version="0.3.0", lifespan=configured_lifespan)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=app_observability.tracer_provider,
        excluded_urls="/metrics",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origin_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_and_trace_header(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            trace_id = current_trace_id()
            if trace_id is not None:
                response.headers["X-Trace-ID"] = trace_id
            return cast(Response, response)
        finally:
            if request.url.path != "/metrics" and hasattr(app.state, "services"):
                route = request.scope.get("route")
                route_path = getattr(route, "path", "unmatched")
                app.state.services.observability.record_http(
                    request.method, route_path, status, time.perf_counter() - started
                )

    def services() -> Services:
        return cast(Services, app.state.services)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        content, content_type = services().observability.render_metrics()
        return Response(content=content, headers={"Content-Type": content_type})

    @app.get("/ready")
    async def readiness() -> dict[str, Any]:
        return {
            "status": "ready",
            "state_lock": "held",
            "vector_backend": config.vector_backend,
            "metrics": config.metrics_enabled,
            "trace_jsonl": config.trace_jsonl_enabled,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        svc = services()
        return {
            "status": "ok",
            "version": "0.3.0",
            "model_routes": [route.name for route in svc.gateway.routes],
            "online_model": svc.gateway.online,
            "memory": "sqlite-wal",
            "vector_backend": config.vector_backend,
            "metrics": config.metrics_enabled,
            "trace_jsonl": config.trace_jsonl_enabled,
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
            "rag_ocr_enabled": config.rag_ocr_enabled,
            "rag_parser_max_attempts": config.rag_parser_max_attempts,
            "rag_ingestion_parallelism": config.rag_ingestion_parallelism,
            "metrics_enabled": config.metrics_enabled,
            "trace_jsonl_enabled": config.trace_jsonl_enabled,
            "trace_jsonl_max_bytes": config.trace_jsonl_max_bytes,
            "trace_jsonl_backup_count": config.trace_jsonl_backup_count,
            "backup_freshness_seconds": config.backup_freshness_seconds,
            "deprecated_fields": ["POST /documents/ingest"],
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
            raise HTTPException(
                status_code=500, detail=f"agent execution failed: {type(exc).__name__}"
            ) from exc

        reply = result["final_reply"] or "任务执行结束，但模型未返回可展示文本。"
        citations = [Citation.model_validate(item) for item in result["citations"]]
        reply = _ensure_citation_footer(reply, citations)
        await svc.memory.append_messages(
            namespace,
            [
                ChatMessage(role="user", content=request.message),
                ChatMessage(role="assistant", content=reply),
            ],
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

    @app.post("/documents/ingest", response_model=DocumentIngestResponse, deprecated=True)
    async def ingest_document(
        request: DocumentIngestRequest, response: Response
    ) -> DocumentIngestResponse:
        response.headers["Deprecation"] = "true"
        response.headers["X-AgentForge-Migration"] = (
            "Use POST /rag/ingestions; removal is planned for v1.0."
        )
        try:
            return await services().rag.ingest(request)
        except DocumentNeedsReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/rag/ingestions", response_model=IngestionJobResponse, status_code=202)
    async def create_ingestion_job(
        request: DocumentIngestRequest, response: Response
    ) -> IngestionJobResponse:
        job = await services().ingestion_jobs.create(request)
        response.headers["Location"] = f"/rag/ingestions/{job.job_id}"
        return job

    @app.get("/rag/ingestions/{job_id}", response_model=IngestionJobResponse)
    async def get_ingestion_job(job_id: str) -> IngestionJobResponse:
        try:
            return await services().ingestion_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="ingestion job not found") from exc

    @app.post("/rag/ingestions/{job_id}/cancel", response_model=IngestionJobResponse)
    async def cancel_ingestion_job(job_id: str) -> IngestionJobResponse:
        try:
            return await services().ingestion_jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="ingestion job not found") from exc

    @app.post("/documents/search", response_model=RetrievalResponse)
    async def search_documents(request: RetrievalRequest) -> RetrievalResponse:
        return await services().rag.retrieve(
            request.query,
            request.top_k,
            request.source_ids,
            request.tenant_id,
            request.principals,
        )

    @app.post("/memory", response_model=MemoryRecord)
    async def write_memory(request: MemoryWrite) -> MemoryRecord:
        return await services().memory.remember(request)

    return app


def _ensure_citation_footer(reply: str, citations: list[Citation]) -> str:
    if not citations:
        return reply
    footer = []
    for citation in citations:
        locator = (
            f"p.{citation.page}"
            if citation.page
            else citation.locator or citation.chunk_id or "source"
        )
        footer.append(f"- [{citation.source_name} — {locator}]")
    heading = "\n\nSources:\n"
    if "Sources:" in reply or "来源：" in reply:
        return reply
    return reply.rstrip() + heading + "\n".join(footer)
