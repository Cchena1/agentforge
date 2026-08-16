from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .document_models import CANONICAL_DOCUMENT_SCHEMA_VERSION, Chunk, ParsedDocument
from .documents import resolve_workspace_file
from .embeddings import EmbeddingProvider
from .observability import Observability
from .query_planning import QueryVariant, build_query_plan
from .rag_registry import ActiveDocumentVersion, InMemoryVersionRegistry, VersionRegistry
from .schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    IngestionJobStatus,
    RetrievalHit,
    RetrievalResponse,
)
from .vector_store import VectorDocument, VectorStore

ProgressCallback = Callable[[IngestionJobStatus], Awaitable[None]]


class DocumentParserLike(Protocol):
    @property
    def profile_id(self) -> str: ...
    async def parse(self, path: Path) -> ParsedDocument: ...


class ChunkerLike(Protocol):
    @property
    def profile_id(self) -> str: ...
    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class _QuerySearchResult:
    variant: QueryVariant
    hits: list[RetrievalHit]
    warnings: list[str]


class RAGService:
    def __init__(self, workspace_root: Path, parser: DocumentParserLike, chunker: ChunkerLike,
                 embeddings: EmbeddingProvider, store: VectorStore,
                 version_registry: VersionRegistry | None = None, *, max_corrective_rounds: int = 2,
                 query_max_parallel: int = 2, query_timeout_seconds: float = 10.0,
                 query_min_relevance_score: float = 0.2, query_rrf_k: int = 60,
                 observability: Observability | None = None) -> None:
        if not 0 <= max_corrective_rounds <= 2:
            raise ValueError("max_corrective_rounds must be between 0 and 2")
        if not 1 <= query_max_parallel <= 4:
            raise ValueError("query_max_parallel must be between 1 and 4")
        if not 0 < query_timeout_seconds <= 60:
            raise ValueError("query_timeout_seconds must be between 0 and 60")
        if not 0 <= query_min_relevance_score <= 1:
            raise ValueError("query_min_relevance_score must be between 0 and 1")
        if not 1 <= query_rrf_k <= 1000:
            raise ValueError("query_rrf_k must be between 1 and 1000")
        self.workspace_root = workspace_root
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store
        self.version_registry = version_registry or InMemoryVersionRegistry()
        self.max_corrective_rounds = max_corrective_rounds
        self.query_max_parallel = query_max_parallel
        self.query_timeout_seconds = query_timeout_seconds
        self.query_min_relevance_score = query_min_relevance_score
        self.query_rrf_k = query_rrf_k
        self.observability = observability
        self._source_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.version_registry.initialize()

    async def ingest(self, request: DocumentIngestRequest, *, progress: ProgressCallback | None = None) -> DocumentIngestResponse:
        started = time.perf_counter()
        observer = self.observability
        try:
            if observer is None:
                result = await self._ingest(request, progress)
            else:
                with observer.span("rag.ingest", {"rag.operation": "ingest"}):
                    result = await self._ingest(request, progress)
        except Exception:
            if observer is not None:
                observer.record_rag("ingest", "failure", time.perf_counter() - started)
            raise
        if observer is not None:
            observer.record_rag("ingest", "success", time.perf_counter() - started)
        return result

    async def _ingest(self, request: DocumentIngestRequest, progress: ProgressCallback | None) -> DocumentIngestResponse:
        path = resolve_workspace_file(self.workspace_root, request.file_path)
        source_id = request.source_id or hashlib.sha256(str(path).encode()).hexdigest()[:24]
        lock_key = f"{request.tenant_id}\0{source_id}"
        source_lock = self._source_locks.setdefault(lock_key, asyncio.Lock())
        async with source_lock:
            return await self._ingest_locked(request, path, source_id, progress)

    async def _ingest_locked(self, request: DocumentIngestRequest, path: Path, source_id: str,
                             progress: ProgressCallback | None) -> DocumentIngestResponse:
        content_sha256 = await _sha256_file(path)
        pipeline_profile = self._pipeline_profile()
        version_id = _version_id(source_id, content_sha256, pipeline_profile, request.tenant_id, request.acl)
        active = await self.version_registry.get_active(source_id, request.tenant_id)
        if active is not None and active.version_id == version_id:
            return DocumentIngestResponse(source_id=source_id, source_name=active.source_name,
                chunks_created=active.chunks_count, parser=active.parser, warnings=list(active.warnings),
                version_id=active.version_id, content_sha256=active.content_sha256, idempotent=True)

        await _notify(progress, IngestionJobStatus.PARSING)
        parsed = await self.parser.parse(path)
        await _notify(progress, IngestionJobStatus.QUALITY_CHECK)
        quality_report = getattr(parsed, "quality_report", None)
        if quality_report is not None and not quality_report.accepted:
            raise RuntimeError("parser output failed the deterministic quality gate")
        if len(getattr(parsed, "attempts", [])) > 1:
            await _notify(progress, IngestionJobStatus.FALLBACK)
        if await _sha256_file(path) != content_sha256:
            raise RuntimeError("source file changed during ingestion; retry with a stable file")

        await _notify(progress, IngestionJobStatus.CHUNKING)
        chunks = self.chunker.chunk(parsed)
        if not chunks:
            raise ValueError("document produced no indexable chunks")
        await _notify(progress, IngestionJobStatus.EMBEDDING)
        vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        if any(len(vector) != self.embeddings.dimension for vector in vectors):
            raise RuntimeError("embedding dimension does not match the configured profile")

        warnings = [*parsed.warnings]
        if quality_report is not None:
            warnings.extend(issue for issue in quality_report.issues if issue not in warnings)
        await self.version_registry.record_building(source_id=source_id, version_id=version_id,
            source_name=parsed.source_name, content_sha256=content_sha256, pipeline_profile=pipeline_profile,
            parser=parsed.parser, chunks_count=len(chunks), warnings=warnings,
            tenant_id=request.tenant_id, acl=request.acl)
        await _notify(progress, IngestionJobStatus.INDEXING)
        try:
            await self.store.upsert([
                VectorDocument(chunk_id=f"{version_id}:{chunk.chunk_id}", source_id=source_id,
                    source_name=parsed.source_name, version_id=version_id, tenant_id=request.tenant_id,
                    acl=tuple(request.acl), text=chunk.text, embedding=embedding, page=chunk.page,
                    locator=chunk.locator, metadata={**request.metadata, **chunk.metadata,
                        "content_sha256": content_sha256, "pipeline_profile": pipeline_profile,
                        "document_schema_version": parsed.schema_version,
                        "parser": parsed.parser, "parent_id": getattr(chunk, "parent_id", None),
                        "location": getattr(chunk, "location", None),
                        "parse_quality": quality_report.score if quality_report else None})
                for chunk, embedding in zip(chunks, vectors, strict=True)])
            await _notify(progress, IngestionJobStatus.VALIDATING)
            activated = await self.version_registry.activate(version_id)
        except Exception as exc:
            await self.version_registry.mark_failed(version_id, f"{type(exc).__name__}: {exc}")
            raise
        return DocumentIngestResponse(source_id=source_id, source_name=parsed.source_name,
            chunks_created=len(chunks), parser=parsed.parser, warnings=warnings,
            version_id=activated.version_id, content_sha256=content_sha256, idempotent=False)

    async def retrieve(self, query: str, top_k: int, source_ids: list[str] | None = None,
                       tenant_id: str = "public", principals: list[str] | None = None) -> RetrievalResponse:
        started = time.perf_counter()
        observer = self.observability
        try:
            if observer is None:
                response = await self._retrieve(query, top_k, source_ids, tenant_id, principals, started)
            else:
                with observer.span("rag.retrieve", {
                    "rag.operation": "retrieve", "rag.top_k": top_k,
                    "rag.source_filter_count": len(source_ids or []),
                }):
                    response = await self._retrieve(
                        query, top_k, source_ids, tenant_id, principals, started
                    )
        except Exception:
            if observer is not None:
                observer.record_rag("retrieve", "failure", time.perf_counter() - started)
            raise
        if observer is not None:
            observer.record_rag(
                "retrieve", "success", time.perf_counter() - started,
                degraded=response.degraded_retrieval,
            )
        return response

    async def _retrieve(self, query: str, top_k: int, source_ids: list[str] | None,
                        tenant_id: str, principals: list[str] | None,
                        started: float) -> RetrievalResponse:
        tenant_active = await self.version_registry.list_active(source_ids, tenant_id)
        principal_set = set(principals or [])
        authorized_active = {source_id: item for source_id, item in tenant_active.items()
                             if not item.acl or principal_set.intersection(item.acl)}
        active_versions = {source_id: item.version_id for source_id, item in authorized_active.items()}
        plan = build_query_plan(query, max_variants=self.max_corrective_rounds + 1)
        warnings = list(plan.warnings)
        semaphore = asyncio.Semaphore(self.query_max_parallel)
        subqueries = [variant for variant in plan.variants[1:] if variant.kind == "subquery"]
        subqueries = subqueries[:self.max_corrective_rounds]
        initial_variants = [plan.variants[0], *subqueries]
        results = await self._search_variants(initial_variants, top_k, source_ids, tenant_active,
            active_versions, tenant_id, principals, semaphore)
        for result in results:
            warnings.extend(result.warnings)

        initial_hits = [hit for result in results for hit in result.hits]
        remaining_rounds = self.max_corrective_rounds - len(subqueries)
        if remaining_rounds > 0 and not _evidence_sufficient(initial_hits, self.query_min_relevance_score):
            corrective = next((variant for variant in plan.variants[1:]
                if variant.kind == "normalized" and variant not in subqueries), None)
            if corrective is not None:
                warnings.append("corrective_retrieval:triggered")
                corrective_results = await self._search_variants([corrective], top_k, source_ids,
                    tenant_active, active_versions, tenant_id, principals, semaphore)
                results.extend(corrective_results)
                for result in corrective_results:
                    warnings.extend(result.warnings)

        hits = _fuse_query_results(results, self.query_rrf_k)
        if len(results) > 1:
            warnings.append(f"query_fusion:rrf:{len(results)}")
        warnings = list(dict.fromkeys(warnings))
        degraded = any(_is_degradation_warning(item) for item in warnings)
        return RetrievalResponse(hits=hits[:top_k],
            latency_ms=(time.perf_counter() - started) * 1000, index_versions=active_versions,
            warnings=warnings, degraded_retrieval=degraded)

    async def _search_variants(self, variants: list[QueryVariant], top_k: int,
                               source_ids: list[str] | None, tenant_active: dict[str, ActiveDocumentVersion],
                               active_versions: dict[str, str], tenant_id: str,
                               principals: list[str] | None,
                               semaphore: asyncio.Semaphore) -> list[_QuerySearchResult]:
        tasks: list[asyncio.Task[_QuerySearchResult]] = []
        async with asyncio.TaskGroup() as task_group:
            for variant in variants:
                tasks.append(task_group.create_task(self._search_variant(variant, top_k, source_ids,
                    tenant_active, active_versions, tenant_id, principals, semaphore)))
        return [task.result() for task in tasks]

    async def _search_variant(self, variant: QueryVariant, top_k: int,
                              source_ids: list[str] | None, tenant_active: dict[str, ActiveDocumentVersion],
                              active_versions: dict[str, str], tenant_id: str,
                              principals: list[str] | None,
                              semaphore: asyncio.Semaphore) -> _QuerySearchResult:
        warnings: list[str] = []
        try:
            async with asyncio.timeout(self.query_timeout_seconds):
                async with semaphore:
                    candidate_hits, _ = await self.store.search(variant.query,
                        top_k=max(top_k * 2, top_k), source_ids=source_ids,
                        active_versions=active_versions,
                        legacy_excluded_source_ids=set(tenant_active), tenant_id=tenant_id,
                        principals=principals)
                    consume_diagnostics = getattr(self.store, "consume_diagnostics", None)
                    if callable(consume_diagnostics):
                        warnings.extend(consume_diagnostics())
        except TimeoutError:
            warnings.append(f"query_search_timeout:{variant.kind}")
            return _QuerySearchResult(variant=variant, hits=[], warnings=warnings)
        except Exception as exc:  # noqa: BLE001 - isolate bounded query branches
            warnings.append(f"query_search_error:{variant.kind}:{type(exc).__name__}")
            return _QuerySearchResult(variant=variant, hits=[], warnings=warnings)
        hits = _verified_hits(candidate_hits, active_versions)
        if len(hits) < len(candidate_hits):
            warnings.append(f"invalid_citations_dropped:{variant.kind}")
        return _QuerySearchResult(variant=variant, hits=hits, warnings=warnings)

    def _pipeline_profile(self) -> str:
        chunker: Any = self.chunker
        chunker_profile = getattr(chunker, "profile_id", type(chunker).__name__)
        if hasattr(chunker, "target_chars"):
            chunker_profile = (
                f"{chunker_profile}:{chunker.target_chars}:"
                f"{chunker.max_chars}:{chunker.overlap_chars}"
            )
        elif hasattr(chunker, "target_tokens"):
            chunker_profile = (
                f"{chunker_profile}:{chunker.target_tokens}:"
                f"{chunker.max_tokens}:{chunker.overlap_tokens}"
            )
        return (
            f"document_schema={CANONICAL_DOCUMENT_SCHEMA_VERSION}|parser={self.parser.profile_id}"
            f"|chunker={chunker_profile}|embedding={self.embeddings.profile_id}"
        )


async def _notify(callback: ProgressCallback | None, status: IngestionJobStatus) -> None:
    if callback is not None:
        await callback(status)


def _verified_hits(hits: list[RetrievalHit], active_versions: dict[str, str]) -> list[RetrievalHit]:
    verified: list[RetrievalHit] = []
    for hit in hits:
        citation = hit.citation
        active_version = active_versions.get(citation.source_id)
        if active_version is not None and (
            citation.chunk_id is None
            or not citation.chunk_id.startswith(f"{active_version}:")
        ):
            continue
        if citation.quote is None or _normalize_quote(citation.quote) not in _normalize_quote(hit.text):
            continue
        verified.append(hit)
    return verified


def _normalize_quote(value: str) -> str:
    return " ".join(value.replace("\u0000", "").split())


def _evidence_sufficient(hits: list[RetrievalHit], minimum_score: float) -> bool:
    return bool(hits) and max(hit.rerank_score for hit in hits) >= minimum_score


def _fuse_query_results(results: list[_QuerySearchResult], rrf_k: int) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    best_hits: dict[str, RetrievalHit] = {}
    best_scores: dict[str, float] = {}
    for result in results:
        for rank, hit in enumerate(result.hits, start=1):
            key = _retrieval_hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in best_hits or hit.rerank_score > best_scores[key]:
                best_hits[key] = hit
                best_scores[key] = hit.rerank_score
    return sorted(best_hits.values(), key=lambda hit: (
        scores[_retrieval_hit_key(hit)], hit.rerank_score), reverse=True)


def _retrieval_hit_key(hit: RetrievalHit) -> str:
    citation = hit.citation
    if citation.chunk_id is not None:
        return citation.chunk_id
    digest = hashlib.sha256(hit.text.encode()).hexdigest()[:16]
    return f"{citation.source_id}:{digest}"


def _is_degradation_warning(warning: str) -> bool:
    return warning.startswith((
        "degraded_retrieval:",
        "invalid_citations_dropped:",
        "query_search_error:",
        "query_search_timeout:",
    ))


async def _sha256_file(path: Path) -> str:
    return await asyncio.to_thread(_sha256_file_sync, path)


def _sha256_file_sync(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_id(source_id: str, content_sha256: str, pipeline_profile: str, tenant_id: str, acl: list[str]) -> str:
    version_profile = json.dumps({"acl": sorted(set(acl)), "content_sha256": content_sha256,
        "pipeline_profile": pipeline_profile, "source_id": source_id, "tenant_id": tenant_id},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(version_profile.encode()).hexdigest()[:24]
    return f"rv_{digest}"
