from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from .document_models import CANONICAL_DOCUMENT_SCHEMA_VERSION, Chunk, ParsedDocument
from .documents import resolve_workspace_file
from .embeddings import EmbeddingProvider
from .rag_registry import InMemoryVersionRegistry, VersionRegistry
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


class RAGService:
    def __init__(self, workspace_root: Path, parser: DocumentParserLike, chunker: ChunkerLike,
                 embeddings: EmbeddingProvider, store: VectorStore,
                 version_registry: VersionRegistry | None = None, *, max_corrective_rounds: int = 2) -> None:
        if not 0 <= max_corrective_rounds <= 2:
            raise ValueError("max_corrective_rounds must be between 0 and 2")
        self.workspace_root = workspace_root
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store
        self.version_registry = version_registry or InMemoryVersionRegistry()
        self.max_corrective_rounds = max_corrective_rounds
        self._source_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.version_registry.initialize()

    async def ingest(self, request: DocumentIngestRequest, *, progress: ProgressCallback | None = None) -> DocumentIngestResponse:
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
        tenant_active = await self.version_registry.list_active(source_ids, tenant_id)
        principal_set = set(principals or [])
        authorized_active = {source_id: item for source_id, item in tenant_active.items()
                             if not item.acl or principal_set.intersection(item.acl)}
        active_versions = {source_id: item.version_id for source_id, item in authorized_active.items()}
        warnings: list[str] = []
        total_latency = 0.0
        hits: list[RetrievalHit] = []
        candidate_hits: list[RetrievalHit] = []
        queries = [query, _broaden_query(query)]
        rounds = min(self.max_corrective_rounds + 1, len(queries))
        for round_index in range(rounds):
            candidate_hits, latency_ms = await self.store.search(queries[round_index], top_k=max(top_k * 2, top_k),
                source_ids=source_ids, active_versions=active_versions,
                legacy_excluded_source_ids=set(tenant_active), tenant_id=tenant_id, principals=principals)
            total_latency += latency_ms
            consume_diagnostics = getattr(self.store, "consume_diagnostics", None)
            if callable(consume_diagnostics):
                warnings.extend(consume_diagnostics())
            hits = _verified_hits(candidate_hits, active_versions)
            if len(hits) >= top_k or queries[round_index] == queries[-1]:
                break
            warnings.append(f"corrective_retrieval_round:{round_index + 1}")
        if len(hits) < len(candidate_hits):
            warnings.append("invalid_citations_dropped")
        return RetrievalResponse(hits=hits[:top_k], latency_ms=total_latency,
            index_versions=active_versions, warnings=warnings, degraded_retrieval=bool(warnings))

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


def _broaden_query(query: str) -> str:
    terms = re.findall(r"[\w\-]+", query.lower(), flags=re.UNICODE)
    return " ".join(dict.fromkeys(terms)) or query


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
