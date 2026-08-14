from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from .documents import CompositeDocumentParser, SemanticChunker, resolve_workspace_file
from .embeddings import EmbeddingProvider
from .rag_registry import InMemoryVersionRegistry, VersionRegistry
from .schemas import DocumentIngestRequest, DocumentIngestResponse, RetrievalResponse
from .vector_store import VectorDocument, VectorStore


class RAGService:
    def __init__(
        self,
        workspace_root: Path,
        parser: CompositeDocumentParser,
        chunker: SemanticChunker,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        version_registry: VersionRegistry | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store
        self.version_registry = version_registry or InMemoryVersionRegistry()
        # Serialize mutations for one logical source while preserving cross-source concurrency.
        self._source_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.version_registry.initialize()

    async def ingest(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        path = resolve_workspace_file(self.workspace_root, request.file_path)
        source_id = request.source_id or hashlib.sha256(str(path).encode()).hexdigest()[:24]
        lock_key = f"{request.tenant_id}\0{source_id}"
        source_lock = self._source_locks.setdefault(lock_key, asyncio.Lock())
        async with source_lock:
            return await self._ingest_locked(request, path, source_id)

    async def _ingest_locked(
        self, request: DocumentIngestRequest, path: Path, source_id: str
    ) -> DocumentIngestResponse:
        content_sha256 = await _sha256_file(path)
        pipeline_profile = self._pipeline_profile()
        version_id = _version_id(
            source_id, content_sha256, pipeline_profile, request.tenant_id, request.acl
        )
        active = await self.version_registry.get_active(source_id, request.tenant_id)
        if active is not None and active.version_id == version_id:
            return DocumentIngestResponse(
                source_id=source_id,
                source_name=active.source_name,
                chunks_created=active.chunks_count,
                parser=active.parser,
                warnings=list(active.warnings),
                version_id=active.version_id,
                content_sha256=active.content_sha256,
                idempotent=True,
            )

        parsed = await self.parser.parse(path)
        if await _sha256_file(path) != content_sha256:
            raise RuntimeError("source file changed during ingestion; retry with a stable file")
        chunks = self.chunker.chunk(parsed)
        if not chunks:
            raise ValueError("document produced no indexable chunks")
        vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        if any(len(vector) != self.embeddings.dimension for vector in vectors):
            raise RuntimeError("embedding dimension does not match the configured profile")

        await self.version_registry.record_building(
            source_id=source_id,
            version_id=version_id,
            source_name=parsed.source_name,
            content_sha256=content_sha256,
            pipeline_profile=pipeline_profile,
            parser=parsed.parser,
            chunks_count=len(chunks),
            warnings=parsed.warnings,
            tenant_id=request.tenant_id,
            acl=request.acl,
        )
        try:
            await self.store.upsert(
                [
                    VectorDocument(
                        chunk_id=f"{version_id}:{chunk.chunk_id}",
                        source_id=source_id,
                        source_name=parsed.source_name,
                        version_id=version_id,
                        tenant_id=request.tenant_id,
                        acl=tuple(request.acl),
                        text=chunk.text,
                        embedding=embedding,
                        page=chunk.page,
                        locator=chunk.locator,
                        metadata={
                            **request.metadata,
                            **chunk.metadata,
                            "content_sha256": content_sha256,
                            "pipeline_profile": pipeline_profile,
                        },
                    )
                    for chunk, embedding in zip(chunks, vectors, strict=True)
                ]
            )
            activated = await self.version_registry.activate(version_id)
        except Exception as exc:
            await self.version_registry.mark_failed(version_id, f"{type(exc).__name__}: {exc}")
            raise

        return DocumentIngestResponse(
            source_id=source_id,
            source_name=parsed.source_name,
            chunks_created=len(chunks),
            parser=parsed.parser,
            warnings=parsed.warnings,
            version_id=activated.version_id,
            content_sha256=content_sha256,
            idempotent=False,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int,
        source_ids: list[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> RetrievalResponse:
        tenant_active = await self.version_registry.list_active(source_ids, tenant_id)
        principal_set = set(principals or [])
        authorized_active = {
            source_id: item
            for source_id, item in tenant_active.items()
            if not item.acl or principal_set.intersection(item.acl)
        }
        hits, latency_ms = await self.store.search(
            query,
            top_k=top_k,
            source_ids=source_ids,
            active_versions={
                source_id: item.version_id for source_id, item in authorized_active.items()
            },
            legacy_excluded_source_ids=set(tenant_active),
            tenant_id=tenant_id,
            principals=principals,
        )
        return RetrievalResponse(
            hits=hits,
            latency_ms=latency_ms,
            index_versions={
                source_id: item.version_id for source_id, item in authorized_active.items()
            },
        )

    def _pipeline_profile(self) -> str:
        chunker_profile = (
            f"semantic-char-v1:{self.chunker.target_chars}:"
            f"{self.chunker.max_chars}:{self.chunker.overlap_chars}"
        )
        return (
            f"parser={self.parser.profile_id}|{chunker_profile}|"
            f"embedding={self.embeddings.profile_id}"
        )


async def _sha256_file(path: Path) -> str:
    return await asyncio.to_thread(_sha256_file_sync, path)


def _sha256_file_sync(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_id(
    source_id: str,
    content_sha256: str,
    pipeline_profile: str,
    tenant_id: str,
    acl: list[str],
) -> str:
    version_profile = json.dumps(
        {
            "acl": sorted(set(acl)),
            "content_sha256": content_sha256,
            "pipeline_profile": pipeline_profile,
            "source_id": source_id,
            "tenant_id": tenant_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(version_profile.encode()).hexdigest()[:24]
    return f"rv_{digest}"
