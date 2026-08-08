from __future__ import annotations

import hashlib
from pathlib import Path

from .documents import CompositeDocumentParser, SemanticChunker, resolve_workspace_file
from .embeddings import EmbeddingProvider
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
    ) -> None:
        self.workspace_root = workspace_root
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store

    async def ingest(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        path = resolve_workspace_file(self.workspace_root, request.file_path)
        parsed = await self.parser.parse(path)
        source_id = request.source_id or hashlib.sha256(str(path).encode()).hexdigest()[:24]
        chunks = self.chunker.chunk(parsed)
        vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
        await self.store.delete_source(source_id)
        await self.store.upsert(
            [
                VectorDocument(
                    chunk_id=f"{source_id}:{chunk.chunk_id}",
                    source_id=source_id,
                    source_name=parsed.source_name,
                    text=chunk.text,
                    embedding=embedding,
                    page=chunk.page,
                    locator=chunk.locator,
                    metadata={**request.metadata, **chunk.metadata},
                )
                for chunk, embedding in zip(chunks, vectors, strict=True)
            ]
        )
        return DocumentIngestResponse(
            source_id=source_id,
            source_name=parsed.source_name,
            chunks_created=len(chunks),
            parser=parsed.parser,
            warnings=parsed.warnings,
        )

    async def retrieve(self, query: str, top_k: int, source_ids: list[str] | None = None) -> RetrievalResponse:
        hits, latency_ms = await self.store.search(query, top_k=top_k, source_ids=source_ids)
        return RetrievalResponse(hits=hits, latency_ms=latency_ms)
