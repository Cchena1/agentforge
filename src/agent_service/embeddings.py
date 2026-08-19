from __future__ import annotations

import asyncio
import hashlib
import importlib
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from importlib.util import find_spec
from typing import Any, Protocol, cast

from openai import AsyncOpenAI

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    dimension: int
    profile_id: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


async def embed_documents_compat(
    provider: EmbeddingProvider, texts: Sequence[str]
) -> list[list[float]]:
    """Compatibility window for pre-v0.5 embedding plugins exposing only ``embed``."""

    method = getattr(provider, "embed_documents", None)
    if callable(method):
        typed_method = cast(
            Callable[[Sequence[str]], Awaitable[list[list[float]]]], method
        )
        return await typed_method(texts)
    return await provider.embed(texts)


async def embed_query_compat(provider: EmbeddingProvider, query: str) -> list[float]:
    """Compatibility window for pre-v0.5 embedding plugins exposing only ``embed``."""

    method = getattr(provider, "embed_query", None)
    if callable(method):
        typed_method = cast(Callable[[str], Awaitable[list[float]]], method)
        return await typed_method(query)
    return (await provider.embed([query]))[0]


class HashEmbedding:
    """Deterministic test double. It is not a semantic production embedding provider."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("dimension must be >= 64")
        self.dimension = dimension
        self.profile_id = f"hash-blake2b-test-only-v1:{dimension}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed([query]))[0]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbedding:
    def __init__(self, client: AsyncOpenAI, model: str, dimension: int = 1536) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension
        self.profile_id = f"openai:{model}:{dimension}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimension,
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        self._validate(vectors)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_documents([query])
        return vectors[0]

    def _validate(self, vectors: Sequence[Sequence[float]]) -> None:
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                f"embedding dimension mismatch for {self.profile_id}: expected {self.dimension}"
            )


SentenceTransformerFactory = Callable[[str, str | None], Any]


class SentenceTransformerEmbedding:
    """Lazy local semantic embeddings with query/document asymmetric encoding support."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        *,
        dimension: int = 1024,
        device: str | None = None,
        batch_size: int = 16,
        max_parallel: int = 1,
        model_factory: SentenceTransformerFactory | None = None,
    ) -> None:
        if dimension < 64:
            raise ValueError("dimension must be >= 64")
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if not 1 <= max_parallel <= 8:
            raise ValueError("max_parallel must be between 1 and 8")
        if model_factory is None and find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "sentence-transformers is required for semantic embeddings; "
                "install it with `uv sync --extra semantic`"
            )
        self.model_name = model
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size
        self.profile_id = f"sentence-transformers:{model}:{dimension}"
        self._model_factory = model_factory or _default_sentence_transformer_factory
        self._model: Any | None = None
        self._model_lock = asyncio.Lock()
        self._encode_semaphore = asyncio.Semaphore(max_parallel)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._encode(texts, kind="document")

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self._encode([query], kind="query")
        return vectors[0]

    async def _encode(
        self, texts: Sequence[str], *, kind: str
    ) -> list[list[float]]:
        if not texts:
            return []
        model = await self._load_model()
        async with self._encode_semaphore:
            vectors = await asyncio.to_thread(self._encode_sync, model, list(texts), kind)
        self._validate(vectors)
        return vectors

    async def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                loaded_model = await asyncio.to_thread(
                    self._model_factory, self.model_name, self.device
                )
                actual_dimension = loaded_model.get_sentence_embedding_dimension()
                if actual_dimension is not None and int(actual_dimension) != self.dimension:
                    raise RuntimeError(
                        f"configured embedding dimension {self.dimension} does not match "
                        f"{self.model_name} dimension {actual_dimension}"
                    )
                self._model = loaded_model
        assert self._model is not None
        return self._model

    def _encode_sync(self, model: Any, texts: list[str], kind: str) -> list[list[float]]:
        method_name = "encode_query" if kind == "query" else "encode_document"
        method = getattr(model, method_name, None)
        if not callable(method):
            method = model.encode
        raw = method(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in row] for row in raw]

    def _validate(self, vectors: Sequence[Sequence[float]]) -> None:
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                f"embedding dimension mismatch for {self.profile_id}: expected {self.dimension}"
            )


def _default_sentence_transformer_factory(model: str, device: str | None) -> Any:
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Semantic embedding dependency is missing; run `uv sync --extra semantic`"
        ) from exc
    sentence_transformer = module.SentenceTransformer
    return sentence_transformer(model, device=device)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
