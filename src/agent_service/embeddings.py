from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    dimension: int
    profile_id: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedding:
    """Deterministic offline baseline; useful for tests and lexical-heavy small corpora."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("dimension must be >= 64")
        self.dimension = dimension
        self.profile_id = f"hash-blake2b-v1:{dimension}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

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
        if not texts:
            return []
        response = await self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                f"embedding dimension mismatch for {self.profile_id}: expected {self.dimension}"
            )
        return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
