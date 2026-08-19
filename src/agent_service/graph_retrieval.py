from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .knowledge_graph import KnowledgeGraphStore
from .query_planning import QueryPlan
from .schemas import RetrievalHit
from .vector_store import _citation_from_evidence


@dataclass(frozen=True, slots=True)
class GraphExpansionResult:
    """Bounded graph-expansion output; warnings are safe for the public response."""

    hits: tuple[RetrievalHit, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class StructuralGraphStore(Protocol):
    """Compatibility capability for structural-neighbor fallback."""

    async def graph_neighbors(
        self,
        query: str,
        *,
        seed_hits: Sequence[RetrievalHit],
        max_neighbors: int,
        neighbors_per_seed: int,
        source_ids: list[str] | None = None,
        active_versions: dict[str, str] | None = None,
        legacy_excluded_source_ids: set[str] | None = None,
        tenant_id: str = "public",
        principals: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], float]: ...


class GraphRetriever(Protocol):
    """Replaceable graph-retrieval plugin boundary."""

    @property
    def profile_id(self) -> str: ...

    async def expand(
        self,
        query: str,
        plan: QueryPlan,
        seed_hits: Sequence[RetrievalHit],
        *,
        max_seed_hits: int,
        max_neighbors: int,
        neighbors_per_seed: int,
        max_hops: int,
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult: ...


class KnowledgeGraphRetriever:
    """Entity/relation graph traversal with version, tenant, ACL, and hop bounds."""

    profile_id = "knowledge-graph-local-search-v1"

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    async def expand(
        self,
        query: str,
        plan: QueryPlan,
        seed_hits: Sequence[RetrievalHit],
        *,
        max_seed_hits: int,
        max_neighbors: int,
        neighbors_per_seed: int,
        max_hops: int,
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult:
        del neighbors_per_seed, legacy_excluded_source_ids
        if not seed_hits or not should_expand_graph(plan):
            return GraphExpansionResult()
        bounded_hops = max(1, min(2, max_hops, plan.graph_hops))
        seed_chunk_ids = [
            hit.citation.chunk_id
            for hit in seed_hits[:max_seed_hits]
            if hit.citation.chunk_id is not None
        ]
        evidence = await self.store.expand(
            seed_chunk_ids,
            query=query,
            max_hops=bounded_hops,
            max_results=max_neighbors,
            source_ids=source_ids,
            active_versions=active_versions,
            tenant_id=tenant_id,
            principals=principals,
        )
        query_terms = Counter(_terms(query))
        hits = tuple(
            RetrievalHit(
                text=item.text,
                citation=_citation_from_evidence(
                    source_id=item.source_id,
                    source_name=item.source_name,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    page=item.page,
                    locator=item.locator,
                    metadata=item.metadata,
                    query_terms=query_terms,
                    score=item.graph_score,
                ),
                vector_score=0.0,
                lexical_score=0.0,
                fused_score=item.graph_score,
                rerank_score=item.graph_score,
            )
            for item in evidence
        )
        if not hits:
            return GraphExpansionResult(warnings=("graph_expansion:no_neighbors",))
        observed_hops = max(item.hops for item in evidence)
        return GraphExpansionResult(
            hits=hits,
            warnings=(
                f"graph_expansion:knowledge_graph:{len(hits)}",
                f"graph_expansion:max_hops:{observed_hops}",
            ),
        )


class StoreBackedStructuralGraphRetriever:
    """Compatibility fallback using existing parent/heading relationships as graph edges."""

    profile_id = "structural-one-hop-v1"

    def __init__(self, store: object) -> None:
        self.store = store

    async def expand(
        self,
        query: str,
        plan: QueryPlan,
        seed_hits: Sequence[RetrievalHit],
        *,
        max_seed_hits: int,
        max_neighbors: int,
        neighbors_per_seed: int,
        max_hops: int,
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult:
        del max_hops
        if not seed_hits or not should_expand_graph(plan):
            return GraphExpansionResult()
        if not isinstance(self.store, StructuralGraphStore):
            return GraphExpansionResult(warnings=("graph_expansion:unsupported_backend",))
        hits, _ = await self.store.graph_neighbors(
            query,
            seed_hits=tuple(seed_hits[:max_seed_hits]),
            max_neighbors=max_neighbors,
            neighbors_per_seed=neighbors_per_seed,
            source_ids=source_ids,
            active_versions=active_versions,
            legacy_excluded_source_ids=legacy_excluded_source_ids,
            tenant_id=tenant_id,
            principals=principals,
        )
        warning = (
            f"graph_expansion:structural_fallback:{len(hits)}"
            if hits
            else "graph_expansion:no_neighbors"
        )
        return GraphExpansionResult(hits=tuple(hits), warnings=(warning,))


class FallbackGraphRetriever:
    """Prefer the entity graph, then degrade to structural adjacency without losing seeds."""

    profile_id = "knowledge-graph-with-structural-fallback-v1"

    def __init__(self, primary: GraphRetriever, fallback: GraphRetriever) -> None:
        self.primary = primary
        self.fallback = fallback

    async def expand(
        self,
        query: str,
        plan: QueryPlan,
        seed_hits: Sequence[RetrievalHit],
        *,
        max_seed_hits: int,
        max_neighbors: int,
        neighbors_per_seed: int,
        max_hops: int,
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult:
        try:
            result = await self.primary.expand(
                query,
                plan,
                seed_hits,
                max_seed_hits=max_seed_hits,
                max_neighbors=max_neighbors,
                neighbors_per_seed=neighbors_per_seed,
                max_hops=max_hops,
                source_ids=source_ids,
                active_versions=active_versions,
                legacy_excluded_source_ids=legacy_excluded_source_ids,
                tenant_id=tenant_id,
                principals=principals,
            )
            if result.hits:
                return result
        except Exception as exc:  # noqa: BLE001 - explicit graph backend fallback boundary
            primary_warning = f"graph_expansion:knowledge_graph_failed:{type(exc).__name__}"
        else:
            primary_warning = "graph_expansion:knowledge_graph_empty"
        fallback = await self.fallback.expand(
            query,
            plan,
            seed_hits,
            max_seed_hits=max_seed_hits,
            max_neighbors=max_neighbors,
            neighbors_per_seed=neighbors_per_seed,
            max_hops=max_hops,
            source_ids=source_ids,
            active_versions=active_versions,
            legacy_excluded_source_ids=legacy_excluded_source_ids,
            tenant_id=tenant_id,
            principals=principals,
        )
        return GraphExpansionResult(
            hits=fallback.hits,
            warnings=(primary_warning, *fallback.warnings),
        )


def should_expand_graph(plan: QueryPlan) -> bool:
    """Only graph-routed questions can spend the graph traversal budget."""

    return plan.route == "graph" and plan.graph_hops > 0


def _terms(value: str) -> list[str]:
    import re

    return [
        item.casefold()
        for item in re.findall(r"[\w\u4e00-\u9fff]+", value)
        if len(item) > 1
    ]
