from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .query_planning import QueryPlan
from .schemas import RetrievalHit


@dataclass(frozen=True, slots=True)
class GraphExpansionResult:
    """Bounded graph-expansion output; warnings are safe for the public response."""

    hits: tuple[RetrievalHit, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class StructuralGraphStore(Protocol):
    """Optional vector-store capability for one-hop structural neighbors."""

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
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult: ...


class StoreBackedStructuralGraphRetriever:
    """GraphRAG-lite adapter using existing parent/heading relationships as graph edges."""

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
        source_ids: list[str] | None,
        active_versions: dict[str, str],
        legacy_excluded_source_ids: set[str],
        tenant_id: str,
        principals: list[str] | None,
    ) -> GraphExpansionResult:
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
            f"graph_expansion:one_hop:{len(hits)}" if hits else "graph_expansion:no_neighbors"
        )
        return GraphExpansionResult(hits=tuple(hits), warnings=(warning,))


_RELATIONAL_MARKERS = (
    "compare",
    "comparison",
    "relationship",
    "related",
    "depend",
    "impact",
    "cause",
    "across",
    "between",
    "multi-hop",
    "why",
    "对比",
    "比较",
    "关系",
    "关联",
    "依赖",
    "影响",
    "导致",
    "原因",
    "跨章节",
    "多跳",
)


def should_expand_graph(plan: QueryPlan) -> bool:
    """Route only compound or relational questions to graph expansion."""

    if any(variant.kind == "subquery" for variant in plan.variants):
        return True
    folded = plan.original_query.casefold()
    return any(marker in folded for marker in _RELATIONAL_MARKERS)
