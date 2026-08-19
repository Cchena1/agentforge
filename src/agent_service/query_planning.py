from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schemas import ChatMessage

QueryVariantKind = Literal["original", "subquery", "normalized", "rewrite"]
RetrievalRoute = Literal["hybrid", "graph"]


class QueryVariant(BaseModel):
    """A bounded, explainable retrieval query variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=10_000)
    kind: QueryVariantKind
    reason: str = Field(min_length=1, max_length=200)
    protected_anchors: tuple[str, ...] = ()


class QueryPlan(BaseModel):
    """Bounded query plan; original_query is always the first variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str = Field(min_length=1, max_length=10_000)
    variants: tuple[QueryVariant, ...] = Field(min_length=1, max_length=4)
    route: RetrievalRoute = "hybrid"
    graph_hops: int = Field(default=0, ge=0, le=2)
    intent: str = Field(default="fact", min_length=1, max_length=80)
    discarded_variants: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class QueryPlanner(Protocol):
    """Replaceable query-planning boundary for deterministic or model-backed planners."""

    @property
    def profile_id(self) -> str: ...

    async def plan(self, query: str, *, max_variants: int = 3) -> QueryPlan: ...


class QueryPlanningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: RetrievalRoute
    graph_hops: int = Field(default=0, ge=0, le=2)
    intent: str = Field(default="fact", min_length=1, max_length=80)
    subqueries: tuple[str, ...] = Field(default=(), max_length=2)
    rewrites: tuple[str, ...] = Field(default=(), max_length=2)
    rationale: str = Field(default="", max_length=500)


class QueryPlanningModel(Protocol):
    @property
    def online(self) -> bool: ...

    async def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        response_model: type[QueryPlanningDecision],
    ) -> QueryPlanningDecision: ...


class DeterministicQueryPlanner:
    """Zero-LLM-cost planner with bounded decomposition and drift guards."""

    profile_id = "deterministic-query-plan-v2"

    async def plan(self, query: str, *, max_variants: int = 3) -> QueryPlan:
        return build_query_plan(query, max_variants=max_variants)


class AdaptiveQueryPlanner:
    """Use a structured model only for ambiguous, decomposable, or relational questions."""

    profile_id = "adaptive-structured-query-plan-v1"

    def __init__(
        self,
        model: QueryPlanningModel,
        fallback: QueryPlanner | None = None,
        *,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.model = model
        self.fallback = fallback or DeterministicQueryPlanner()
        self.timeout_seconds = timeout_seconds

    async def plan(self, query: str, *, max_variants: int = 3) -> QueryPlan:
        base = await self.fallback.plan(query, max_variants=max_variants)
        if not self.model.online or not requires_model_planning(base):
            return base
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Plan retrieval for an enterprise RAG system. Choose graph only for relational, "
                    "multi-hop, comparison, dependency, or cross-section questions. Use at most two "
                    "graph hops. Preserve quoted phrases, identifiers, versions, numbers, units, and "
                    "negations exactly. Decompose only independent information needs. Rewrites must "
                    "retain the user's intent and must not add facts."
                ),
            },
            {"role": "user", "content": base.original_query},
        ]
        try:
            import asyncio

            async with asyncio.timeout(self.timeout_seconds):
                decision = await self.model.structured(messages, QueryPlanningDecision)
        except Exception as exc:  # noqa: BLE001 - model planner is an optional bounded branch
            return base.model_copy(
                update={
                    "warnings": (*base.warnings, f"query_plan:model_fallback:{type(exc).__name__}")
                }
            )
        return _merge_model_decision(base, decision, max_variants=max_variants)


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", flags=re.UNICODE)
_QUOTED_PATTERN = re.compile(r'["“]([^"”]{1,160})["”]')
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{1,20}\b")
_VERSION_PATTERN = re.compile(r"\b[vV]?\d+(?:\.\d+){1,4}\b")
_IDENTIFIER_PATTERN = re.compile(r"\b[\w]+(?:-[\w]+)+\b", flags=re.UNICODE)
_NUMBER_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|min|h|kb|mb|gb|tb|%|hz|khz|mhz|ghz|v|a|w|°c)(?=\W|$)",
    flags=re.IGNORECASE,
)
_NEGATIONS = (
    "must not",
    "do not",
    "does not",
    "not",
    "without",
    "exclude",
    "禁止",
    "不得",
    "不要",
    "不能",
    "不含",
    "未",
)
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
_AMBIGUITY_MARKERS = (
    "it",
    "they",
    "this approach",
    "that system",
    "前者",
    "后者",
    "该方案",
    "这个",
    "上述",
)


def build_query_plan(query: str, *, max_variants: int = 3) -> QueryPlan:
    """Build a zero-LLM-cost plan with explicit decomposition and drift guards."""

    if not 1 <= max_variants <= 4:
        raise ValueError("max_variants must be between 1 and 4")
    raw_query = query.strip()
    original = _collapse_whitespace(raw_query)
    if not original:
        raise ValueError("query must not be blank")

    anchors = extract_protected_anchors(original)
    variants = [
        QueryVariant(
            query=original,
            kind="original",
            reason="authoritative user query; never replaced",
            protected_anchors=anchors,
        )
    ]
    discarded: list[str] = []
    warnings: list[str] = []

    parts = [_strip_terminal_delimiters(part) for part in _split_explicit_parts(raw_query)]
    parts = [part for part in parts if _is_meaningful_part(part)]
    if len(parts) > 1:
        for part in parts:
            if len(variants) >= max_variants:
                discarded.append("subquery_limit")
                break
            if part.casefold() == original.casefold() or _contains_query(variants, part):
                continue
            variants.append(
                QueryVariant(
                    query=part,
                    kind="subquery",
                    reason="explicit user-delimited sub-question",
                    protected_anchors=extract_protected_anchors(part),
                )
            )
        if len(variants) > 1:
            warnings.append(f"query_plan:explicit_decomposition:{len(variants) - 1}")

    normalized = normalize_lexical_query(original)
    if (
        len(variants) < max_variants
        and normalized.casefold() != original.casefold()
        and not _contains_query(variants, normalized)
    ):
        if preserves_protected_anchors(normalized, anchors):
            variants.append(
                QueryVariant(
                    query=normalized,
                    kind="normalized",
                    reason="punctuation/case normalization for lexical retrieval",
                    protected_anchors=anchors,
                )
            )
        else:
            discarded.append("normalized_anchor_drift")
            warnings.append("query_plan:discarded_anchor_drift")

    graph_route = is_relational_query(original) or any(
        variant.kind == "subquery" for variant in variants
    )
    graph_hops = 2 if graph_route and _requires_two_hops(original) else int(graph_route)
    return QueryPlan(
        original_query=original,
        variants=tuple(variants),
        route="graph" if graph_route else "hybrid",
        graph_hops=graph_hops,
        intent="relational" if graph_route else "fact",
        discarded_variants=tuple(dict.fromkeys(discarded)),
        warnings=tuple(warnings),
    )


def requires_model_planning(plan: QueryPlan) -> bool:
    folded = plan.original_query.casefold()
    return (
        plan.route == "graph"
        or len(plan.original_query) > 180
        or any(marker in folded for marker in _AMBIGUITY_MARKERS)
    )


def is_relational_query(query: str) -> bool:
    folded = query.casefold()
    return any(marker in folded for marker in _RELATIONAL_MARKERS)


def normalize_lexical_query(query: str) -> str:
    terms = [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(query)]
    return " ".join(dict.fromkeys(terms)) or _collapse_whitespace(query)


def extract_protected_anchors(query: str) -> tuple[str, ...]:
    anchors: list[str] = []
    for pattern in (
        _QUOTED_PATTERN,
        _ACRONYM_PATTERN,
        _VERSION_PATTERN,
        _IDENTIFIER_PATTERN,
        _NUMBER_UNIT_PATTERN,
    ):
        for match in pattern.finditer(query):
            value = match.group(1) if pattern is _QUOTED_PATTERN else match.group(0)
            if value.strip():
                anchors.append(value.strip())
    folded = query.casefold()
    anchors.extend(item for item in _NEGATIONS if item.casefold() in folded)
    return tuple(dict.fromkeys(anchors))


def preserves_protected_anchors(candidate: str, anchors: tuple[str, ...]) -> bool:
    candidate_key = _comparison_key(candidate)
    return all(_comparison_key(anchor) in candidate_key for anchor in anchors)


def _merge_model_decision(
    base: QueryPlan,
    decision: QueryPlanningDecision,
    *,
    max_variants: int,
) -> QueryPlan:
    variants = [base.variants[0]]
    discarded = list(base.discarded_variants)
    anchors = extract_protected_anchors(base.original_query)
    candidates = [
        *(QueryVariant(
            query=value.strip(),
            kind="subquery",
            reason="structured model decomposition",
            protected_anchors=extract_protected_anchors(value),
        ) for value in decision.subqueries if value.strip()),
        *(QueryVariant(
            query=value.strip(),
            kind="rewrite",
            reason="structured model semantic rewrite",
            protected_anchors=anchors,
        ) for value in decision.rewrites if value.strip()),
    ]
    for candidate in candidates:
        if len(variants) >= max_variants:
            discarded.append("model_variant_limit")
            break
        if _contains_query(variants, candidate.query):
            continue
        if candidate.kind == "rewrite" and not preserves_protected_anchors(candidate.query, anchors):
            discarded.append("model_rewrite_anchor_drift")
            continue
        variants.append(candidate)
    if len(variants) < max_variants:
        for candidate in base.variants[1:]:
            if not _contains_query(variants, candidate.query):
                variants.append(candidate)
            if len(variants) >= max_variants:
                break
    graph_hops = min(2, decision.graph_hops) if decision.route == "graph" else 0
    return QueryPlan(
        original_query=base.original_query,
        variants=tuple(variants),
        route=decision.route,
        graph_hops=graph_hops,
        intent=decision.intent,
        discarded_variants=tuple(dict.fromkeys(discarded)),
        warnings=(
            *base.warnings,
            "query_plan:model_routed",
            *(() if "model_rewrite_anchor_drift" not in discarded else (
                "query_plan:discarded_anchor_drift",
            )),
        ),
    )


def _requires_two_hops(query: str) -> bool:
    folded = query.casefold()
    return any(
        marker in folded
        for marker in ("multi-hop", "across", "why", "多跳", "跨章节", "原因", "如何导致")
    )


def _split_explicit_parts(value: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    quote_end: str | None = None
    quote_pairs = {'"': '"', "“": "”"}
    for character in value:
        if quote_end is not None:
            buffer.append(character)
            if character == quote_end:
                quote_end = None
            continue
        if character in quote_pairs:
            quote_end = quote_pairs[character]
            buffer.append(character)
            continue
        if character in "?？;；\r\n":
            if buffer:
                parts.append("".join(buffer))
                buffer.clear()
            continue
        buffer.append(character)
    if buffer:
        parts.append("".join(buffer))
    return parts


def _contains_query(variants: list[QueryVariant], candidate: str) -> bool:
    folded = candidate.casefold()
    return any(item.query.casefold() == folded for item in variants)


def _comparison_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _strip_terminal_delimiters(value: str) -> str:
    return value.strip().strip("?？;；").strip()


def _is_meaningful_part(value: str) -> bool:
    return len(_comparison_key(value)) >= 2
