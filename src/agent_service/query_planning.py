from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

QueryVariantKind = Literal["original", "subquery", "normalized"]


class QueryVariant(BaseModel):
    """A bounded, explainable retrieval query variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=10_000)
    kind: QueryVariantKind
    reason: str = Field(min_length=1, max_length=200)
    protected_anchors: tuple[str, ...] = ()


class QueryPlan(BaseModel):
    """Deterministic query plan; original_query is always the first variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str = Field(min_length=1, max_length=10_000)
    variants: tuple[QueryVariant, ...] = Field(min_length=1, max_length=4)
    discarded_variants: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class QueryPlanner(Protocol):
    """Replaceable query-planning boundary for deterministic or model-backed planners."""

    @property
    def profile_id(self) -> str: ...

    async def plan(self, query: str, *, max_variants: int = 3) -> QueryPlan: ...


class DeterministicQueryPlanner:
    """Default zero-LLM-cost planner with bounded rewrite and drift guards."""

    profile_id = "deterministic-query-plan-v1"

    async def plan(self, query: str, *, max_variants: int = 3) -> QueryPlan:
        return build_query_plan(query, max_variants=max_variants)


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

    return QueryPlan(
        original_query=original,
        variants=tuple(variants),
        discarded_variants=tuple(dict.fromkeys(discarded)),
        warnings=tuple(warnings),
    )


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
