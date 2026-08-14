from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=10_000)
    tenant_id: str = Field(min_length=1, max_length=128)
    acl: list[str]
    source_ids: list[str] | None = None
    relevant_source_ids: list[str]
    relevant_chunk_ids: list[str]
    answerable: bool
    expected_citation_source_ids: list[str]
    tags: list[str]
    notes: str = ""

    @model_validator(mode="after")
    def validate_relevance(self) -> GoldenQuery:
        if self.answerable and not self.relevant_source_ids:
            raise ValueError("answerable queries require at least one relevant source")
        if not self.answerable and self.relevant_source_ids:
            raise ValueError("unanswerable queries cannot declare relevant sources")
        return self


@dataclass(slots=True, frozen=True)
class RankedResult:
    source_ids: list[str]


def load_queries(path: Path) -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            queries.append(GoldenQuery.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not queries:
        raise ValueError(f"{path} contains no evaluation queries")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be unique")
    return queries


def load_results(path: Path) -> dict[str, RankedResult]:
    payload: dict[str, RankedResult] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        item: dict[str, Any] = json.loads(line)
        query_id = str(item["query_id"])
        if query_id in payload:
            raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id}")
        payload[query_id] = RankedResult(source_ids=[str(value) for value in item["source_ids"]])
    return payload


def evaluate(queries: list[GoldenQuery], results: dict[str, RankedResult], top_k: int) -> dict[str, Any]:
    answerable = [query for query in queries if query.answerable]
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    empty_results = 0
    for query in queries:
        ranked = results.get(query.query_id, RankedResult([])).source_ids[:top_k]
        if not ranked:
            empty_results += 1
        if not query.answerable:
            continue
        relevant = set(query.relevant_source_ids)
        matched = [source_id for source_id in ranked if source_id in relevant]
        recalls.append(len(set(matched)) / len(relevant))
        first_rank = next((rank for rank, source_id in enumerate(ranked, 1) if source_id in relevant), None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        dcg = sum(1.0 / math.log2(rank + 1) for rank, source_id in enumerate(ranked, 1) if source_id in relevant)
        ideal_hits = min(len(relevant), top_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
    denominator = len(answerable) or 1
    return {
        "schema_version": 1,
        "query_count": len(queries),
        "answerable_query_count": len(answerable),
        "top_k": top_k,
        "source_recall_at_k": sum(recalls) / denominator,
        "mrr_at_k": sum(reciprocal_ranks) / denominator,
        "ndcg_at_k": sum(ndcgs) / denominator,
        "empty_result_rate": empty_results / len(queries),
        "scope": "functional retrieval quality; no pressure-test claims",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and score AgentForge RAG evaluation records.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    queries = load_queries(args.queries)
    if args.results is None:
        report: dict[str, Any] = {
            "schema_version": 1,
            "query_count": len(queries),
            "status": "validated",
            "scope": "contract validation only; no quality or pressure-test claim",
        }
    else:
        report = evaluate(queries, load_results(args.results), args.top_k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
