from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evaluate_rag import evaluate, load_queries, load_results


def test_evaluation_contract_validates_example_file() -> None:
    path = Path(__file__).parents[1] / "eval" / "golden_queries.example.jsonl"
    queries = load_queries(path)
    assert len(queries) == 2
    assert queries[0].answerable is True
    assert queries[1].answerable is False


def test_evaluation_metrics_are_deterministic(tmp_path) -> None:
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "question",
                "tenant_id": "test",
                "acl": ["test"],
                "source_ids": None,
                "relevant_source_ids": ["a"],
                "relevant_chunk_ids": [],
                "answerable": True,
                "expected_citation_source_ids": ["a"],
                "tags": ["unit"],
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"query_id":"q1","source_ids":["b","a"]}\n', encoding="utf-8")

    report = evaluate(load_queries(query_path), load_results(results_path), top_k=2)

    assert report["source_recall_at_k"] == 1.0
    assert report["mrr_at_k"] == 0.5
    assert 0 < report["ndcg_at_k"] < 1


def test_answerable_query_requires_relevance_label(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"query_id":"q","query":"x","tenant_id":"t","acl":[],"source_ids":null,'
        '"relevant_source_ids":[],"relevant_chunk_ids":[],"answerable":true,'
        '"expected_citation_source_ids":[],"tags":[],"notes":""}\n',
        encoding="utf-8",
    )
    with pytest.raises((ValueError, ValidationError)):
        load_queries(path)
