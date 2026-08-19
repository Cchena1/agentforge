from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agent_service.embeddings import SentenceTransformerEmbedding
from agent_service.knowledge_graph import (
    ChunkGraphExtraction,
    GraphChunk,
    GraphEntityDraft,
    GraphRelationDraft,
    KnowledgeGraphBuildContext,
    KnowledgeGraphBuilder,
    SQLiteKnowledgeGraphStore,
)
from agent_service.query_planning import (
    AdaptiveQueryPlanner,
    QueryPlanningDecision,
    build_query_plan,
)
from agent_service.rag_reflection import (
    AdaptiveEvidenceReflector,
    EvidenceReflectionDecision,
)
from agent_service.schemas import ChatMessage


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 64

    def encode_query(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append("query")
        return [[1.0, *([0.0] * 63)] for _ in texts]

    def encode_document(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append("document")
        return [[0.0, 1.0, *([0.0] * 62)] for _ in texts]


def test_sentence_transformer_embedding_fails_fast_without_semantic_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_service.embeddings.find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="uv sync --extra semantic"):
        SentenceTransformerEmbedding()


@pytest.mark.asyncio
async def test_sentence_transformer_embedding_separates_query_and_document_encoding() -> None:
    fake = FakeSentenceTransformer()
    provider = SentenceTransformerEmbedding(
        "test-semantic-model",
        dimension=64,
        model_factory=lambda _model, _device: fake,
    )

    document_vectors = await provider.embed_documents(["alpha"])
    query_vector = await provider.embed_query("alpha")

    assert fake.calls == ["document", "query"]
    assert document_vectors[0][1] == 1.0
    assert query_vector[0] == 1.0
    assert provider.profile_id == "sentence-transformers:test-semantic-model:64"


class ChainExtractor:
    profile_id = "test-chain-extractor-v1"

    async def extract(
        self, chunks: Sequence[GraphChunk]
    ) -> tuple[ChunkGraphExtraction, ...]:
        outputs = {
            "v1:c1": ChunkGraphExtraction(
                chunk_id="v1:c1",
                entities=[GraphEntityDraft(key="alpha", name="Alpha")],
            ),
            "v1:c2": ChunkGraphExtraction(
                chunk_id="v1:c2",
                entities=[
                    GraphEntityDraft(key="alpha", name="Alpha"),
                    GraphEntityDraft(key="beta", name="Beta"),
                ],
                relations=[
                    GraphRelationDraft(
                        source_key="alpha",
                        target_key="beta",
                        relation_type="DEPENDS_ON",
                    )
                ],
            ),
            "v1:c3": ChunkGraphExtraction(
                chunk_id="v1:c3",
                entities=[GraphEntityDraft(key="gamma", name="Gamma")],
            ),
            "v1:c4": ChunkGraphExtraction(
                chunk_id="v1:c4",
                entities=[
                    GraphEntityDraft(key="beta", name="Beta"),
                    GraphEntityDraft(key="gamma", name="Gamma"),
                ],
                relations=[
                    GraphRelationDraft(
                        source_key="beta",
                        target_key="gamma",
                        relation_type="CONNECTS_TO",
                    )
                ],
            ),
        }
        return tuple(outputs[chunk.chunk_id] for chunk in chunks)


@pytest.mark.asyncio
async def test_knowledge_graph_store_returns_bounded_two_hop_evidence(tmp_path) -> None:
    chunks = [
        GraphChunk("v1:c1", "Alpha overview", 1, "section:one", {"heading_path": ["One"]}),
        GraphChunk(
            "v1:c2",
            "Alpha depends on Beta",
            2,
            "section:two",
            {"heading_path": ["Two"]},
        ),
        GraphChunk(
            "v1:c3",
            "Gamma operational details",
            3,
            "section:three",
            {"heading_path": ["Three"]},
        ),
        GraphChunk(
            "v1:c4",
            "Beta connects to Gamma",
            4,
            "section:four",
            {"heading_path": ["Four"]},
        ),
    ]
    builder = KnowledgeGraphBuilder(ChainExtractor())
    document = await builder.build(
        KnowledgeGraphBuildContext(
            tenant_id="tenant-a",
            source_id="source-a",
            source_name="graph.pdf",
            version_id="v1",
            acl=("team-a",),
        ),
        chunks,
    )
    store = SQLiteKnowledgeGraphStore(tmp_path / "knowledge_graph.sqlite3")
    await store.replace_version(document)

    one_hop = await store.expand(
        ["v1:c1"],
        query="How is Alpha connected to Gamma?",
        max_hops=1,
        max_results=10,
        source_ids=["source-a"],
        active_versions={"source-a": "v1"},
        tenant_id="tenant-a",
        principals=["team-a"],
    )
    two_hop = await store.expand(
        ["v1:c1"],
        query="How is Alpha connected to Gamma?",
        max_hops=2,
        max_results=10,
        source_ids=["source-a"],
        active_versions={"source-a": "v1"},
        tenant_id="tenant-a",
        principals=["team-a"],
    )

    assert "v1:c3" not in {item.chunk_id for item in one_hop}
    gamma = next(item for item in two_hop if item.chunk_id == "v1:c3")
    assert gamma.hops == 2
    assert gamma.path[-1] == "Gamma"
    assert not await store.expand(
        ["v1:c1"],
        query="Alpha Gamma",
        max_hops=2,
        max_results=10,
        source_ids=["source-a"],
        active_versions={"source-a": "v1"},
        tenant_id="tenant-a",
        principals=["other-team"],
    )


class CrossAclExtractor:
    profile_id = "test-cross-acl-extractor-v1"

    async def extract(
        self, chunks: Sequence[GraphChunk]
    ) -> tuple[ChunkGraphExtraction, ...]:
        outputs = {
            "public:p1": ChunkGraphExtraction(
                chunk_id="public:p1",
                entities=[GraphEntityDraft(key="alpha", name="Alpha")],
            ),
            "public:p2": ChunkGraphExtraction(
                chunk_id="public:p2",
                entities=[GraphEntityDraft(key="gamma", name="Gamma")],
            ),
            "secret:s1": ChunkGraphExtraction(
                chunk_id="secret:s1",
                entities=[
                    GraphEntityDraft(key="alpha", name="Alpha"),
                    GraphEntityDraft(key="beta", name="Beta"),
                ],
                relations=[
                    GraphRelationDraft(
                        source_key="alpha",
                        target_key="beta",
                        relation_type="SECRET_DEPENDENCY",
                    )
                ],
            ),
            "secret:s2": ChunkGraphExtraction(
                chunk_id="secret:s2",
                entities=[
                    GraphEntityDraft(key="beta", name="Beta"),
                    GraphEntityDraft(key="gamma", name="Gamma"),
                ],
                relations=[
                    GraphRelationDraft(
                        source_key="beta",
                        target_key="gamma",
                        relation_type="SECRET_CONNECTION",
                    )
                ],
            ),
        }
        return tuple(outputs[chunk.chunk_id] for chunk in chunks)


@pytest.mark.asyncio
async def test_knowledge_graph_does_not_traverse_relations_hidden_by_acl(tmp_path) -> None:
    builder = KnowledgeGraphBuilder(CrossAclExtractor())
    public_document = await builder.build(
        KnowledgeGraphBuildContext(
            tenant_id="tenant-a",
            source_id="source-public",
            source_name="public.md",
            version_id="v-public",
            acl=("team-a",),
        ),
        [
            GraphChunk(
                "public:p1",
                "Alpha public overview",
                1,
                "public:one",
                {"heading_path": ["Alpha"]},
            ),
            GraphChunk(
                "public:p2",
                "Gamma public overview",
                2,
                "public:two",
                {"heading_path": ["Gamma"]},
            ),
        ],
    )
    secret_document = await builder.build(
        KnowledgeGraphBuildContext(
            tenant_id="tenant-a",
            source_id="source-secret",
            source_name="secret.md",
            version_id="v-secret",
            acl=("team-secret",),
        ),
        [
            GraphChunk("secret:s1", "Alpha secret dependency Beta", 1, "secret:one", {}),
            GraphChunk("secret:s2", "Beta secret connection Gamma", 2, "secret:two", {}),
        ],
    )
    store = SQLiteKnowledgeGraphStore(tmp_path / "knowledge_graph.sqlite3")
    await store.replace_version(public_document)
    await store.replace_version(secret_document)

    evidence = await store.expand(
        ["public:p1"],
        query="How is Alpha connected to Gamma?",
        max_hops=2,
        max_results=10,
        source_ids=None,
        active_versions={"source-public": "v-public", "source-secret": "v-secret"},
        tenant_id="tenant-a",
        principals=["team-a"],
    )

    assert "public:p2" not in {item.chunk_id for item in evidence}


class PlannerModel:
    online = True

    async def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        response_model: type[QueryPlanningDecision],
    ) -> QueryPlanningDecision:
        del messages, response_model
        return QueryPlanningDecision(
            route="graph",
            graph_hops=2,
            intent="dependency",
            rewrites=["Compare API-v3 dependency across sections"],
            rationale="Multi-hop dependency question.",
        )


@pytest.mark.asyncio
async def test_adaptive_query_planner_routes_two_hops_and_rejects_anchor_drift() -> None:
    planner = AdaptiveQueryPlanner(PlannerModel())
    plan = await planner.plan("Compare API-v2 dependency across sections", max_variants=3)

    assert plan.route == "graph"
    assert plan.graph_hops == 2
    assert all("API-v3" not in variant.query for variant in plan.variants)
    assert "query_plan:discarded_anchor_drift" in plan.warnings


class ReflectionModel:
    online = True

    async def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        response_model: type[EvidenceReflectionDecision],
    ) -> EvidenceReflectionDecision:
        del messages, response_model
        return EvidenceReflectionDecision(
            sufficient=False,
            confidence=0.2,
            failure_type="no_evidence",
            rewrites=[
                "API-v2 dependency relationship",
                "API-v3 dependency relationship",
            ],
            rationale="No evidence covers the dependency chain.",
        )


@pytest.mark.asyncio
async def test_reflection_rewrite_is_bounded_and_anchor_safe() -> None:
    query = "Why does API-v2 depend on the gateway?"
    reflector = AdaptiveEvidenceReflector(ReflectionModel())
    result = await reflector.reflect(
        query,
        build_query_plan(query),
        [],
        min_relevance_score=0.2,
        max_rewrites=2,
    )

    assert result.rewrites == ("API-v2 dependency relationship",)
    assert "reflection:rewrite_discarded_anchor_drift" in result.warnings
