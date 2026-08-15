from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from pypdf import PdfWriter

from agent_service.documents import BuiltinDocumentParser, CompositeDocumentParser, SemanticChunker
from agent_service.embeddings import HashEmbedding
from agent_service.rag import RAGService
from agent_service.rag_registry import SQLiteVersionRegistry
from agent_service.schemas import DocumentIngestRequest
from agent_service.vector_store import SQLiteVectorStore, VectorDocument, _rank_scores


class FailingUpsertStore(SQLiteVectorStore):
    def __init__(self, db_path, embedding_provider) -> None:
        super().__init__(db_path, embedding_provider)
        self.fail_next = False

    async def upsert(self, documents) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated vector-store failure")
        await super().upsert(documents)


class WrongDimensionEmbedding(HashEmbedding):
    async def embed(self, texts):
        vectors = await super().embed(texts)
        return [vector[:-1] for vector in vectors]


class MutatingParser(CompositeDocumentParser):
    async def parse(self, path):
        parsed = await super().parse(path)
        path.write_text("mutated after parse", encoding="utf-8")
        return parsed


class BlockingUpsertStore(SQLiteVectorStore):
    def __init__(self, db_path, embedding_provider) -> None:
        super().__init__(db_path, embedding_provider)
        self.block_next = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def upsert(self, documents) -> None:
        if self.block_next:
            self.started.set()
            await self.release.wait()
            self.block_next = False
        await super().upsert(documents)


class TrackingUpsertStore(SQLiteVectorStore):
    def __init__(self, db_path, embedding_provider) -> None:
        super().__init__(db_path, embedding_provider)
        self.active_upserts = 0
        self.max_active_upserts = 0

    async def upsert(self, documents) -> None:
        self.active_upserts += 1
        self.max_active_upserts = max(self.max_active_upserts, self.active_upserts)
        try:
            await asyncio.sleep(0.03)
            await super().upsert(documents)
        finally:
            self.active_upserts -= 1


def build_rag(tmp_path, store_class=SQLiteVectorStore) -> RAGService:
    embeddings = HashEmbedding()
    store = store_class(tmp_path / "rag.sqlite3", embeddings)
    return RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        store,
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
    )


@pytest.mark.asyncio
async def test_existing_unversioned_sqlite_index_migrates_without_data_loss(tmp_path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    embeddings = HashEmbedding()
    vector = (await embeddings.embed(["legacy searchable evidence"]))[0]
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE document_chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                page INTEGER,
                locator TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_chunks_source ON document_chunks(source_id);
            """
        )
        db.execute(
            """
            INSERT INTO document_chunks(
                chunk_id, source_id, source_name, text, embedding_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, '{}')
            """,
            ("legacy:0", "legacy", "legacy.md", "legacy searchable evidence", json.dumps(vector)),
        )
        db.commit()

    store = SQLiteVectorStore(db_path, embeddings)
    await store.initialize()
    hits, _ = await store.search(
        "legacy searchable evidence",
        top_k=3,
        active_versions={},
    )

    with sqlite3.connect(db_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(document_chunks)")}
    assert {"version_id", "tenant_id", "acl_json"} <= columns
    assert hits
    assert hits[0].citation.source_id == "legacy"


@pytest.mark.asyncio
async def test_legacy_rows_remain_visible_until_first_version_activates(tmp_path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    embeddings = HashEmbedding()
    legacy_vector = (await embeddings.embed(["legacy active evidence"]))[0]
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE document_chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                page INTEGER,
                locator TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_chunks_source ON document_chunks(source_id);
            """
        )
        db.execute(
            """
            INSERT INTO document_chunks(
                chunk_id, source_id, source_name, text, embedding_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, '{}')
            """,
            (
                "legacy:0",
                "legacy",
                "legacy.md",
                "legacy active evidence",
                json.dumps(legacy_vector),
            ),
        )
        db.commit()

    (tmp_path / "replacement.md").write_text("replacement evidence", encoding="utf-8")
    store = BlockingUpsertStore(db_path, embeddings)
    rag = RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        store,
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
    )
    store.block_next = True
    update = asyncio.create_task(
        rag.ingest(DocumentIngestRequest(file_path="replacement.md", source_id="legacy"))
    )
    await store.started.wait()

    during = await rag.retrieve("legacy active evidence", top_k=5, source_ids=["legacy"])
    assert during.index_versions == {}
    assert during.hits
    assert all("legacy active evidence" in hit.text for hit in during.hits)

    store.release.set()
    activated = await update
    after = await rag.retrieve("replacement evidence", top_k=5, source_ids=["legacy"])
    assert after.index_versions == {"legacy": activated.version_id}
    assert after.hits
    assert all("replacement evidence" in hit.text for hit in after.hits)


@pytest.mark.asyncio
async def test_rag_semantic_chunks_and_returns_citations(tmp_path) -> None:
    document = tmp_path / "guide.md"
    document.write_text(
        "# Retry policy\n\nTransient timeouts use exponential backoff.\n\n"
        "# Fallback\n\nAfter retries, route to a backup model and then request human review.",
        encoding="utf-8",
    )
    rag = build_rag(tmp_path)
    ingested = await rag.ingest(DocumentIngestRequest(file_path="guide.md"))
    response = await rag.retrieve("backup model after transient timeout", top_k=3)
    assert ingested.chunks_created >= 2
    assert ingested.version_id
    assert ingested.content_sha256
    assert response.hits
    assert response.hits[0].citation.source_name == "guide.md"
    assert response.hits[0].citation.chunk_id
    assert response.index_versions[ingested.source_id] == ingested.version_id
    assert response.latency_ms >= 0
    with sqlite3.connect(tmp_path / "rag.sqlite3") as db:
        metadata = json.loads(
            db.execute(
                "SELECT metadata_json FROM document_chunks WHERE version_id = ? LIMIT 1",
                (ingested.version_id,),
            ).fetchone()[0]
        )
    assert metadata["document_schema_version"] == 1
    assert metadata["pipeline_profile"].startswith("document_schema=1|")


@pytest.mark.asyncio
async def test_reingesting_identical_content_is_idempotent(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("stable evidence", encoding="utf-8")
    rag = build_rag(tmp_path)

    first = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))
    second = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    assert first.idempotent is False
    assert second.idempotent is True
    assert second.version_id == first.version_id
    assert second.content_sha256 == first.content_sha256


@pytest.mark.asyncio
async def test_failed_replacement_keeps_previous_version_searchable(tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("old evidence remains active", encoding="utf-8")
    rag = build_rag(tmp_path, FailingUpsertStore)
    first = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    path.write_text("new evidence must not activate", encoding="utf-8")
    assert isinstance(rag.store, FailingUpsertStore)
    rag.store.fail_next = True
    with pytest.raises(RuntimeError, match="simulated"):
        await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    response = await rag.retrieve("old evidence", top_k=5, source_ids=["guide"])
    assert response.index_versions == {"guide": first.version_id}
    assert response.hits
    assert all("old evidence" in hit.text for hit in response.hits)
    assert all("new evidence" not in hit.text for hit in response.hits)


@pytest.mark.asyncio
async def test_failed_first_ingest_does_not_expose_partial_version(tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("partial evidence must remain invisible", encoding="utf-8")
    rag = build_rag(tmp_path, FailingUpsertStore)
    assert isinstance(rag.store, FailingUpsertStore)
    rag.store.fail_next = True

    with pytest.raises(RuntimeError, match="simulated"):
        await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    response = await rag.retrieve("partial evidence", top_k=5, source_ids=["guide"])
    assert response.index_versions == {}
    assert response.hits == []


@pytest.mark.asyncio
async def test_building_version_is_invisible_until_activation(tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("version one evidence", encoding="utf-8")
    rag = build_rag(tmp_path, BlockingUpsertStore)
    first = await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    path.write_text("version two evidence", encoding="utf-8")
    assert isinstance(rag.store, BlockingUpsertStore)
    rag.store.block_next = True
    update = asyncio.create_task(
        rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))
    )
    await rag.store.started.wait()

    during = await rag.retrieve("version evidence", top_k=5, source_ids=["guide"])
    assert during.index_versions == {"guide": first.version_id}
    assert during.hits
    assert all("version one" in hit.text for hit in during.hits)

    rag.store.release.set()
    second = await update
    after = await rag.retrieve("version evidence", top_k=5, source_ids=["guide"])
    assert second.version_id != first.version_id
    assert after.index_versions == {"guide": second.version_id}
    assert after.hits
    assert all("version two" in hit.text for hit in after.hits)


@pytest.mark.asyncio
async def test_same_source_ingests_are_serialized(tmp_path) -> None:
    (tmp_path / "first.md").write_text("first source revision", encoding="utf-8")
    (tmp_path / "second.md").write_text("second source revision", encoding="utf-8")
    rag = build_rag(tmp_path, TrackingUpsertStore)

    first_task = asyncio.create_task(
        rag.ingest(DocumentIngestRequest(file_path="first.md", source_id="shared"))
    )
    await asyncio.sleep(0)
    second_task = asyncio.create_task(
        rag.ingest(DocumentIngestRequest(file_path="second.md", source_id="shared"))
    )
    first, second = await asyncio.gather(first_task, second_task)

    assert isinstance(rag.store, TrackingUpsertStore)
    assert rag.store.max_active_upserts == 1
    response = await rag.retrieve("source revision", top_k=5, source_ids=["shared"])
    assert first.version_id != second.version_id
    assert response.index_versions == {"shared": second.version_id}
    assert response.hits
    assert all("second source revision" in hit.text for hit in response.hits)


@pytest.mark.asyncio
async def test_different_source_ingests_can_overlap(tmp_path) -> None:
    (tmp_path / "alpha.md").write_text("alpha evidence", encoding="utf-8")
    (tmp_path / "beta.md").write_text("beta evidence", encoding="utf-8")
    rag = build_rag(tmp_path, TrackingUpsertStore)

    await asyncio.gather(
        rag.ingest(DocumentIngestRequest(file_path="alpha.md", source_id="alpha")),
        rag.ingest(DocumentIngestRequest(file_path="beta.md", source_id="beta")),
    )

    assert isinstance(rag.store, TrackingUpsertStore)
    assert rag.store.max_active_upserts == 2


@pytest.mark.asyncio
async def test_source_mutation_during_ingestion_is_rejected(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("original stable evidence", encoding="utf-8")
    embeddings = HashEmbedding()
    rag = RAGService(
        tmp_path,
        MutatingParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings),
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
    )

    with pytest.raises(RuntimeError, match="changed during ingestion"):
        await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    response = await rag.retrieve("original stable evidence", top_k=5, source_ids=["guide"])
    assert response.index_versions == {}
    assert response.hits == []


@pytest.mark.asyncio
async def test_embedding_dimension_mismatch_is_rejected_before_indexing(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("dimension safety", encoding="utf-8")
    embeddings = WrongDimensionEmbedding()
    rag = RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings),
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
    )

    with pytest.raises(RuntimeError, match="dimension"):
        await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    response = await rag.retrieve("dimension safety", top_k=5, source_ids=["guide"])
    assert response.index_versions == {}
    assert response.hits == []


@pytest.mark.asyncio
async def test_table_rows_are_preserved_as_semantic_blocks(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("name,value,unit\nlatency,12,ms\nrecall,0.91,ratio\n", encoding="utf-8")
    parsed = await BuiltinDocumentParser().parse(path)
    assert parsed.parser == "builtin-table"
    assert parsed.blocks[0].kind == "table_row"
    assert "name=latency" in parsed.blocks[0].text
    assert parsed.blocks[0].locator == "row:2"


@pytest.mark.asyncio
async def test_scanned_pdf_is_detected_and_requests_ocr_fallback(tmp_path) -> None:
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)
    parsed = await BuiltinDocumentParser().parse(path)
    assert any("SCANNED_PDF_SUSPECTED" in warning for warning in parsed.warnings)
    assert "Docling" in parsed.warnings[0]


@pytest.mark.asyncio
async def test_registry_migrates_legacy_rows_to_public_tenant(tmp_path) -> None:
    db_path = tmp_path / "rag_registry.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE rag_sources (
                source_id TEXT PRIMARY KEY,
                active_version_id TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE rag_document_versions (
                version_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                pipeline_profile TEXT NOT NULL,
                parser TEXT NOT NULL,
                chunks_count INTEGER NOT NULL,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO rag_sources(source_id, active_version_id)
            VALUES ('legacy', 'rv_legacy');
            INSERT INTO rag_document_versions(
                version_id, source_id, source_name, content_sha256,
                pipeline_profile, parser, chunks_count, warnings_json, status
            ) VALUES (
                'rv_legacy', 'legacy', 'legacy.md',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'legacy-profile', 'builtin-text', 1, '[]', 'active'
            );
            """
        )
        db.commit()

    registry = SQLiteVersionRegistry(db_path)
    await registry.initialize()
    active = await registry.get_active("legacy")

    assert active is not None
    assert active.tenant_id == "public"
    assert active.acl == ()
    with sqlite3.connect(db_path) as db:
        source_columns = {row[1] for row in db.execute("PRAGMA table_info(rag_sources)")}
        version_columns = {row[1] for row in db.execute("PRAGMA table_info(rag_document_versions)")}
    assert "tenant_id" in source_columns
    assert {"tenant_id", "acl_json"} <= version_columns
    with sqlite3.connect(db_path) as db:
        scope = db.execute(
            "SELECT tenant_id, active_version_id FROM rag_source_scopes WHERE source_id = ?",
            ("legacy",),
        ).fetchone()
    assert scope == ("public", "rv_legacy")
    with sqlite3.connect(db_path) as db:
        migrated_version = db.execute(
            "SELECT tenant_id, source_id, status FROM rag_document_versions_v2 WHERE version_id = ?",
            ("rv_legacy",),
        ).fetchone()
        foreign_key_errors = db.execute("PRAGMA foreign_key_check").fetchall()
    assert migrated_version == ("public", "legacy", "active")
    assert foreign_key_errors == []


@pytest.mark.asyncio
async def test_registry_v2_uses_composite_identity_and_public_dual_write(tmp_path) -> None:
    db_path = tmp_path / "rag_registry.sqlite3"
    registry = SQLiteVersionRegistry(db_path)

    for tenant_id, version_id in (("tenant-a", "rv_a"), ("tenant-b", "rv_b")):
        await registry.record_building(
            source_id="shared",
            version_id=version_id,
            source_name=f"{tenant_id}.md",
            content_sha256=tenant_id.ljust(64, "0"),
            pipeline_profile="profile",
            parser="builtin-text",
            chunks_count=1,
            warnings=[],
            tenant_id=tenant_id,
        )
        await registry.activate(version_id)

    await registry.record_building(
        source_id="public-doc",
        version_id="rv_public",
        source_name="public.md",
        content_sha256="f" * 64,
        pipeline_profile="profile",
        parser="builtin-text",
        chunks_count=1,
        warnings=[],
    )
    await registry.activate("rv_public")

    assert (await registry.get_active("shared", "tenant-a")).version_id == "rv_a"  # type: ignore[union-attr]
    assert (await registry.get_active("shared", "tenant-b")).version_id == "rv_b"  # type: ignore[union-attr]
    with sqlite3.connect(db_path) as db:
        scoped_rows = db.execute(
            "SELECT tenant_id, source_id, version_id FROM rag_document_versions_v2 ORDER BY tenant_id"
        ).fetchall()
        legacy_public = db.execute(
            "SELECT tenant_id, source_id, version_id FROM rag_document_versions"
        ).fetchall()
        foreign_key_errors = db.execute("PRAGMA foreign_key_check").fetchall()
    assert scoped_rows == [
        ("public", "public-doc", "rv_public"),
        ("tenant-a", "shared", "rv_a"),
        ("tenant-b", "shared", "rv_b"),
    ]
    assert legacy_public == [("public", "public-doc", "rv_public")]
    assert foreign_key_errors == []


@pytest.mark.asyncio
async def test_registry_reconciles_public_writes_after_legacy_rollback(tmp_path) -> None:
    db_path = tmp_path / "rag_registry.sqlite3"
    registry = SQLiteVersionRegistry(db_path)
    await registry.record_building(
        source_id="guide",
        version_id="rv_v1",
        source_name="guide-v1.md",
        content_sha256="1" * 64,
        pipeline_profile="profile",
        parser="builtin-text",
        chunks_count=1,
        warnings=[],
    )
    await registry.activate("rv_v1")

    # Simulate a temporary rollback to the original public-only registry writer.
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            INSERT INTO rag_document_versions(
                version_id, source_id, source_name, content_sha256,
                pipeline_profile, parser, chunks_count, warnings_json,
                status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', 'active', '2099-01-01 00:00:00')
            """,
            ("rv_v2", "guide", "guide-v2.md", "2" * 64, "profile", "builtin-text", 1),
        )
        db.execute(
            """
            UPDATE rag_document_versions
            SET status = 'retired', updated_at = '2099-01-01 00:00:00'
            WHERE version_id = 'rv_v1'
            """
        )
        db.execute(
            """
            UPDATE rag_sources
            SET active_version_id = 'rv_v2', updated_at = '2099-01-01 00:00:00'
            WHERE source_id = 'guide'
            """
        )
        db.commit()

    upgraded = SQLiteVersionRegistry(db_path)
    active = await upgraded.get_active("guide")

    assert active is not None
    assert active.version_id == "rv_v2"
    with sqlite3.connect(db_path) as db:
        statuses = dict(
            db.execute(
                "SELECT version_id, status FROM rag_document_versions_v2 WHERE source_id = 'guide'"
            ).fetchall()
        )
        foreign_key_errors = db.execute("PRAGMA foreign_key_check").fetchall()
    assert statuses == {"rv_v1": "retired", "rv_v2": "active"}
    assert foreign_key_errors == []


@pytest.mark.asyncio
async def test_cross_tenant_retrieval_isolated_before_scoring(tmp_path) -> None:
    (tmp_path / "alpha.md").write_text("alpha tenant evidence", encoding="utf-8")
    (tmp_path / "beta.md").write_text("beta tenant evidence", encoding="utf-8")
    rag = build_rag(tmp_path)
    alpha = await rag.ingest(
        DocumentIngestRequest(file_path="alpha.md", source_id="alpha", tenant_id="tenant-a")
    )
    await rag.ingest(
        DocumentIngestRequest(file_path="beta.md", source_id="beta", tenant_id="tenant-b")
    )

    response = await rag.retrieve("tenant evidence", 10, tenant_id="tenant-a")
    explicit_other = await rag.retrieve(
        "beta tenant evidence", 10, source_ids=["beta"], tenant_id="tenant-a"
    )

    assert response.index_versions == {"alpha": alpha.version_id}
    assert response.hits
    assert all(hit.citation.source_id == "alpha" for hit in response.hits)
    assert explicit_other.index_versions == {}
    assert explicit_other.hits == []


@pytest.mark.asyncio
async def test_acl_requires_principal_intersection_and_hides_index_versions(tmp_path) -> None:
    (tmp_path / "public.md").write_text("tenant wide evidence", encoding="utf-8")
    (tmp_path / "restricted.md").write_text("engineering secret evidence", encoding="utf-8")
    rag = build_rag(tmp_path)
    public = await rag.ingest(
        DocumentIngestRequest(file_path="public.md", source_id="public-doc", tenant_id="tenant-a")
    )
    restricted = await rag.ingest(
        DocumentIngestRequest(
            file_path="restricted.md",
            source_id="restricted-doc",
            tenant_id="tenant-a",
            acl=["group:engineering"],
        )
    )

    anonymous = await rag.retrieve("evidence", 10, tenant_id="tenant-a")
    engineer = await rag.retrieve(
        "evidence", 10, tenant_id="tenant-a", principals=["group:engineering"]
    )
    finance = await rag.retrieve("evidence", 10, tenant_id="tenant-a", principals=["group:finance"])

    assert anonymous.index_versions == {"public-doc": public.version_id}
    assert {hit.citation.source_id for hit in anonymous.hits} == {"public-doc"}
    assert engineer.index_versions == {
        "public-doc": public.version_id,
        "restricted-doc": restricted.version_id,
    }
    assert {hit.citation.source_id for hit in engineer.hits} == {
        "public-doc",
        "restricted-doc",
    }
    assert finance.index_versions == {"public-doc": public.version_id}
    assert {hit.citation.source_id for hit in finance.hits} == {"public-doc"}


@pytest.mark.asyncio
async def test_restricted_active_version_suppresses_legacy_rows_for_unauthorized_principal(
    tmp_path,
) -> None:
    embeddings = HashEmbedding()
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3", embeddings)
    legacy_vector = (await embeddings.embed(["legacy secret should disappear"]))[0]
    await store.upsert(
        [
            VectorDocument(
                chunk_id="legacy:0",
                source_id="restricted",
                source_name="legacy.md",
                text="legacy secret should disappear",
                embedding=legacy_vector,
                tenant_id="tenant-a",
            )
        ]
    )
    (tmp_path / "replacement.md").write_text("new restricted secret", encoding="utf-8")
    rag = RAGService(
        tmp_path,
        CompositeDocumentParser(prefer_docling=False),
        SemanticChunker(target_chars=80, max_chars=120, overlap_chars=10),
        embeddings,
        store,
        SQLiteVersionRegistry(tmp_path / "rag_registry.sqlite3"),
    )
    await rag.ingest(
        DocumentIngestRequest(
            file_path="replacement.md",
            source_id="restricted",
            tenant_id="tenant-a",
            acl=["group:engineering"],
        )
    )

    unauthorized = await rag.retrieve("secret", 10, source_ids=["restricted"], tenant_id="tenant-a")
    authorized = await rag.retrieve(
        "secret",
        10,
        source_ids=["restricted"],
        tenant_id="tenant-a",
        principals=["group:engineering"],
    )

    assert unauthorized.index_versions == {}
    assert unauthorized.hits == []
    assert authorized.hits
    assert all("new restricted secret" in hit.text for hit in authorized.hits)


@pytest.mark.asyncio
async def test_same_source_id_is_isolated_per_tenant(tmp_path) -> None:
    (tmp_path / "first.md").write_text("first tenant evidence", encoding="utf-8")
    (tmp_path / "second.md").write_text("second tenant evidence", encoding="utf-8")
    rag = build_rag(tmp_path)
    first = await rag.ingest(
        DocumentIngestRequest(file_path="first.md", source_id="shared", tenant_id="tenant-a")
    )
    second = await rag.ingest(
        DocumentIngestRequest(file_path="second.md", source_id="shared", tenant_id="tenant-b")
    )

    tenant_a = await rag.retrieve("tenant evidence", 5, source_ids=["shared"], tenant_id="tenant-a")
    tenant_b = await rag.retrieve("tenant evidence", 5, source_ids=["shared"], tenant_id="tenant-b")

    assert first.version_id != second.version_id
    assert tenant_a.index_versions == {"shared": first.version_id}
    assert tenant_b.index_versions == {"shared": second.version_id}
    assert tenant_a.hits and all("first tenant" in hit.text for hit in tenant_a.hits)
    assert tenant_b.hits and all("second tenant" in hit.text for hit in tenant_b.hits)


@pytest.mark.asyncio
async def test_acl_change_creates_a_new_immutable_version(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("stable content", encoding="utf-8")
    rag = build_rag(tmp_path)
    unrestricted = await rag.ingest(
        DocumentIngestRequest(file_path="guide.md", source_id="guide", tenant_id="tenant-a")
    )
    restricted = await rag.ingest(
        DocumentIngestRequest(
            file_path="guide.md",
            source_id="guide",
            tenant_id="tenant-a",
            acl=["group:engineering", "group:engineering"],
        )
    )

    assert unrestricted.version_id != restricted.version_id
    assert restricted.idempotent is False
    unauthorized = await rag.retrieve("stable content", 5, tenant_id="tenant-a")
    authorized = await rag.retrieve(
        "stable content", 5, tenant_id="tenant-a", principals=["group:engineering"]
    )
    assert unauthorized.hits == []
    assert authorized.index_versions == {"guide": restricted.version_id}
    assert authorized.hits


@pytest.mark.asyncio
async def test_sqlite_authorization_filter_runs_before_vector_decoding(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3", HashEmbedding())
    await store.initialize()
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            """
            INSERT INTO document_chunks(
                chunk_id, source_id, source_name, tenant_id, acl_json,
                text, embedding_json, metadata_json
            ) VALUES (?, ?, ?, ?, '[]', ?, ?, '{}')
            """,
            (
                "broken:1",
                "other-tenant",
                "broken.md",
                "tenant-b",
                "must never be scored",
                "not-valid-json",
            ),
        )
        db.commit()

    hits, _ = await store.search("scored", top_k=5, tenant_id="tenant-a")

    assert hits == []


@pytest.mark.asyncio
async def test_acl_version_identity_uses_unambiguous_canonical_encoding(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("canonical authorization profile", encoding="utf-8")
    rag = build_rag(tmp_path)
    combined = await rag.ingest(
        DocumentIngestRequest(
            file_path="guide.md",
            source_id="guide",
            tenant_id="tenant-a",
            acl=["group:a,group:b"],
        )
    )
    split = await rag.ingest(
        DocumentIngestRequest(
            file_path="guide.md",
            source_id="guide",
            tenant_id="tenant-a",
            acl=["group:a", "group:b"],
        )
    )

    assert combined.version_id != split.version_id


@pytest.mark.asyncio
async def test_retrieval_degrades_when_one_scoring_branch_fails(tmp_path, monkeypatch) -> None:
    import agent_service.vector_store as vector_store_module

    rag = build_rag(tmp_path)
    (tmp_path / "guide.md").write_text(
        "Retry fallback evidence remains searchable.", encoding="utf-8"
    )
    await rag.ingest(DocumentIngestRequest(file_path="guide.md", source_id="guide"))

    def fail_vector_branch(*args, **kwargs):
        raise ValueError("simulated vector scorer failure")

    monkeypatch.setattr(vector_store_module, "_vector_scores", fail_vector_branch)
    response = await rag.retrieve("retry fallback evidence", 5, source_ids=["guide"])

    assert response.hits
    assert response.degraded_retrieval is True
    assert "degraded_retrieval:vector:ValueError" in response.warnings


def test_rrf_rank_filters_zero_score_candidates_and_respects_limit() -> None:
    scores = {"strong": 3.0, "weak": 1.0, "zero-a": 0.0, "zero-b": 0.0}

    ranks = _rank_scores(scores, limit=1, positive_only=True)

    assert ranks == {"strong": 1}
