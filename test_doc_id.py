"""Tests for external doc_id support in HiRAG.

Tests cover: DocIdIndex, graph store delete operations,
indexing pipeline doc_id tracking, HiRAG facade methods,
and project_id isolation.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from hirag_haystack import HiRAG
from hirag_haystack.stores.vector_store import DocIdIndex, ChunkVectorStore, EntityVectorStore, KVStore
from hirag_haystack.stores.networkx_store import NetworkXGraphStore
from hirag_haystack.pipelines.indexing import HiRAGIndexingPipeline


# ===== Fixtures =====


@pytest.fixture
def tmp_dir():
    """Create a temporary working directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def doc_id_index(tmp_dir):
    return DocIdIndex(working_dir=tmp_dir)


@pytest.fixture
def graph_store(tmp_dir):
    return NetworkXGraphStore(namespace="test", working_dir=tmp_dir)


@pytest.fixture
def chunk_store(tmp_dir):
    return ChunkVectorStore(working_dir=tmp_dir)


@pytest.fixture
def entity_store(tmp_dir):
    return EntityVectorStore(working_dir=tmp_dir)


@pytest.fixture
def pipeline(tmp_dir, graph_store, chunk_store, entity_store):
    """Create a pipeline without LLM (entity extractor=None)."""
    return HiRAGIndexingPipeline(
        graph_store=graph_store,
        chunk_store=chunk_store,
        entity_store=entity_store,
        entity_extractor=None,
        report_generator=None,
        working_dir=tmp_dir,
    )


# ===== DocIdIndex Unit Tests =====


class TestDocIdIndex:
    def test_add_and_get_chunks(self, doc_id_index):
        doc_id_index.add_chunks("doc1", ["c1", "c2"])
        assert doc_id_index.get_chunks("doc1") == ["c1", "c2"]

    def test_add_chunks_deduplicates(self, doc_id_index):
        doc_id_index.add_chunks("doc1", ["c1", "c2"])
        doc_id_index.add_chunks("doc1", ["c2", "c3"])
        assert doc_id_index.get_chunks("doc1") == ["c1", "c2", "c3"]

    def test_add_and_get_entities(self, doc_id_index):
        doc_id_index.add_entities("doc1", ["entity_a", "entity_b"])
        assert doc_id_index.get_entities("doc1") == ["entity_a", "entity_b"]

    def test_add_entities_deduplicates(self, doc_id_index):
        doc_id_index.add_entities("doc1", ["entity_a"])
        doc_id_index.add_entities("doc1", ["entity_a", "entity_b"])
        assert doc_id_index.get_entities("doc1") == ["entity_a", "entity_b"]

    def test_list_doc_ids(self, doc_id_index):
        doc_id_index.add_chunks("doc2", ["c1"])
        doc_id_index.add_chunks("doc1", ["c2"])
        assert doc_id_index.list_doc_ids() == ["doc1", "doc2"]

    def test_has_doc(self, doc_id_index):
        assert not doc_id_index.has_doc("doc1")
        doc_id_index.add_chunks("doc1", ["c1"])
        assert doc_id_index.has_doc("doc1")

    def test_remove_doc(self, doc_id_index):
        doc_id_index.add_chunks("doc1", ["c1"])
        doc_id_index.add_entities("doc1", ["ent1"])
        doc_id_index.remove_doc("doc1")
        assert not doc_id_index.has_doc("doc1")
        assert doc_id_index.get_chunks("doc1") == []
        assert doc_id_index.get_entities("doc1") == []

    def test_remove_nonexistent_doc(self, doc_id_index):
        # Should not raise
        doc_id_index.remove_doc("nonexistent")

    def test_all_entity_doc_ids(self, doc_id_index):
        doc_id_index.add_entities("doc1", ["shared_ent", "ent1"])
        doc_id_index.add_entities("doc2", ["shared_ent", "ent2"])
        assert sorted(doc_id_index.all_entity_doc_ids("shared_ent")) == ["doc1", "doc2"]
        assert doc_id_index.all_entity_doc_ids("ent1") == ["doc1"]

    def test_persistence(self, tmp_dir):
        idx = DocIdIndex(working_dir=tmp_dir)
        idx.add_chunks("doc1", ["c1", "c2"])
        idx.add_entities("doc1", ["ent1"])
        idx.save_to_disk()

        idx2 = DocIdIndex(working_dir=tmp_dir)
        assert idx2.get_chunks("doc1") == ["c1", "c2"]
        assert idx2.get_entities("doc1") == ["ent1"]
        assert idx2.has_doc("doc1")

    def test_get_missing_doc_returns_empty(self, doc_id_index):
        assert doc_id_index.get_chunks("missing") == []
        assert doc_id_index.get_entities("missing") == []


# ===== NetworkXGraphStore Delete Tests =====


class TestNetworkXGraphStoreDelete:
    def test_delete_node(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "TEST", "source_id": "c1"})
        assert graph_store.has_node("A")
        graph_store.delete_node("A")
        assert not graph_store.has_node("A")

    def test_delete_node_removes_edges(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_node("B", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_edge("A", "B", {"weight": 1.0, "source_id": "c1"})
        assert graph_store.has_edge("A", "B")
        graph_store.delete_node("A")
        assert not graph_store.has_node("A")
        assert not graph_store.has_edge("A", "B")

    def test_delete_edge(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_node("B", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_edge("A", "B", {"weight": 1.0, "source_id": "c1"})
        graph_store.delete_edge("A", "B")
        assert not graph_store.has_edge("A", "B")
        # Nodes should still exist
        assert graph_store.has_node("A")
        assert graph_store.has_node("B")

    def test_get_all_edges(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_node("B", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_edge("A", "B", {"weight": 1.0, "source_id": "c1"})
        edges = graph_store.get_all_edges()
        assert ("A", "B") in edges

    def test_remove_source_from_node_keeps_if_other_sources(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1|c2"})
        deleted = graph_store.remove_source_from_node("A", "c1")
        assert not deleted
        assert graph_store.has_node("A")
        node = graph_store.get_node("A")
        assert node["source_id"] == "c2"

    def test_remove_source_from_node_deletes_if_last(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        deleted = graph_store.remove_source_from_node("A", "c1")
        assert deleted
        assert not graph_store.has_node("A")

    def test_remove_source_from_edge_keeps_if_other_sources(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_node("B", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_edge("A", "B", {"weight": 1.0, "source_id": "c1|c2"})
        deleted = graph_store.remove_source_from_edge("A", "B", "c1")
        assert not deleted
        assert graph_store.has_edge("A", "B")
        edge = graph_store.get_edge("A", "B")
        assert edge["source_id"] == "c2"

    def test_remove_source_from_edge_deletes_if_last(self, graph_store):
        graph_store.upsert_node("A", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_node("B", {"entity_type": "T", "source_id": "c1"})
        graph_store.upsert_edge("A", "B", {"weight": 1.0, "source_id": "c1"})
        deleted = graph_store.remove_source_from_edge("A", "B", "c1")
        assert deleted
        assert not graph_store.has_edge("A", "B")

    def test_delete_nonexistent_node(self, graph_store):
        # Should not raise
        graph_store.delete_node("nonexistent")

    def test_delete_nonexistent_edge(self, graph_store):
        # Should not raise
        graph_store.delete_edge("X", "Y")


# ===== Pipeline Doc ID Tests =====


class TestPipelineDocId:
    def test_index_with_doc_ids(self, pipeline):
        result = pipeline.index(
            documents=["Content about AI.", "Content about ML."],
            doc_ids=["doc_ai", "doc_ml"],
        )
        assert result["status"] == "success"
        assert result["documents_count"] == 2
        assert pipeline.has_document("doc_ai")
        assert pipeline.has_document("doc_ml")
        assert sorted(pipeline.list_documents()) == ["doc_ai", "doc_ml"]

    def test_doc_ids_length_mismatch_raises(self, pipeline):
        with pytest.raises(ValueError, match="doc_ids length"):
            pipeline.index(
                documents=["Content 1", "Content 2"],
                doc_ids=["only_one"],
            )

    def test_index_without_doc_ids(self, pipeline):
        result = pipeline.index(documents=["Some content"])
        assert result["status"] == "success"
        # No doc_id tracking when doc_ids not provided
        assert pipeline.list_documents() == []

    def test_delete_document(self, pipeline):
        pipeline.index(
            documents=["Delete me content."],
            doc_ids=["to_delete"],
        )
        assert pipeline.has_document("to_delete")

        result = pipeline.delete_document("to_delete")
        assert result["status"] == "deleted"
        assert not pipeline.has_document("to_delete")
        assert pipeline.list_documents() == []

    def test_delete_nonexistent_document(self, pipeline):
        result = pipeline.delete_document("no_such_doc")
        assert result["status"] == "not_found"

    def test_delete_multiple_documents(self, pipeline):
        pipeline.index(
            documents=["Doc A content", "Doc B content", "Doc C content"],
            doc_ids=["a", "b", "c"],
        )
        result = pipeline.delete_documents(["a", "c"])
        assert result["total_deleted"] == 2
        assert pipeline.list_documents() == ["b"]

    def test_update_document(self, pipeline):
        pipeline.index(
            documents=["Original content."],
            doc_ids=["updatable"],
        )
        assert pipeline.has_document("updatable")

        result = pipeline.update_document("updatable", "Updated content.")
        assert result["status"] == "updated"
        assert pipeline.has_document("updatable")
        # doc should still be in the list
        assert pipeline.list_documents() == ["updatable"]

    def test_list_documents(self, pipeline):
        assert pipeline.list_documents() == []
        pipeline.index(documents=["A"], doc_ids=["d1"])
        pipeline.index(documents=["B"], doc_ids=["d2"])
        assert sorted(pipeline.list_documents()) == ["d1", "d2"]

    def test_has_document(self, pipeline):
        assert not pipeline.has_document("x")
        pipeline.index(documents=["Content"], doc_ids=["x"])
        assert pipeline.has_document("x")


# ===== Integration: Shared Entity Not Deleted =====


class TestSharedEntitySurvival:
    def test_shared_entity_survives_deletion(self, tmp_dir):
        """When two docs share an entity, deleting one doc should not remove the entity."""
        graph_store = NetworkXGraphStore(namespace="test", working_dir=tmp_dir)
        chunk_store = ChunkVectorStore(working_dir=tmp_dir)
        entity_store = EntityVectorStore(working_dir=tmp_dir)

        pipeline = HiRAGIndexingPipeline(
            graph_store=graph_store,
            chunk_store=chunk_store,
            entity_store=entity_store,
            entity_extractor=None,
            report_generator=None,
            working_dir=tmp_dir,
        )

        # Simulate indexing two docs with a shared entity
        pipeline.index(documents=["Doc about Python."], doc_ids=["doc1"])
        pipeline.index(documents=["Doc about Java."], doc_ids=["doc2"])

        # Manually add a shared entity to graph + index
        # (since no entity_extractor, we simulate directly)
        chunk_ids_1 = pipeline.doc_id_index.get_chunks("doc1")
        chunk_ids_2 = pipeline.doc_id_index.get_chunks("doc2")

        source_id_1 = chunk_ids_1[0] if chunk_ids_1 else "fake_c1"
        source_id_2 = chunk_ids_2[0] if chunk_ids_2 else "fake_c2"

        graph_store.upsert_node("SHARED_ENTITY", {
            "entity_type": "CONCEPT",
            "description": "A shared entity",
            "source_id": f"{source_id_1}|{source_id_2}",
        })
        graph_store.upsert_node("DOC1_ONLY", {
            "entity_type": "CONCEPT",
            "description": "Only in doc1",
            "source_id": source_id_1,
        })

        pipeline.doc_id_index.add_entities("doc1", ["SHARED_ENTITY", "DOC1_ONLY"])
        pipeline.doc_id_index.add_entities("doc2", ["SHARED_ENTITY"])
        pipeline.doc_id_index.save_to_disk()

        # Delete doc1
        pipeline.delete_document("doc1")

        # SHARED_ENTITY should still exist (referenced by doc2's chunk)
        assert graph_store.has_node("SHARED_ENTITY")
        # DOC1_ONLY should be deleted (only referenced by doc1's chunk)
        assert not graph_store.has_node("DOC1_ONLY")


# ===== Persistence After Delete =====


class TestPersistenceAfterDelete:
    def test_index_persists_after_delete(self, tmp_dir):
        pipeline = HiRAGIndexingPipeline(
            graph_store=NetworkXGraphStore(namespace="test", working_dir=tmp_dir),
            chunk_store=ChunkVectorStore(working_dir=tmp_dir),
            entity_store=EntityVectorStore(working_dir=tmp_dir),
            entity_extractor=None,
            report_generator=None,
            working_dir=tmp_dir,
        )

        pipeline.index(documents=["Keep me", "Delete me"], doc_ids=["keep", "del"])
        pipeline.delete_document("del")

        # Reload from disk
        idx = DocIdIndex(working_dir=tmp_dir)
        assert idx.has_doc("keep")
        assert not idx.has_doc("del")


# ===== ChunkVectorStore.get_chunks_by_doc_id =====


class TestChunkVectorStoreDocId:
    def test_get_chunks_by_doc_id(self, chunk_store):
        chunk_store.upsert({
            "c1": {"content": "hello", "full_doc_id": "doc1", "chunk_order_index": 0, "tokens": 1},
            "c2": {"content": "world", "full_doc_id": "doc1", "chunk_order_index": 1, "tokens": 1},
            "c3": {"content": "other", "full_doc_id": "doc2", "chunk_order_index": 0, "tokens": 1},
        })
        results = chunk_store.get_chunks_by_doc_id("doc1")
        ids = {r["id"] for r in results}
        assert ids == {"c1", "c2"}

    def test_get_chunks_by_doc_id_empty(self, chunk_store):
        assert chunk_store.get_chunks_by_doc_id("nonexistent") == []


# ===== Project ID Isolation Tests =====


class TestProjectIdIsolation:
    """Tests for project_id-based data isolation in HiRAG facade."""

    @pytest.fixture
    def hirag(self, tmp_dir):
        return HiRAG(working_dir=tmp_dir)

    def test_project_id_isolation(self, hirag):
        """Index into two projects and verify each only sees its own docs."""
        hirag.index(["AI content"], doc_ids=["d1"], project_id="proj_a")
        hirag.index(["ML content"], doc_ids=["d2"], project_id="proj_b")

        assert hirag.list_documents(project_id="proj_a") == ["d1"]
        assert hirag.list_documents(project_id="proj_b") == ["d2"]

        assert hirag.has_document("d1", project_id="proj_a")
        assert not hirag.has_document("d1", project_id="proj_b")
        assert hirag.has_document("d2", project_id="proj_b")
        assert not hirag.has_document("d2", project_id="proj_a")

    def test_default_project_id(self, hirag):
        """Omitting project_id uses 'default'."""
        hirag.index(["Default content"], doc_ids=["d0"])
        assert hirag.list_documents() == ["d0"]
        assert hirag.list_documents(project_id="default") == ["d0"]
        assert hirag.has_document("d0")
        assert hirag.has_document("d0", project_id="default")

    def test_default_does_not_leak_to_other_project(self, hirag):
        """Default project data is not visible in a named project."""
        hirag.index(["Default content"], doc_ids=["d0"])
        assert hirag.list_documents(project_id="other") == []
        assert not hirag.has_document("d0", project_id="other")

    def test_delete_with_project_id(self, hirag):
        """Delete in one project does not affect another."""
        hirag.index(["Content A"], doc_ids=["d1"], project_id="proj_a")
        hirag.index(["Content B"], doc_ids=["d1"], project_id="proj_b")

        hirag.delete("d1", project_id="proj_a")

        assert not hirag.has_document("d1", project_id="proj_a")
        assert hirag.has_document("d1", project_id="proj_b")

    def test_project_pipelines_are_cached(self, hirag):
        """Accessing the same project_id twice returns the same pipeline objects."""
        p1 = hirag._get_project("cached_proj")
        p2 = hirag._get_project("cached_proj")
        assert p1[0] is p2[0]
        assert p1[1] is p2[1]
        assert p1[2] is p2[2]

    def test_backward_compat_aliases(self, hirag):
        """self.indexing_pipeline, query_pipeline, graph_store point to default project."""
        default = hirag._get_project("default")
        assert hirag.indexing_pipeline is default[0]
        assert hirag.query_pipeline is default[1]
        assert hirag.graph_store is default[2]

    def test_project_uses_separate_directories(self, hirag):
        """Each project stores data under {working_dir}/{project_id}/."""
        hirag.index(["A"], doc_ids=["d1"], project_id="alpha")
        hirag.index(["B"], doc_ids=["d2"], project_id="beta")

        alpha_dir = Path(hirag.working_dir) / "alpha"
        beta_dir = Path(hirag.working_dir) / "beta"
        assert alpha_dir.exists()
        assert beta_dir.exists()
