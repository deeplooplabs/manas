"""Indexing pipeline for HiRAG.

This module implements the document indexing pipeline that:
1. Splits documents into chunks
2. Extracts entities and relationships
3. Builds the knowledge graph
4. Detects communities
5. Generates community reports
6. Supports incremental updates
"""

import json
from pathlib import Path
from typing import Any, Optional

from haystack import Pipeline, PreFetcher, component
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from hirag_haystack.components.entity_extractor import (
    EntityExtractor,
    EntityExtractionResult,
    merge_entities,
    merge_relations,
)
from hirag_haystack.components.community_detector import CommunityDetector
from hirag_haystack.components.report_generator import CommunityReportGenerator
from hirag_haystack.core.graph import Entity, Relation
from hirag_haystack.stores.base import GraphDocumentStore
from hirag_haystack.stores.networkx_store import NetworkXGraphStore
from hirag_haystack.stores.vector_store import EntityVectorStore, ChunkVectorStore, KVStore
from hirag_haystack.utils.token_utils import (
    compute_mdhash_id,
    count_tokens,
    truncate_string_by_token_size,
)


@component
class GraphIndexer:
    """Index entities and relations into the graph store."""

    def __init__(self, graph_store: GraphDocumentStore):
        """Initialize the graph indexer.

        Args:
            graph_store: The graph store to index into.
        """
        self.graph_store = graph_store

    @component.output_types(success=bool)
    def run(
        self,
        entities: list,
        relations: list,
    ) -> dict:
        """Index entities and relations into the graph store.

        Args:
            entities: List of Entity objects.
            relations: List of Relation objects.

        Returns:
            Dictionary indicating success.
        """
        # Merge with existing data
        existing_entities = {}
        for entity in entities:
            name = entity.entity_name
            if name not in existing_entities:
                # Check if already in graph
                existing_data = self.graph_store.get_node(name)
                if existing_data:
                    existing_entities[name] = entity.__class__(
                        entity_name=name,
                        entity_type=existing_data.get("entity_type", "UNKNOWN"),
                        description=existing_data.get("description", ""),
                        source_id=existing_data.get("source_id", ""),
                    )
                else:
                    existing_entities[name] = entity

        existing_relations = {}
        for relation in relations:
            key = relation.sorted_pair
            if key not in existing_relations:
                # Check if already in graph
                existing_data = self.graph_store.get_edge(key[0], key[1])
                if existing_data:
                    existing_relations[key] = relation.__class__(
                        src_id=key[0],
                        tgt_id=key[1],
                        weight=existing_data.get("weight", 1.0),
                        description=existing_data.get("description", ""),
                        source_id=existing_data.get("source_id", ""),
                    )
                else:
                    existing_relations[key] = relation

        # Upsert all entities
        for entity in existing_entities.values():
            self.graph_store.upsert_node(
                entity.entity_name,
                {
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "source_id": entity.source_id,
                    "clusters": json.dumps(entity.clusters),
                },
            )

        # Upsert all relations
        for relation in existing_relations.values():
            self.graph_store.upsert_edge(
                relation.src_id,
                relation.tgt_id,
                {
                    "weight": relation.weight,
                    "description": relation.description,
                    "source_id": relation.source_id,
                    "order": relation.order,
                },
            )

        return {"success": True}


class HiRAGIndexingPipeline:
    """Pipeline for indexing documents into HiRAG.

    This pipeline orchestrates the full indexing process:
    1. Document splitting
    2. Entity extraction
    3. Graph construction
    4. Community detection
    5. Report generation
    6. Incremental updates (optional)
    """

    def __init__(
        self,
        graph_store: GraphDocumentStore | None = None,
        document_store: Any = None,
        entity_extractor: EntityExtractor | None = None,
        community_detector: CommunityDetector | None = None,
        report_generator: CommunityReportGenerator | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        working_dir: str = "./hirag_cache",
        enable_incremental: bool = True,
    ):
        """Initialize the indexing pipeline.

        Args:
            graph_store: Graph store for knowledge graph.
            document_store: Document store for text chunks.
            entity_extractor: Component for extracting entities and relations.
            community_detector: Component for detecting communities.
            report_generator: Component for generating community reports.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks.
            working_dir: Directory for storing cache and index data.
            enable_incremental: Whether to support incremental updates.
        """
        self.graph_store = graph_store or NetworkXGraphStore(
            namespace="hirag",
            working_dir=working_dir,
        )
        self.document_store = document_store
        self.entity_extractor = entity_extractor
        self.community_detector = community_detector or CommunityDetector()
        self.report_generator = report_generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.working_dir = working_dir
        self.enable_incremental = enable_incremental

        # Initialize stores for incremental indexing
        self.full_docs_store = KVStore(
            namespace="full_docs",
            working_dir=working_dir,
        )
        self.text_chunks_store = KVStore(
            namespace="text_chunks",
            working_dir=working_dir,
        )

        # Vector stores
        self.entity_vector_store = None
        self.chunk_vector_store = None

        # Communities storage
        self._communities: dict = {}
        self._reports: dict = {}

        # Load existing data
        self._load_state()

    def index(
        self,
        documents: list[Document] | list[str] | str,
    ) -> dict:
        """Index documents into the HiRAG system.

        Args:
            documents: Documents to index. Can be:
                - List of Document objects
                - List of strings (treated as document content)
                - Single string (treated as single document content)

        Returns:
            Dictionary with indexing statistics.
        """
        # Normalize input to Document objects
        docs = self._normalize_documents(documents)

        if not docs:
            return {"status": "no_documents", "count": 0}

        # Step 1: Split documents into chunks
        chunks = self._split_documents(docs)

        # Step 2: Extract entities and relations
        if self.entity_extractor:
            extraction_result = self.entity_extractor.run(documents=chunks)
            entities = extraction_result.get("entities", [])
            relations = extraction_result.get("relations", [])
        else:
            entities = []
            relations = []

        # Step 3: Index into graph store
        graph_indexer = GraphIndexer(self.graph_store)
        graph_indexer.run(entities=entities, relations=relations)

        # Step 4: Detect communities
        communities_result = self.community_detector.run(graph_store=self.graph_store)
        self._communities = communities_result.get("communities", {})

        # Step 5: Generate reports
        if self.report_generator and self._communities:
            reports_result = self.report_generator.run(
                graph_store=self.graph_store,
                communities=self._communities,
            )
            self._reports = reports_result.get("reports", {})

        # Save to disk
        self.graph_store.index_done_callback()

        return {
            "status": "success",
            "documents_count": len(docs),
            "chunks_count": len(chunks),
            "entities_count": len(entities),
            "relations_count": len(relations),
            "communities_count": len(self._communities),
        }

    def _normalize_documents(
        self,
        documents: list[Document] | list[str] | str,
    ) -> list[Document]:
        """Normalize various input formats to Document objects."""
        if isinstance(documents, str):
            documents = [documents]

        result = []
        for doc in documents:
            if isinstance(doc, Document):
                result.append(doc)
            elif isinstance(doc, str):
                result.append(Document(content=doc))
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")

        return result

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        # Simple character-based splitting (in production, use tokenizer)
        chunks = []
        chunk_size_chars = self.chunk_size * 4  # Rough estimate: 4 chars per token

        for doc_idx, doc in enumerate(documents):
            content = doc.content
            meta = doc.meta or {}
            meta["source_doc_id"] = doc.id or f"doc_{doc_idx}"

            for i in range(0, len(content), chunk_size_chars - self.chunk_overlap):
                chunk_content = content[i:i + chunk_size_chars]
                chunk_meta = meta.copy()
                chunk_meta["chunk_index"] = i // (chunk_size_chars - self.chunk_overlap)
                chunk_meta["chunk_order_index"] = chunk_meta["chunk_index"]

                chunks.append(Document(
                    content=chunk_content,
                    meta=chunk_meta,
                    id=f"{doc.id}_chunk_{chunk_meta['chunk_index']}" if doc.id else f"chunk_{len(chunks)}",
                ))

        return chunks

    @property
    def communities(self) -> dict:
        """Get detected communities."""
        return self._communities

    @property
    def reports(self) -> dict:
        """Get generated community reports."""
        return self._reports

    def _load_state(self) -> None:
        """Load existing state from disk."""
        # Load communities
        comm_store = KVStore("communities", self.working_dir)
        self._communities = comm_store.get_all()

        # Load reports
        report_store = KVStore("reports", self.working_dir)
        self._reports = report_store.get_all()

    def index_incremental(
        self,
        documents: list[Document] | list[str] | str,
        force_reindex: bool = False,
    ) -> dict:
        """Index documents with incremental update support.

        Only processes new documents that haven't been indexed before.
        For truly new content, updates communities and reports incrementally.

        Args:
            documents: Documents to index.
            force_reindex: If True, reindex all documents (ignores existing).

        Returns:
            Dictionary with indexing statistics.
        """
        # Normalize to Document objects
        docs = self._normalize_documents(documents)

        if not docs:
            return {"status": "no_documents", "count": 0}

        # Compute hashes for deduplication
        new_docs = {}
        for doc in docs:
            content = doc.content.strip()
            doc_hash = compute_mdhash_id(content, prefix="doc-")

            if not force_reindex:
                existing = self.full_docs_store.get(doc_hash)
                if existing:
                    continue  # Skip already indexed document

            new_docs[doc_hash] = {"content": content}

        if not new_docs:
            return {"status": "all_existing", "new_count": 0, "count": len(docs)}

        # Split into chunks
        chunks = self._split_documents_batch(new_docs)

        # Filter existing chunks
        chunk_hashes = list(chunks.keys())
        new_chunk_hashes = self.text_chunks_store.filter_keys(chunk_hashes)
        new_chunks = {k: v for k, v in chunks.items() if k in new_chunk_hashes}

        if not new_chunks:
            return {"status": "all_existing", "new_count": len(new_docs), "chunks_count": 0}

        # Extract entities and relations
        if self.entity_extractor:
            chunk_docs = [
                Document(id=k, content=v.get("content", ""), meta=v)
                for k, v in new_chunks.items()
            ]
            extraction_result = self.entity_extractor.run(documents=chunk_docs)
            entities = extraction_result.get("entities", [])
            relations = extraction_result.get("relations", [])
        else:
            entities = []
            relations = []

        # Index into graph
        graph_indexer = GraphIndexer(self.graph_store)
        graph_indexer.run(entities=entities, relations=relations)

        # Update communities (drop and regenerate for simplicity)
        # TODO: Implement true incremental community update
        self.community_detector.run(graph_store=self.graph_store)
        self._communities = self.graph_store.community_schema()

        # Update reports
        if self.report_generator and self._communities:
            self.report_generator.run(
                graph_store=self.graph_store,
                communities=self._communities,
            )
            # Get reports from generator
            if hasattr(self.report_generator, "reports"):
                self._reports = self.report_generator.reports

        # Store documents and chunks
        self.full_docs_store.set_batch(new_docs)
        self.text_chunks_store.set_batch(new_chunks)

        # Update entity vectors if available
        if self.entity_vector_store and entities:
            entity_data = {}
            for entity in entities:
                entity_hash = compute_mdhash_id(entity.entity_name, prefix="ent-")
                entity_data[entity_hash] = {
                    "content": entity.entity_name + " " + entity.description,
                    "entity_name": entity.entity_name,
                }
            self.entity_vector_store.upsert(entity_data)

        # Save state
        self._save_state()
        self.graph_store.save_to_disk()

        return {
            "status": "success",
            "new_documents": len(new_docs),
            "new_chunks": len(new_chunks),
            "entities_count": len(entities),
            "relations_count": len(relations),
            "communities_count": len(self._communities),
        }

    def _split_documents_batch(
        self,
        docs: dict,
    ) -> dict:
        """Split a batch of documents into chunks.

        Args:
            docs: Dictionary mapping doc_hash to doc data.

        Returns:
            Dictionary mapping chunk_hash to chunk data.
        """
        chunks = {}
        chunk_size_chars = self.chunk_size * 4

        for doc_idx, (doc_hash, doc_data) in enumerate(docs.items()):
            content = doc_data["content"]

            for i in range(0, len(content), chunk_size_chars - self.chunk_overlap):
                chunk_content = content[i:i + chunk_size_chars]
                chunk_hash = compute_mdhash_id(
                    f"{doc_hash}_{i}",
                    prefix="chunk-"
                )

                chunks[chunk_hash] = {
                    "content": chunk_content,
                    "tokens": count_tokens(chunk_content),
                    "chunk_order_index": i // (chunk_size_chars - self.chunk_overlap),
                    "full_doc_id": doc_hash,
                }

        return chunks

    def _save_state(self) -> None:
        """Save current state to disk."""
        # Save communities
        comm_store = KVStore("communities", self.working_dir)
        comm_store.set_batch(self._communities)
        comm_store.save_to_disk()

        # Save reports
        report_store = KVStore("reports", self.working_dir)
        report_store.set_batch(self._reports)
        report_store.save_to_disk()

        # Save stores
        self.full_docs_store.save_to_disk()
        self.text_chunks_store.save_to_disk()

        if self.entity_vector_store:
            self.entity_vector_store.save_to_disk()
        if self.chunk_vector_store:
            self.chunk_vector_store.save_to_disk()


def build_indexing_pipeline(
    graph_store: GraphDocumentStore,
    generator: Any = None,
    document_store: Any = None,
) -> Pipeline:
    """Build a Haystack Pipeline for HiRAG indexing.

    Args:
        graph_store: Graph store for knowledge graph.
        generator: LLM generator for entity/report generation.
        document_store: Optional document store for chunks.

    Returns:
        Configured Haystack Pipeline.
    """
    pipeline = Pipeline()

    # Add components
    splitter = DocumentSplitter(
        split_by="word",
        split_length=300,
        split_overlap=50,
    )
    entity_extractor = EntityExtractor(generator=generator)
    community_detector = CommunityDetector()
    report_generator = CommunityReportGenerator(generator=generator)

    # For Haystack pipeline, we'd need to wrap these properly
    # This is a simplified version for demonstration

    return pipeline
