# flake8: noqa
from pathlib import Path
from typing import Any

from haystack.dataclasses import Document

from hirag_haystack.core import (
    Entity,
    Relation,
    NodeType,
    Community,
    CommunitySchema,
    QueryParam,
    RetrievalMode,
)
from hirag_haystack.stores import (
    GraphDocumentStore,
    NetworkXGraphStore,
    Neo4jGraphStore,
    EntityVectorStore,
    ChunkVectorStore,
    KVStore,
    DocIdIndex,
)
from hirag_haystack.pipelines import HiRAGIndexingPipeline, HiRAGQueryPipeline
from hirag_haystack.components import (
    EntityExtractor,
    CommunityDetector,
    CommunityReportGenerator,
    HierarchicalEntityExtractor,
    HierarchicalClusterDetector,
    EntityRetriever,
    HierarchicalRetriever,
    ContextBuilder,
    PathFinder,
    PathScorer,
    GraphVisualizer,
)
from hirag_haystack.document_loader import DocumentLoader

__all__ = [
    # Core
    "Entity",
    "Relation",
    "NodeType",
    "Community",
    "CommunitySchema",
    "QueryParam",
    "RetrievalMode",
    # Stores
    "GraphDocumentStore",
    "NetworkXGraphStore",
    "Neo4jGraphStore",
    "EntityVectorStore",
    "ChunkVectorStore",
    "KVStore",
    "DocIdIndex",
    # Components
    "EntityExtractor",
    "CommunityDetector",
    "CommunityReportGenerator",
    "HierarchicalEntityExtractor",
    "HierarchicalClusterDetector",
    "EntityRetriever",
    "HierarchicalRetriever",
    "ContextBuilder",
    "PathFinder",
    "PathScorer",
    "GraphVisualizer",
    # Pipelines
    "HiRAGIndexingPipeline",
    "HiRAGQueryPipeline",
    # High-level API
    "HiRAG",
    # Document loading
    "DocumentLoader",
]

__version__ = "0.1.0"


class HiRAG:
    """High-level API for HiRAG.

    This class provides a simple interface for indexing and querying
    documents using the hierarchical knowledge approach.

    Example:
        ```python
        from hirag_haystack import HiRAG

        # Initialize
        hirag = HiRAG(working_dir="./hirag_data")

        # Index documents
        hirag.index("path/to/document.txt")

        # Query
        result = hirag.query("What are the main themes?")
        print(result["answer"])
        ```
    """

    def __init__(
        self,
        working_dir: str = "./hirag_cache",
        graph_backend: str = "networkx",
        generator: Any = None,
        entity_store: Any = None,
        chunk_store: Any = None,
        top_k: int = 20,
        top_m: int = 10,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        """Initialize HiRAG.

        Args:
            working_dir: Directory for storing cached data.
            graph_backend: Graph storage backend ("networkx" or "neo4j").
            generator: LLM generator for entity extraction and answer generation.
            entity_store: Vector store for entity embeddings.
            chunk_store: Document store for text chunks.
            top_k: Number of entities to retrieve.
            top_m: Key entities per community for path finding.
            chunk_size: Chunk size for document splitting.
            chunk_overlap: Overlap between chunks.
        """
        self.working_dir = working_dir
        self.top_k = top_k
        self.top_m = top_m
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize graph store
        if graph_backend == "networkx":
            self.graph_store = NetworkXGraphStore(
                namespace="hirag",
                working_dir=working_dir,
            )
        else:
            from hirag_haystack.stores.neo4j_store import Neo4jGraphStore
            self.graph_store = Neo4jGraphStore(
                namespace="hirag",
                working_dir=working_dir,
            )

        # Stores
        self.entity_store = entity_store
        self.chunk_store = chunk_store

        # Components
        self.generator = generator
        self.entity_extractor = EntityExtractor(generator=generator) if generator else None
        self.community_detector = CommunityDetector()
        self.report_generator = CommunityReportGenerator(generator=generator) if generator else None

        # Pipelines
        self.indexing_pipeline = HiRAGIndexingPipeline(
            graph_store=self.graph_store,
            document_store=chunk_store,
            entity_store=entity_store,
            chunk_store=chunk_store,
            entity_extractor=self.entity_extractor,
            community_detector=self.community_detector,
            report_generator=self.report_generator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.query_pipeline = HiRAGQueryPipeline(
            graph_store=self.graph_store,
            entity_store=entity_store,
            chunk_store=chunk_store,
            generator=generator,
            top_k=top_k,
            top_m=top_m,
        )

        # Initialize visualizer
        self.visualizer = GraphVisualizer(
            output_dir=str(Path(working_dir) / "visualizations")
        )

    def index(
        self,
        documents: list[str] | str | list[Document],
        doc_ids: list[str] | None = None,
        incremental: bool = False,
        force_reindex: bool = False,
    ) -> dict:
        """Index documents into the HiRAG system.

        Args:
            documents: Documents to index. Can be:
                - A string (treated as single document content)
                - A list of strings (each treated as document content)
                - A list of Haystack Document objects
            doc_ids: Optional list of external document IDs. Must match
                the number of documents. Enables delete/update by doc_id.
            incremental: If True, only index new documents (incremental update).
            force_reindex: If True, reindex all documents (ignores existing).

        Returns:
            Dictionary with indexing statistics.
        """
        if incremental:
            return self.indexing_pipeline.index_incremental(
                documents, force_reindex=force_reindex, doc_ids=doc_ids
            )
        return self.indexing_pipeline.index(documents, doc_ids=doc_ids)

    def query(
        self,
        query: str,
        mode: str = "hi",
        param: QueryParam | None = None,
    ) -> dict:
        """Query the HiRAG system.

        Args:
            query: User query string.
            mode: Retrieval mode ("naive", "hi_local", "hi_global", "hi_bridge",
                  "hi_nobridge", "hi").
            param: Optional QueryParam for detailed configuration.

        Returns:
            Dictionary with:
                - answer: Generated answer
                - context: Retrieved context
                - mode: Actual mode used
        """
        return self.query_pipeline.query(query, mode=mode, param=param)

    def query_local(self, query: str, param: QueryParam | None = None) -> dict:
        """Query using local context retrieval (entity-level).

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_local", param=param)

    def query_global(self, query: str, param: QueryParam | None = None) -> dict:
        """Query using global community reports.

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_global", param=param)

    def query_nobridge(self, query: str, param: QueryParam | None = None) -> dict:
        """Query using combined local and global knowledge without bridge paths.

        This mode combines entity-level and community-level retrieval
        without computing cross-community paths, which can be faster
        for large knowledge graphs.

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_nobridge", param=param)

    # ===== Document Management =====

    def delete(self, doc_ids: str | list[str]) -> dict:
        """Delete one or more documents by their external doc_ids.

        Removes all associated data: chunks, graph references,
        orphaned entities, and regenerates communities.

        Args:
            doc_ids: A single doc_id string or a list of doc_id strings.

        Returns:
            Dictionary with deletion statistics.
        """
        if isinstance(doc_ids, str):
            return self.indexing_pipeline.delete_document(doc_ids)
        return self.indexing_pipeline.delete_documents(doc_ids)

    def update(self, doc_id: str, content: str) -> dict:
        """Update a document by deleting and re-indexing it.

        Args:
            doc_id: The external document ID.
            content: New content for the document.

        Returns:
            Dictionary with update statistics.
        """
        return self.indexing_pipeline.update_document(doc_id, content)

    def list_documents(self) -> list[str]:
        """List all registered external document IDs.

        Returns:
            Sorted list of doc_id strings.
        """
        return self.indexing_pipeline.list_documents()

    def has_document(self, doc_id: str) -> bool:
        """Check if a document ID is registered.

        Args:
            doc_id: The external document ID.

        Returns:
            True if the doc_id exists, False otherwise.
        """
        return self.indexing_pipeline.has_document(doc_id)

    @property
    def communities(self) -> dict:
        """Get detected communities."""
        return self.graph_store._communities if hasattr(self.graph_store, "_communities") else {}

    @property
    def reports(self) -> dict:
        """Get community reports."""
        return self.graph_store._reports if hasattr(self.graph_store, "_reports") else {}

    def visualize(
        self,
        kind: str = "all",
        **kwargs
    ) -> dict[str, str]:
        """Generate visualizations for the knowledge graph.

        This is a convenience method that uses the internal GraphVisualizer
        to create interactive HTML visualizations.

        Args:
            kind: Type of visualization to generate:
                  - "graph": Knowledge graph only
                  - "communities": Community structure only
                  - "stats": Entity statistics only
                  - "all": Generate all visualizations (default)
            **kwargs: Additional arguments passed to specific visualizations.
                      Common options include:
                      - layout: "force", "hierarchical", "circular"
                      - color_by: "entity_type", "community", "degree"
                      - show_labels: bool
                      - physics: bool
                      - filter_min_degree: int
                      - filter_max_nodes: int

        Returns:
            Dictionary mapping visualization names to their HTML file paths.

        Example:
            ```python
            hirag = HiRAG(working_dir="./hirag_data")

            # Generate all visualizations
            results = hirag.visualize(kind="all")

            # Generate only knowledge graph
            kg_path = hirag.visualize(kind="graph", layout="force")
            ```
        """
        # Detect communities if not already done
        if not self.communities:
            self.graph_store._communities = self.graph_store.clustering()

        if kind == "graph":
            return {"graph": self.visualizer.visualize_knowledge_graph(
                graph_store=self.graph_store,
                **kwargs
            )}
        elif kind == "communities":
            return {"communities": self.visualizer.visualize_communities(
                communities=self.communities,
                graph_store=self.graph_store,
                **kwargs
            )}
        elif kind == "stats":
            return {"stats": self.visualizer.visualize_entity_stats(
                graph_store=self.graph_store,
                **kwargs
            )}
        elif kind == "all":
            return self.visualizer.visualize_all(
                graph_store=self.graph_store,
                communities=self.communities,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown visualization type: {kind}. "
                "Choose from: 'graph', 'communities', 'stats', 'all'"
            )
