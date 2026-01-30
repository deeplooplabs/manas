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

    Supports ``project_id`` for full data isolation – each project gets
    its own sub-directory with independent stores and pipelines.

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

        # Multi-project usage
        hirag.index(["AI content"], doc_ids=["d1"], project_id="proj_a")
        hirag.query("What is AI?", project_id="proj_a")
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
        self.graph_backend = graph_backend
        self.generator = generator
        self.entity_store = entity_store
        self.chunk_store = chunk_store
        self.top_k = top_k
        self.top_m = top_m
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Per-project pipeline cache: project_id -> (indexing, query, graph_store)
        self._project_pipelines: dict[str, tuple] = {}

        # Pre-create the "default" project for backward compatibility
        default = self._create_project_pipelines("default")
        self._project_pipelines["default"] = default

        # Backward-compatible aliases pointing to the default project
        self.indexing_pipeline = default[0]
        self.query_pipeline = default[1]
        self.graph_store = default[2]

        # Initialize visualizer
        self.visualizer = GraphVisualizer(
            output_dir=str(Path(working_dir) / "visualizations")
        )

    # ===== Project Pipeline Management =====

    def _create_project_pipelines(
        self, project_id: str
    ) -> tuple:
        """Create isolated stores and pipelines for a project.

        Args:
            project_id: Project identifier used for directory and namespace isolation.

        Returns:
            Tuple of (indexing_pipeline, query_pipeline, graph_store).
        """
        project_dir = str(Path(self.working_dir) / project_id)
        namespace = f"hirag_{project_id}"

        # Graph store
        if self.graph_backend == "networkx":
            graph_store = NetworkXGraphStore(
                namespace=namespace,
                working_dir=project_dir,
            )
        else:
            from hirag_haystack.stores.neo4j_store import Neo4jGraphStore
            graph_store = Neo4jGraphStore(
                namespace=namespace,
                working_dir=project_dir,
            )

        # Components
        entity_extractor = (
            EntityExtractor(generator=self.generator) if self.generator else None
        )
        community_detector = CommunityDetector()
        report_generator = (
            CommunityReportGenerator(generator=self.generator) if self.generator else None
        )

        # Pipelines
        indexing_pipeline = HiRAGIndexingPipeline(
            graph_store=graph_store,
            document_store=self.chunk_store,
            entity_store=self.entity_store,
            chunk_store=self.chunk_store,
            entity_extractor=entity_extractor,
            community_detector=community_detector,
            report_generator=report_generator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            working_dir=project_dir,
        )

        query_pipeline = HiRAGQueryPipeline(
            graph_store=graph_store,
            entity_store=self.entity_store,
            chunk_store=self.chunk_store,
            generator=self.generator,
            top_k=self.top_k,
            top_m=self.top_m,
        )

        return (indexing_pipeline, query_pipeline, graph_store)

    def _get_project(self, project_id: str) -> tuple:
        """Return cached (indexing_pipeline, query_pipeline, graph_store) for a project.

        Lazily creates and caches pipelines on first access.

        Args:
            project_id: Project identifier.

        Returns:
            Tuple of (indexing_pipeline, query_pipeline, graph_store).
        """
        if project_id not in self._project_pipelines:
            self._project_pipelines[project_id] = self._create_project_pipelines(project_id)
        return self._project_pipelines[project_id]

    def index(
        self,
        documents: list[str] | str | list[Document],
        doc_ids: list[str] | None = None,
        project_id: str = "default",
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
            project_id: Project identifier for data isolation (default: "default").
            incremental: If True, only index new documents (incremental update).
            force_reindex: If True, reindex all documents (ignores existing).

        Returns:
            Dictionary with indexing statistics.
        """
        indexing_pipeline = self._get_project(project_id)[0]
        if incremental:
            return indexing_pipeline.index_incremental(
                documents, force_reindex=force_reindex, doc_ids=doc_ids
            )
        return indexing_pipeline.index(documents, doc_ids=doc_ids)

    def query(
        self,
        query: str,
        mode: str = "hi",
        param: QueryParam | None = None,
        project_id: str = "default",
    ) -> dict:
        """Query the HiRAG system.

        Args:
            query: User query string.
            mode: Retrieval mode ("naive", "hi_local", "hi_global", "hi_bridge",
                  "hi_nobridge", "hi").
            param: Optional QueryParam for detailed configuration.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with:
                - answer: Generated answer
                - context: Retrieved context
                - mode: Actual mode used
        """
        query_pipeline = self._get_project(project_id)[1]
        return query_pipeline.query(query, mode=mode, param=param)

    def query_local(
        self, query: str, param: QueryParam | None = None, project_id: str = "default"
    ) -> dict:
        """Query using local context retrieval (entity-level).

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_local", param=param, project_id=project_id)

    def query_global(
        self, query: str, param: QueryParam | None = None, project_id: str = "default"
    ) -> dict:
        """Query using global community reports.

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_global", param=param, project_id=project_id)

    def query_nobridge(
        self, query: str, param: QueryParam | None = None, project_id: str = "default"
    ) -> dict:
        """Query using combined local and global knowledge without bridge paths.

        This mode combines entity-level and community-level retrieval
        without computing cross-community paths, which can be faster
        for large knowledge graphs.

        Args:
            query: User query string.
            param: Optional QueryParam for detailed configuration.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with answer and context.
        """
        return self.query(query, mode="hi_nobridge", param=param, project_id=project_id)

    # ===== Document Management =====

    def delete(self, doc_ids: str | list[str], project_id: str = "default") -> dict:
        """Delete one or more documents by their external doc_ids.

        Removes all associated data: chunks, graph references,
        orphaned entities, and regenerates communities.

        Args:
            doc_ids: A single doc_id string or a list of doc_id strings.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with deletion statistics.
        """
        indexing_pipeline = self._get_project(project_id)[0]
        if isinstance(doc_ids, str):
            return indexing_pipeline.delete_document(doc_ids)
        return indexing_pipeline.delete_documents(doc_ids)

    def update(self, doc_id: str, content: str, project_id: str = "default") -> dict:
        """Update a document by deleting and re-indexing it.

        Args:
            doc_id: The external document ID.
            content: New content for the document.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Dictionary with update statistics.
        """
        indexing_pipeline = self._get_project(project_id)[0]
        return indexing_pipeline.update_document(doc_id, content)

    def list_documents(self, project_id: str = "default") -> list[str]:
        """List all registered external document IDs.

        Args:
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            Sorted list of doc_id strings.
        """
        indexing_pipeline = self._get_project(project_id)[0]
        return indexing_pipeline.list_documents()

    def has_document(self, doc_id: str, project_id: str = "default") -> bool:
        """Check if a document ID is registered.

        Args:
            doc_id: The external document ID.
            project_id: Project identifier for data isolation (default: "default").

        Returns:
            True if the doc_id exists, False otherwise.
        """
        indexing_pipeline = self._get_project(project_id)[0]
        return indexing_pipeline.has_document(doc_id)

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
        project_id: str = "default",
        **kwargs,
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
            project_id: Project identifier for data isolation (default: "default").
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
        graph_store = self._get_project(project_id)[2]
        communities = (
            graph_store._communities if hasattr(graph_store, "_communities") else {}
        )

        # Detect communities if not already done
        if not communities:
            graph_store._communities = graph_store.clustering()
            communities = graph_store._communities

        if kind == "graph":
            return {"graph": self.visualizer.visualize_knowledge_graph(
                graph_store=graph_store,
                **kwargs
            )}
        elif kind == "communities":
            return {"communities": self.visualizer.visualize_communities(
                communities=communities,
                graph_store=graph_store,
                **kwargs
            )}
        elif kind == "stats":
            return {"stats": self.visualizer.visualize_entity_stats(
                graph_store=graph_store,
                **kwargs
            )}
        elif kind == "all":
            return self.visualizer.visualize_all(
                graph_store=graph_store,
                communities=communities,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown visualization type: {kind}. "
                "Choose from: 'graph', 'communities', 'stats', 'all'"
            )
