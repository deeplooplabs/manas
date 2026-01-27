# flake8: noqa
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
)

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
    # Pipelines
    "HiRAGIndexingPipeline",
    "HiRAGQueryPipeline",
    # High-level API
    "HiRAG",
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

    def index(
        self,
        documents: list[str] | str,
        incremental: bool = False,
        force_reindex: bool = False,
    ) -> dict:
        """Index documents into the HiRAG system.

        Args:
            documents: Documents to index. Can be a string or list of strings.
            incremental: If True, only index new documents (incremental update).
            force_reindex: If True, reindex all documents (ignores existing).

        Returns:
            Dictionary with indexing statistics.
        """
        if incremental:
            return self.indexing_pipeline.index_incremental(documents, force_reindex=force_reindex)
        return self.indexing_pipeline.index(documents)

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

    @property
    def communities(self) -> dict:
        """Get detected communities."""
        return self.graph_store._communities if hasattr(self.graph_store, "_communities") else {}

    @property
    def reports(self) -> dict:
        """Get community reports."""
        return self.graph_store._reports if hasattr(self.graph_store, "_reports") else {}
