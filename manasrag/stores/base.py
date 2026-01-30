"""Base class for graph document storage in ManasRAG.

This module defines the abstract interface for graph storage backends
that support hierarchical knowledge operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from manasrag.core.community import Community, CommunitySchema, SingleCommunitySchema
from manasrag.core.graph import Entity, Relation


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    namespace: str = "default"
    working_dir: str = "./manas_data"
    global_config: dict = field(default_factory=dict)


class GraphDocumentStore(ABC):
    """Abstract base class for graph storage backends.

    A GraphDocumentStore stores entities as nodes and their relationships
    as edges, with support for community detection and hierarchical organization.
    """

    def __init__(
        self,
        namespace: str = "default",
        working_dir: str = "./hirag_cache",
        global_config: dict | None = None,
    ):
        """Initialize the graph store.

        Args:
            namespace: Namespace for this store instance.
            working_dir: Directory for persistent storage.
            global_config: Additional configuration parameters.
        """
        self.namespace = namespace
        self.working_dir = working_dir
        self.global_config = global_config or {}

    # ===== Node Operations =====

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The entity name to check.

        Returns:
            True if the node exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node data by ID.

        Args:
            node_id: The entity name to retrieve.

        Returns:
            Dictionary with node data (entity_type, description, source_id, etc.)
            or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_node(self, node_id: str, node_data: dict) -> None:
        """Create or update a node.

        Args:
            node_id: The entity name.
            node_data: Dictionary with entity_type, description, source_id, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node.

        Args:
            node_id: The entity name.

        Returns:
            The degree of the node.
        """
        raise NotImplementedError

    @abstractmethod
    def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        """Get all edges connected to a node.

        Args:
            node_id: The entity name.

        Returns:
            List of (source, target) tuples for edges connected to this node.
        """
        raise NotImplementedError

    # ===== Edge Operations =====

    @abstractmethod
    def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.

        Returns:
            True if the edge exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, src_id: str, tgt_id: str) -> Optional[dict]:
        """Get edge data between two nodes.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.

        Returns:
            Dictionary with edge data (weight, description, source_id, etc.)
            or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict) -> None:
        """Create or update an edge between two nodes.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.
            edge_data: Dictionary with weight, description, source_id, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the degree of an edge.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.

        Returns:
            The edge degree.
        """
        raise NotImplementedError

    # ===== Community Operations =====

    @abstractmethod
    def clustering(self, algorithm: str = "leiden") -> dict[str, Community]:
        """Perform community detection on the graph.

        Args:
            algorithm: The clustering algorithm to use (e.g., "leiden").

        Returns:
            Dictionary mapping community IDs to Community objects.
        """
        raise NotImplementedError

    @abstractmethod
    def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Get the community structure schema.

        Returns:
            Dictionary mapping community IDs to CommunitySchema objects.
        """
        raise NotImplementedError

    # ===== Path Operations =====

    @abstractmethod
    def shortest_path(self, src: str, tgt: str) -> list[str]:
        """Find the shortest path between two nodes.

        Args:
            src: Source entity name.
            tgt: Target entity name.

        Returns:
            List of entity names forming the path from src to tgt.
        """
        raise NotImplementedError

    @abstractmethod
    def subgraph_edges(self, nodes: list[str]) -> list:
        """Get all edges in the subgraph induced by the given nodes.

        Args:
            nodes: List of entity names.

        Returns:
            List of edge data for edges connecting the given nodes.
        """
        raise NotImplementedError

    # ===== Delete Operations =====

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges from the graph.

        Args:
            node_id: The entity name to delete.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_edge(self, src_id: str, tgt_id: str) -> None:
        """Delete an edge between two nodes.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_edges(self) -> list[tuple[str, str]]:
        """Get all edges in the graph.

        Returns:
            List of (source, target) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_source_from_node(self, node_id: str, source_id: str) -> bool:
        """Remove a source_id reference from a node.

        If the node has no remaining sources, it should be deleted.

        Args:
            node_id: The entity name.
            source_id: The chunk source_id to remove.

        Returns:
            True if the node was deleted (no remaining sources), False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_source_from_edge(
        self, src_id: str, tgt_id: str, source_id: str
    ) -> bool:
        """Remove a source_id reference from an edge.

        If the edge has no remaining sources, it should be deleted.

        Args:
            src_id: Source entity name.
            tgt_id: Target entity name.
            source_id: The chunk source_id to remove.

        Returns:
            True if the edge was deleted (no remaining sources), False otherwise.
        """
        raise NotImplementedError

    # ===== Utility Methods =====

    def index_start_callback(self) -> None:
        """Callback called before indexing starts."""
        pass

    def index_done_callback(self) -> None:
        """Callback called after indexing completes."""
        pass

    def query_done_callback(self) -> None:
        """Callback called after query completes."""
        pass
