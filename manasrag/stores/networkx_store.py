"""NetworkX-based implementation of GraphDocumentStore.

This module provides an in-memory graph store using NetworkX,
with support for community detection using the Leiden algorithm.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import community as community_louvain

from manasrag.core.community import Community, SingleCommunitySchema
from manasrag.core.graph import Entity, Relation
from manasrag.stores.base import GraphDocumentStore


class NetworkXGraphStore(GraphDocumentStore):
    """NetworkX-based graph store for ManasRAG.

    Uses NetworkX for graph operations and python-louvain for
    community detection (Louvain method).

    Note: The original ManasRAG paper uses Leiden algorithm for better
    quality communities. This implementation uses Louvain (via python-louvain
    package) as it's available without external dependencies. For Leiden
    algorithm, use Neo4jGraphStore with Neo4j GDS library.

    Attributes:
        _graph: Internal NetworkX DiGraph storing entities and relations.
        _communities: Dictionary of detected communities.
    """

    def __init__(
        self,
        namespace: str = "default",
        working_dir: str = "./manas_data",
        global_config: dict | None = None,
    ):
        """Initialize the NetworkX graph store.

        Args:
            namespace: Namespace for this store instance.
            working_dir: Directory for persistent storage.
            global_config: Additional configuration parameters.
        """
        super().__init__(namespace, working_dir, global_config)
        self._graph: nx.DiGraph = nx.DiGraph()
        self._communities: dict[str, Community] = {}
        self._reports: dict[str, str] = {}  # Community reports
        self._cluster_algorithm = self.global_config.get("cluster_algorithm", "leiden")

        # Load existing data if available
        self._load_from_disk()

    # ===== Node Operations =====

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node data by ID."""
        if not self.has_node(node_id):
            return None
        return dict(self._graph.nodes[node_id])

    def upsert_node(self, node_id: str, node_data: dict) -> None:
        """Create or update a node."""
        if self.has_node(node_id):
            # Merge with existing data
            existing = self._graph.nodes[node_id]
            merged_data = dict(existing)

            # Merge source_ids if present
            if "source_id" in node_data and "source_id" in existing:
                existing_sources = set(existing["source_id"].split("|"))
                new_sources = set(node_data["source_id"].split("|"))
                merged_data["source_id"] = "|".join(existing_sources | new_sources)

            # Update other fields
            for key, value in node_data.items():
                if key != "source_id":
                    merged_data[key] = value

            nx.set_node_attributes(self._graph, {node_id: merged_data})
        else:
            self._graph.add_node(node_id, **node_data)

    def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""
        if not self.has_node(node_id):
            return 0
        return self._graph.degree(node_id)

    def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        """Get all edges connected to a node."""
        if not self.has_node(node_id):
            return []
        # Return as (source, target) tuples
        return [(str(e[0]), str(e[1])) for e in self._graph.edges(node_id)]

    # ===== Edge Operations =====

    def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        return self._graph.has_edge(src_id, tgt_id)

    def get_edge(self, src_id: str, tgt_id: str) -> Optional[dict]:
        """Get edge data between two nodes."""
        if not self.has_edge(src_id, tgt_id):
            return None
        return dict(self._graph.edges[src_id, tgt_id])

    def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict) -> None:
        """Create or update an edge between two nodes."""
        # Ensure nodes exist
        if not self.has_node(src_id):
            self.upsert_node(src_id, {"entity_type": "UNKNOWN", "description": "", "source_id": ""})
        if not self.has_node(tgt_id):
            self.upsert_node(tgt_id, {"entity_type": "UNKNOWN", "description": "", "source_id": ""})

        if self.has_edge(src_id, tgt_id):
            # Merge with existing data
            existing = self._graph.edges[src_id, tgt_id]
            merged_data = dict(existing)

            # Accumulate weight
            if "weight" in edge_data:
                merged_data["weight"] = merged_data.get("weight", 0) + edge_data["weight"]

            # Merge source_ids
            if "source_id" in edge_data and "source_id" in existing:
                existing_sources = set(existing["source_id"].split("|"))
                new_sources = set(edge_data["source_id"].split("|"))
                merged_data["source_id"] = "|".join(existing_sources | new_sources)

            # Update other fields
            for key, value in edge_data.items():
                if key not in ("weight", "source_id"):
                    merged_data[key] = value

            nx.set_edge_attributes(self._graph, {(src_id, tgt_id): merged_data})
        else:
            # For undirected graphs in HiRAG, add edges in both directions
            self._graph.add_edge(src_id, tgt_id, **edge_data)

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the degree of an edge."""
        if not self.has_edge(src_id, tgt_id):
            return 0
        # Return sum of node degrees as edge importance
        return self.node_degree(src_id) + self.node_degree(tgt_id)

    # ===== Delete Operations =====

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges from the graph."""
        if self.has_node(node_id):
            self._graph.remove_node(node_id)

    def delete_edge(self, src_id: str, tgt_id: str) -> None:
        """Delete an edge between two nodes."""
        if self.has_edge(src_id, tgt_id):
            self._graph.remove_edge(src_id, tgt_id)

    def get_all_edges(self) -> list[tuple[str, str]]:
        """Get all edges in the graph."""
        return [(str(u), str(v)) for u, v in self._graph.edges()]

    def remove_source_from_node(self, node_id: str, source_id: str) -> bool:
        """Remove a source_id reference from a node.

        Returns True if the node was deleted (no remaining sources).
        """
        if not self.has_node(node_id):
            return False

        node_data = dict(self._graph.nodes[node_id])
        current_sources = set(node_data.get("source_id", "").split("|"))
        current_sources.discard(source_id)
        current_sources.discard("")

        if not current_sources:
            self._graph.remove_node(node_id)
            return True

        node_data["source_id"] = "|".join(current_sources)
        nx.set_node_attributes(self._graph, {node_id: node_data})
        return False

    def remove_source_from_edge(
        self, src_id: str, tgt_id: str, source_id: str
    ) -> bool:
        """Remove a source_id reference from an edge.

        Returns True if the edge was deleted (no remaining sources).
        """
        if not self.has_edge(src_id, tgt_id):
            return False

        edge_data = dict(self._graph.edges[src_id, tgt_id])
        current_sources = set(edge_data.get("source_id", "").split("|"))
        current_sources.discard(source_id)
        current_sources.discard("")

        if not current_sources:
            self._graph.remove_edge(src_id, tgt_id)
            return True

        edge_data["source_id"] = "|".join(current_sources)
        nx.set_edge_attributes(self._graph, {(src_id, tgt_id): edge_data})
        return False

    # ===== Community Operations =====

    def clustering(self, algorithm: str = "louvain") -> dict[str, Community]:
        """Perform community detection on the graph.

        Uses the Louvain method (python-louvain package).
        The 'leiden' parameter is accepted for compatibility but uses Louvain internally.

        Note: For true Leiden algorithm with better community quality,
        use Neo4jGraphStore with Neo4j GDS library.
        """
        if self._graph.number_of_nodes() == 0:
            return {}

        # Convert to undirected for community detection
        undirected = self._graph.to_undirected()

        # Detect communities using Louvain
        partition = community_louvain.best_partition(undirected, random_state=42)

        # Group nodes by community
        community_groups: dict[str, list[str]] = {}
        for node, comm_id in partition.items():
            comm_key = f"community_{comm_id}"
            if comm_key not in community_groups:
                community_groups[comm_key] = []
            community_groups[comm_key].append(node)

        # Create Community objects
        self._communities = {}
        for comm_id, nodes in community_groups.items():
            # Get edges within this community
            subgraph = undirected.subgraph(nodes)
            edges = [(str(u), str(v)) for u, v in subgraph.edges()]

            # Get all source chunks
            chunk_ids = set()
            for node in nodes:
                node_data = self.get_node(node)
                if node_data and "source_id" in node_data:
                    chunk_ids.update(node_data["source_id"].split("|"))

            community = Community(
                level=0,
                title=f"Community {comm_id}",
                nodes=nodes,
                edges=edges,
                chunk_ids=list(chunk_ids),
                occurrence=float(len(nodes)),
                community_id=comm_id,
            )
            self._communities[comm_id] = community

        return self._communities

    def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Get the community structure schema."""
        return {
            comm_id: comm.to_schema()
            for comm_id, comm in self._communities.items()
        }

    def get_communities(self) -> dict:
        """Get all detected communities.

        Returns:
            Dictionary mapping community IDs to Community objects.
        """
        return self._communities

    def get_reports(self) -> dict:
        """Get all community reports.

        Returns:
            Dictionary mapping community IDs to report strings.
        """
        return self._reports

    # ===== Path Operations =====

    def shortest_path(self, src: str, tgt: str) -> list[str]:
        """Find the shortest path between two nodes."""
        if not self.has_node(src) or not self.has_node(tgt):
            return []

        try:
            # Use undirected graph for path finding
            undirected = self._graph.to_undirected()
            path = nx.shortest_path(undirected, source=src, target=tgt)
            return [str(node) for node in path]
        except nx.NetworkXNoPath:
            return []

    def subgraph_edges(self, nodes: list[str]) -> list:
        """Get all edges in the subgraph induced by the given nodes."""
        if not nodes:
            return []

        # Filter to existing nodes
        existing_nodes = [n for n in nodes if self.has_node(n)]
        if not existing_nodes:
            return []

        undirected = self._graph.to_undirected()
        subgraph = undirected.subgraph(existing_nodes)

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append({
                "src": str(u),
                "tgt": str(v),
                "weight": data.get("weight", 1.0),
                "description": data.get("description", ""),
            })

        return edges

    # ===== Persistence =====

    def _load_from_disk(self) -> None:
        """Load graph data from disk if available."""
        graph_path = Path(self.working_dir) / f"{self.namespace}_graph.json"
        comm_path = Path(self.working_dir) / f"{self.namespace}_communities.json"
        reports_path = Path(self.working_dir) / f"{self.namespace}_reports.json"

        if graph_path.exists():
            try:
                with open(graph_path, "r") as f:
                    data = json.load(f)
                    self._graph = nx.node_link_graph(data)
            except Exception:
                self._graph = nx.DiGraph()

        if comm_path.exists():
            try:
                with open(comm_path, "r") as f:
                    data = json.load(f)
                    self._communities = {
                        k: Community(**v) for k, v in data.items()
                    }
            except Exception:
                self._communities = {}

        # Load reports
        if reports_path.exists():
            try:
                with open(reports_path, "r") as f:
                    self._reports = json.load(f)
            except Exception:
                self._reports = {}

    def save_to_disk(self) -> None:
        """Save graph data to disk."""
        os.makedirs(self.working_dir, exist_ok=True)

        graph_path = Path(self.working_dir) / f"{self.namespace}_graph.json"
        comm_path = Path(self.working_dir) / f"{self.namespace}_communities.json"
        reports_path = Path(self.working_dir) / f"{self.namespace}_reports.json"

        # Save graph
        graph_data = nx.node_link_data(self._graph)
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        # Save communities
        comm_data = {
            k: {
                "level": v.level,
                "title": v.title,
                "nodes": v.nodes,
                "edges": v.edges,
                "chunk_ids": v.chunk_ids,
                "occurrence": v.occurrence,
                "sub_communities": v.sub_communities,
                "report_string": v.report_string,
                "report_json": v.report_json,
                "community_id": v.community_id,
            }
            for k, v in self._communities.items()
        }
        with open(comm_path, "w") as f:
            json.dump(comm_data, f, indent=2)

        # Save reports
        with open(reports_path, "w") as f:
            json.dump(self._reports, f, indent=2)

    def index_done_callback(self) -> None:
        """Save data after indexing."""
        self.save_to_disk()
