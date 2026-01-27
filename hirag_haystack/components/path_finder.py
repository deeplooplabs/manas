"""Path finder component for HiRAG.

This component provides advanced path finding capabilities for
cross-community reasoning in the knowledge graph.
"""

from typing import Any, List, Optional

from haystack import component

from hirag_haystack.stores.base import GraphDocumentStore


@component
class PathFinder:
    """Find paths through the knowledge graph.

    Provides methods for finding:
    - Shortest paths between entities
    - Paths that pass through required intermediate nodes
    - Cross-community paths that bridge different knowledge areas
    """

    def __init__(
        self,
        graph_store: GraphDocumentStore | None = None,
    ):
        """Initialize the path finder.

        Args:
            graph_store: The graph store to query.
        """
        self.graph_store = graph_store

    @component.output_types(paths=list)
    def run(
        self,
        source: str,
        target: str,
        required_nodes: List[str] | None = None,
        max_path_length: int = 10,
    ) -> dict:
        """Find a path through the graph.

        Args:
            source: Starting entity name.
            target: Target entity name.
            required_nodes: Optional list of intermediate nodes that must be visited.
            max_path_length: Maximum allowed path length.

        Returns:
            Dictionary with list of paths found.
        """
        if self.graph_store is None:
            return {"paths": []}

        if required_nodes:
            path = self._find_path_with_required_nodes(
                source, target, required_nodes, max_path_length
            )
            return {"paths": [path] if path else []}

        # Simple shortest path
        path = self.graph_store.shortest_path(source, target)
        return {"paths": [path] if path else []}

    def _find_path_with_required_nodes(
        self,
        source: str,
        target: str,
        required_nodes: List[str],
        max_length: int = 10,
    ) -> List[str]:
        """Find a path that visits all required nodes.

        This method finds a path from source to target that passes through
        all required intermediate nodes. It uses a greedy approach,
        finding the shortest path segment between consecutive required nodes.

        Args:
            source: Starting node.
            target: Target node.
            required_nodes: Nodes that must be visited in order.
            max_length: Maximum total path length.

        Returns:
            List of node names forming the path.
        """
        if not required_nodes:
            return self.graph_store.shortest_path(source, target)

        # Build path by connecting segments
        full_path = []
        current = source
        remaining_nodes = list(required_nodes)

        for next_node in remaining_nodes + [target]:
            # Find shortest path from current to next
            segment = self.graph_store.shortest_path(current, next_node)

            if not segment:
                # No path found, try direct connection
                segment = [current, next_node]

            # Merge segment (avoid duplicate current node)
            if full_path:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)

            # Check length constraint
            if len(full_path) > max_length:
                # Truncate and return partial path
                return full_path[:max_length]

            current = next_node

        return full_path

    @component.output_types(paths=list, edges=list)
    def find_cross_community_paths(
        self,
        entities: List[str],
        communities: dict,
        top_k: int = 3,
    ) -> dict:
        """Find paths that bridge different communities.

        This method identifies key entities from different communities
        and finds paths that connect them, enabling cross-community reasoning.

        Args:
            entities: List of entity names to consider.
            communities: Dictionary of community objects.
            top_k: Number of top paths to return.

        Returns:
            Dictionary with paths and connecting edges.
        """
        if not entities or not communities or not self.graph_store:
            return {"paths": [], "edges": []}

        # Group entities by community
        entity_communities = {}
        for entity in entities:
            if self.graph_store.has_node(entity):
                node_data = self.graph_store.get_node(entity)
                if node_data:
                    # Get cluster info
                    import json
                    clusters_str = node_data.get("clusters", "[]")
                    try:
                        clusters = json.loads(clusters_str) if isinstance(clusters_str, str) else clusters_str
                        for cluster in clusters:
                            comm_id = cluster.get("cluster", "")
                            if comm_id not in entity_communities:
                                entity_communities[comm_id] = []
                            entity_communities[comm_id].append(entity)
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Select top entities from each community
        key_entities = []
        for comm_id, comm_entities in list(entity_communities.items())[:top_k]:
            # Sort by degree and take top 2
            comm_entities_sorted = sorted(
                comm_entities,
                key=lambda e: self.graph_store.node_degree(e) if self.graph_store else 0,
                reverse=True,
            )[:2]
            key_entities.extend(comm_entities_sorted)

        # Find paths connecting key entities
        paths = []
        all_edges = []

        if len(key_entities) >= 2:
            path = self._find_path_with_required_nodes(
                key_entities[0],
                key_entities[-1],
                key_entities[1:-1] if len(key_entities) > 2 else [],
            )
            paths.append(path)

            # Get edges along the path
            edges = self._get_edges_along_path(path)
            all_edges.extend(edges)

        return {
            "paths": paths,
            "edges": all_edges,
        }

    def _get_edges_along_path(self, path: List[str]) -> List[dict]:
        """Get all edges along a path.

        Args:
            path: List of node names.

        Returns:
            List of edge data dictionaries.
        """
        if not path or len(path) < 2 or not self.graph_store:
            return []

        edges = []

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]

            edge_data = self.graph_store.get_edge(src, tgt)
            if edge_data:
                edges.append({
                    "source": src,
                    "target": tgt,
                    **edge_data,
                    "rank": self.graph_store.edge_degree(src, tgt),
                })

        return edges

    @component.output_types(communities=list)
    def find_bridging_communities(
        self,
        source_entity: str,
        target_entity: str,
        communities: dict,
        max_hops: int = 3,
    ) -> dict:
        """Find communities that serve as bridges between entities.

        Identifies communities that lie on paths connecting two entities,
        which is useful for understanding how different knowledge areas relate.

        Args:
            source_entity: Starting entity.
            target_entity: Target entity.
            communities: Dictionary of all communities.
            max_hops: Maximum path length to consider.

        Returns:
            Dictionary with list of bridging communities.
        """
        if not self.graph_store:
            return {"communities": []}

        path = self.graph_store.shortest_path(source_entity, target_entity)

        if not path or len(path) > max_hops:
            return {"communities": []}

        # Find which communities the path passes through
        bridging_communities = []

        for node in path:
            node_data = self.graph_store.get_node(node)
            if node_data:
                import json
                clusters_str = node_data.get("clusters", "[]")
                try:
                    clusters = json.loads(clusters_str) if isinstance(clusters_str, str) else clusters_str
                    for cluster in clusters:
                        comm_id = cluster.get("cluster", "")
                        if comm_id in communities and comm_id not in bridging_communities:
                            bridging_communities.append(comm_id)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Get community details
        community_details = []
        for comm_id in bridging_communities:
            if comm_id in communities:
                comm = communities[comm_id]
                community_details.append({
                    "community_id": comm_id,
                    "title": getattr(comm, "title", comm_id),
                    "level": getattr(comm, "level", 0),
                    "node_count": len(getattr(comm, "nodes", [])),
                })

        return {"communities": community_details}


@component
class PathScorer:
    """Score and rank paths based on relevance metrics.

    Scoring considers:
    - Path length (shorter is generally better)
    - Edge weights (higher weight indicates stronger connection)
    - Node degrees (connection importance)
    """

    def __init__(
        self,
        graph_store: GraphDocumentStore | None = None,
        length_weight: float = 0.3,
        weight_weight: float = 0.5,
        degree_weight: float = 0.2,
    ):
        """Initialize the path scorer.

        Args:
            graph_store: The graph store.
            length_weight: Weight for path length in scoring.
            weight_weight: Weight for edge weights in scoring.
            degree_weight: Weight for node degrees in scoring.
        """
        self.graph_store = graph_store
        self.length_weight = length_weight
        self.weight_weight = weight_weight
        self.degree_weight = degree_weight

    @component.output_types(scores=list)
    def run(
        self,
        paths: List[List[str]],
    ) -> dict:
        """Score multiple paths and return rankings.

        Args:
            paths: List of paths (each path is a list of node names).

        Returns:
            Dictionary with list of (path_index, score) tuples.
        """
        scores = []

        for i, path in enumerate(paths):
            score = self._score_path(path)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return {"scores": scores}

    def _score_path(self, path: List[str]) -> float:
        """Calculate a composite score for a path.

        Args:
            path: List of node names.

        Returns:
            Composite score (higher is better).
        """
        if not path or not self.graph_store:
            return 0.0

        # Length component (inverse - shorter paths are better)
        length_score = 1.0 / max(1, len(path))

        # Weight component (sum of edge weights)
        weight_sum = 0.0
        weight_count = 0

        # Degree component (sum of node degrees)
        degree_sum = 0

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            edge_data = self.graph_store.get_edge(src, tgt)
            if edge_data:
                weight_sum += edge_data.get("weight", 1.0)
                weight_count += 1

            degree_sum += self.graph_store.node_degree(path[i])

        degree_sum += self.graph_store.node_degree(path[-1])

        # Average components
        avg_weight = weight_sum / max(1, weight_count)
        avg_degree = degree_sum / max(1, len(path))

        # Normalize scores roughly to 0-1 range
        weight_score = min(1.0, avg_weight / 5.0)
        degree_score = min(1.0, avg_degree / 10.0)

        # Composite score
        composite = (
            self.length_weight * length_score +
            self.weight_weight * weight_score +
            self.degree_weight * degree_score
        )

        return composite
