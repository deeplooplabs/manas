"""Community detection component for HiRAG.

This component performs clustering on the knowledge graph to identify
communities of related entities, forming a hierarchical structure.
"""

from typing import Any

from haystack import component

from hirag_haystack.core.community import Community, SingleCommunitySchema
from hirag_haystack.stores.base import GraphDocumentStore


@component
class CommunityDetector:
    """Detect communities in the knowledge graph.

    Uses the graph store's clustering implementation (typically
    Leiden/Louvain algorithm) to identify communities of related entities.
    """

    def __init__(
        self,
        algorithm: str = "leiden",
        max_cluster_size: int = 10,
        seed: int = 0xDEADBEEF,
    ):
        """Initialize the CommunityDetector.

        Args:
            algorithm: Clustering algorithm to use ("leiden" or "louvain").
            max_cluster_size: Target maximum size for clusters.
            seed: Random seed for reproducibility.
        """
        self.algorithm = algorithm
        self.max_cluster_size = max_cluster_size
        self.seed = seed

    @component.output_types(communities=dict)
    def run(
        self,
        graph_store: GraphDocumentStore,
    ) -> dict:
        """Perform community detection on the graph.

        Args:
            graph_store: The graph store containing entities and relations.

        Returns:
            Dictionary with:
                - communities: Dict mapping community IDs to Community objects
        """
        if graph_store is None:
            return {"communities": {}}

        # Use the graph store's clustering implementation
        communities = graph_store.clustering(algorithm=self.algorithm)

        return {"communities": communities}

    def _build_hierarchical_communities(
        self,
        flat_communities: dict[str, Community],
    ) -> dict[str, Community]:
        """Build a hierarchical community structure.

        Groups smaller communities into larger ones to create
        a multi-level hierarchy.

        Args:
            flat_communities: Flat mapping of communities from clustering.

        Returns:
            Hierarchical community structure with levels.
        """
        # For now, return flat communities at level 0
        # TODO: Implement hierarchical clustering
        return flat_communities


@component
class CommunityAssigner:
    """Assign entities to their detected communities.

    This component updates entity records with their community memberships
    after community detection has been performed.
    """

    @component.output_types(entities=list)
    def run(
        self,
        entities: list,
        communities: dict[str, Community],
    ) -> dict:
        """Assign communities to entities.

        Args:
            entities: List of Entity objects.
            communities: Dict of detected communities.

        Returns:
            Dictionary with updated entities including cluster assignments.
        """
        # Build mapping from entity name to communities
        entity_to_communities = {}

        for comm_id, community in communities.items():
            for entity_name in community.nodes:
                if entity_name not in entity_to_communities:
                    entity_to_communities[entity_name] = []
                entity_to_communities[entity_name].append({
                    "level": community.level,
                    "cluster": comm_id,
                })

        # Update entities with cluster information
        for entity in entities:
            if entity.entity_name in entity_to_communities:
                entity.clusters = entity_to_communities[entity.entity_name]

        return {"entities": entities}
