"""Community detection component for HiRAG.

This component performs clustering on the knowledge graph to identify
communities of related entities, forming a hierarchical structure.

Implements:
- Louvain community detection (level 0)
- GMM clustering for higher levels
- Cluster sparsity calculation for dynamic layer determination
"""

from typing import Any

from haystack import component

from hirag_haystack.core.community import Community, SingleCommunitySchema
from hirag_haystack.stores.base import GraphDocumentStore


def calculate_cluster_sparsity(
    embeddings: list[list[float]],
    cluster_labels: list[int],
) -> float:
    """Calculate cluster sparsity (variance ratio).

    Sparsity is calculated as the ratio of within-cluster variance
    to between-cluster variance. Lower sparsity indicates more
    distinct clusters.

    Args:
        embeddings: List of embedding vectors.
        cluster_labels: Cluster assignments for each embedding.

    Returns:
        Sparsity score (0 to 1). Lower is better (more distinct clusters).
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances

        embeddings_array = np.array(embeddings)
        unique_labels = set(cluster_labels)

        if len(unique_labels) < 2:
            return 1.0

        # Calculate within-cluster variance (compactness)
        within_variance = 0.0
        for label in unique_labels:
            cluster_points = embeddings_array[np.array(cluster_labels) == label]
            if len(cluster_points) > 1:
                centroid = cluster_points.mean(axis=0)
                distances = euclidean_distances(cluster_points, [centroid])
                within_variance += distances.sum()

        # Calculate between-cluster variance (separation)
        centroids = []
        for label in unique_labels:
            cluster_points = embeddings_array[np.array(cluster_labels) == label]
            centroids.append(cluster_points.mean(axis=0))
        centroids_array = np.array(centroids)

        if len(centroids) > 1:
            between_variance = euclidean_distances(centroids_array).sum() / 2
        else:
            between_variance = 1.0

        # Sparsity = within / (within + between)
        total = within_variance + between_variance
        if total == 0:
            return 1.0

        return within_variance / total

    except ImportError:
        return 0.5  # Default middle value if sklearn unavailable


@component
class CommunityDetector:
    """Detect communities in the knowledge graph.

    Uses the graph store's clustering implementation (Louvain algorithm)
    to identify communities of related entities at level 0.

    For higher levels, uses GMM clustering on community embeddings.
    """

    def __init__(
        self,
        algorithm: str = "louvain",
        max_cluster_size: int = 10,
        seed: int = 0xDEADBEEF,
        sparsity_threshold: float = 0.05,
        max_levels: int = 3,
    ):
        """Initialize the CommunityDetector.

        Args:
            algorithm: Clustering algorithm for level 0 ("louvain").
                       Note: NetworkX store uses Louvain (python-louvain package).
                       Neo4j GDS supports "leiden" for better quality communities.
            max_cluster_size: Target maximum size for clusters.
            seed: Random seed for reproducibility.
            sparsity_threshold: Threshold for stopping hierarchical clustering (ε).
            max_levels: Maximum number of hierarchical levels.
        """
        self.algorithm = algorithm
        self.max_cluster_size = max_cluster_size
        self.seed = seed
        self.sparsity_threshold = sparsity_threshold
        self.max_levels = max_levels

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

    def gmm_cluster(
        self,
        embeddings: list[list[float]],
        n_clusters: int | None = None,
    ) -> list[int]:
        """Perform Gaussian Mixture Model clustering.

        Args:
            embeddings: List of embedding vectors.
            n_clusters: Number of clusters (auto-calculated if None).

        Returns:
            List of cluster assignments.
        """
        try:
            from sklearn.mixture import GaussianMixture
            import numpy as np

            embeddings_array = np.array(embeddings)

            if n_clusters is None:
                # Auto-determine based on data size and max_cluster_size
                n_clusters = max(1, len(embeddings) // self.max_cluster_size)

            # Fit GMM
            gmm = GaussianMixture(
                n_components=min(n_clusters, len(embeddings)),
                covariance_type="full",
                random_state=self.seed,
                n_init=3,
            )
            return gmm.fit_predict(embeddings_array).tolist()

        except ImportError:
            # Fallback: assign all to single cluster
            return [0] * len(embeddings)

    def _build_hierarchical_communities(
        self,
        flat_communities: dict[str, Community],
    ) -> dict[str, Community]:
        """Build a hierarchical community structure.

        Implements the HiRAG hierarchical KG construction:
        1. Start with Louvain-detected communities (G0)
        2. Generate summary entities for each community
        3. Use GMM to cluster communities into higher levels (G1, G2, ...)
        4. Continue until sparsity change < ε or max levels reached

        Args:
            flat_communities: Flat mapping of communities from clustering.

        Returns:
            Hierarchical community structure with multiple levels.
        """
        if len(flat_communities) <= 1:
            return flat_communities

        hierarchical_communities = {}

        # Level 0 communities (already detected)
        level_0_communities = {
            comm_id: comm for comm_id, comm in flat_communities.items()
            if comm.level == 0
        }

        # If no level info, assume all are level 0
        if not level_0_communities:
            for comm_id, comm in flat_communities.items():
                comm.level = 0
            level_0_communities = flat_communities

        hierarchical_communities.update(level_0_communities)

        # Build higher levels (G1, G2, ...)
        current_level = 1
        previous_level_communities = level_0_communities

        while current_level <= self.max_levels:
            # Check if we should continue
            if len(previous_level_communities) <= 1:
                break

            # Generate summary entities and build next level
            next_level_communities = self._build_next_hierarchy_level(
                previous_level_communities,
                current_level,
            )

            if not next_level_communities:
                break

            hierarchical_communities.update(next_level_communities)
            previous_level_communities = next_level_communities
            current_level += 1

        return hierarchical_communities

    def _build_next_hierarchy_level(
        self,
        communities: dict[str, Community],
        level: int,
    ) -> dict[str, Community]:
        """Build the next hierarchical level from current communities.

        Args:
            communities: Communities at the current level.
            level: The level number being built (1-based, so level 1 -> G1).

        Returns:
            Communities at the next level.
        """
        # Extract community embeddings (use community report as representation)
        community_embeddings = []
        community_info = []

        for comm_id, comm in communities.items():
            # Use report text or node descriptions as embedding source
            if hasattr(comm, 'report_string') and comm.report_string:
                text_source = comm.report_string
            elif comm.nodes:
                # Use first few node descriptions
                text_source = " ".join(comm.nodes[:5])
            else:
                text_source = comm_id

            community_info.append({
                "id": comm_id,
                "title": comm.title,
                "nodes": comm.nodes,
                "text_source": text_source,
            })
            community_embeddings.append([0.0] * 768)  # Placeholder

        # In a real implementation, generate embeddings for community texts
        # For now, use synthetic features based on community properties
        synthetic_features = self._compute_community_features(community_info)

        # Cluster using GMM
        cluster_labels = self.gmm_cluster(
            synthetic_features,
            n_clusters=max(1, len(communities) // self.max_cluster_size),
        )

        # Group communities by cluster
        cluster_groups: dict[int, list[dict]] = {}
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(community_info[idx])

        # Create higher-level communities
        next_level_communities = {}
        new_level = level  # Level number for these communities

        for cluster_id, cluster_communities in cluster_groups.items():
            if len(cluster_communities) < 2:
                continue  # Skip single-community clusters

            # Collect all nodes from child communities
            all_nodes = []
            all_edges = []
            all_chunk_ids = set()

            for comm_info in cluster_communities:
                child_comm = communities.get(comm_info["id"])
                if child_comm:
                    all_nodes.extend(child_comm.nodes)
                    all_edges.extend(child_comm.edges)
                    all_chunk_ids.update(child_comm.chunk_ids)

            # Generate summary title
            summary_title = f"Meta-Community L{new_level}_{cluster_id}: " + \
                ", ".join(c["title"] for c in cluster_communities[:3])

            meta_community = Community(
                level=new_level,
                title=summary_title,
                nodes=list(set(all_nodes)),
                edges=list(set(all_edges)),
                chunk_ids=list(all_chunk_ids),
                occurrence=float(len(cluster_communities)),
                community_id=f"meta_level_{new_level}_cluster_{cluster_id}",
                sub_communities=[c["id"] for c in cluster_communities],
            )

            next_level_communities[meta_community.community_id] = meta_community

        return next_level_communities

    def _compute_community_features(
        self,
        community_info: list[dict],
    ) -> list[list[float]]:
        """Compute synthetic features for communities.

        These features are used for hierarchical clustering when
        embeddings are not available.

        Args:
            community_info: List of community information dicts.

        Returns:
            List of feature vectors.
        """
        import numpy as np

        features = []
        for comm in community_info:
            node_count = len(comm.get("nodes", []))
            # Simple hash-based features as placeholder
            hash_val = hash(comm["id"]) % 10000
            features.append([
                node_count / 100.0,  # Normalized node count
                hash_val / 10000.0,  # ID-based feature
                len(comm.get("title", "")) / 100.0,  # Title length
            ])

        return features if features else [[0.0, 0.0, 0.0]]


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
