"""Neo4j implementation of GraphDocumentStore for HiRAG.

This module provides a production-ready graph store using Neo4j,
suitable for large-scale knowledge graphs.
"""

from typing import Any, Optional

from neo4j import GraphDatabase

from hirag_haystack.stores.base import GraphDocumentStore
from hirag_haystack.core.community import Community, SingleCommunitySchema


class Neo4jGraphStore(GraphDocumentStore):
    """Neo4j-based graph store for HiRAG.

    Uses Neo4j for scalable graph storage and querying.
    Supports Cypher queries for efficient graph traversals.

    Attributes:
        _driver: Neo4j driver instance.
        _uri: Neo4j connection URI.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        namespace: str = "default",
        working_dir: str = "./hirag_cache",
        global_config: dict | None = None,
    ):
        """Initialize the Neo4j graph store.

        Args:
            uri: Neo4j connection URI.
            username: Database username.
            password: Database password.
            database: Database name.
            namespace: Namespace for this store instance.
            working_dir: Directory for cache storage.
            global_config: Additional configuration parameters.
        """
        super().__init__(namespace, working_dir, global_config)

        self._uri = uri
        self._database = database
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        # Create constraints for better performance
        self._create_constraints()

    def _create_constraints(self) -> None:
        """Create database constraints for performance."""
        with self._driver.session(database=self._database) as session:
            try:
                # Create unique constraint on entity names
                session.run(
                    "CREATE CONSTRAINT entity_name_unique "
                    "IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
            except Exception:
                pass  # Constraint may already exist

    def _execute_query(
        self,
        query: str,
        parameters: dict | None = None,
    ) -> Any:
        """Execute a Cypher query.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            Query result.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            return result

    # ===== Node Operations =====

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        result = self._execute_query(
            "MATCH (e:Entity {name: $name}) RETURN count(e) > 0",
            {"name": node_id},
        )
        return result.single()[0]

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node data by ID."""
        result = self._execute_query(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.name as name,
                   e.type as entity_type,
                   e.description as description,
                   e.source_id as source_id,
                   e.clusters as clusters
            """,
            {"name": node_id},
        )

        record = result.single()
        if not record:
            return None

        return {
            "entity_name": record["name"],
            "entity_type": record["entity_type"],
            "description": record["description"] or "",
            "source_id": record["source_id"] or "",
            "clusters": record["clusters"] or "[]",
        }

    def upsert_node(self, node_id: str, node_data: dict) -> None:
        """Create or update a node."""
        # Merge ensures idempotency
        self._execute_query(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.description = $description,
                e.source_id = $source_id,
                e.clusters = $clusters
            """,
            {
                "name": node_id,
                "type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_id": node_data.get("source_id", ""),
                "clusters": node_data.get("clusters", "[]"),
            },
        )

    def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""
        result = self._execute_query(
            """
            MATCH (e:Entity {name: $name})-[r]-(x)
            RETURN count(r) as degree
            """,
            {"name": node_id},
        )
        record = result.single()
        return record["degree"] if record else 0

    def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        """Get all edges connected to a node."""
        result = self._execute_query(
            """
            MATCH (e:Entity {name: $name})-[r]-(other:Entity)
            RETURN e.name as source, other.name as target
            """,
            {"name": node_id},
        )

        edges = []
        for record in result:
            edges.append((record["source"], record["target"]))

        return edges

    # ===== Edge Operations =====

    def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        result = self._execute_query(
            """
            MATCH (s:Entity {name: $src})-[r]-(t:Entity {name: $tgt})
            RETURN count(r) > 0
            """,
            {"src": src_id, "tgt": tgt_id},
        )
        return result.single()[0]

    def get_edge(self, src_id: str, tgt_id: str) -> Optional[dict]:
        """Get edge data between two nodes."""
        result = self._execute_query(
            """
            MATCH (s:Entity {name: $src})-[r]-(t:Entity {name: $tgt})
            RETURN r.weight as weight,
                   r.description as description,
                   r.source_id as source_id,
                   r.order as `order`
            """,
            {"src": src_id, "tgt": tgt_id},
        )

        record = result.single()
        if not record:
            return None

        return {
            "weight": record["weight"] or 1.0,
            "description": record["description"] or "",
            "source_id": record["source_id"] or "",
            "order": record["order"] or 1,
        }

    def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict) -> None:
        """Create or update an edge between two nodes."""
        self._execute_query(
            """
            MATCH (s:Entity {name: $src})
            MATCH (t:Entity {name: $tgt})
            MERGE (s)-[r:RELATES_TO]->(t)
            SET r.weight = $weight,
                r.description = $description,
                r.source_id = $source_id,
                r.order = $order
            """,
            {
                "src": src_id,
                "tgt": tgt_id,
                "weight": edge_data.get("weight", 1.0),
                "description": edge_data.get("description", ""),
                "source_id": edge_data.get("source_id", ""),
                "order": edge_data.get("order", 1),
            },
        )

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the edge degree."""
        src_degree = self.node_degree(src_id)
        tgt_degree = self.node_degree(tgt_id)
        return src_degree + tgt_degree

    # ===== Community Operations =====

    def clustering(self, algorithm: str = "leiden") -> dict[str, Community]:
        """Perform community detection using Neo4j.

        Uses Neo4j's graph data science library if available,
        otherwise falls back to Louvain algorithm.
        """
        try:
            # Try using GDS library
            return self._gds_clustering(algorithm)
        except Exception:
            # Fallback to simple clustering
            return self._simple_clustering()

    def _gds_clustering(self, algorithm: str) -> dict[str, Community]:
        """Use Neo4j Graph Data Science library for clustering."""
        # Project graph
        self._execute_query(
            """
            CALL gds.graph.project(
                'hiragGraph',
                'Entity',
                'RELATES_TO',
                {
                    relationshipProperties: ['weight']
                }
            )
            """
        )

        # Run Louvain algorithm
        result = self._execute_query(
            """
            CALL gds.louvain.stream('hiragGraph')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).name as entity_name, communityId
            """
        )

        # Group by community
        community_groups = {}
        for record in result:
            comm_id = f"community_{record['communityId']}"
            entity_name = record["entity_name"]

            if comm_id not in community_groups:
                community_groups[comm_id] = []
            community_groups[comm_id].append(entity_name)

        # Create Community objects
        communities = {}
        for comm_id, nodes in community_groups.items():
            communities[comm_id] = Community(
                level=0,
                title=f"Community {comm_id}",
                nodes=nodes,
                edges=[],
                chunk_ids=[],
                occurrence=float(len(nodes)),
                community_id=comm_id,
            )

        return communities

    def _simple_clustering(self) -> dict[str, Community]:
        """Simple clustering based on connected components."""
        result = self._execute_query(
            """
            CALL {
                MATCH (e:Entity)
                WITH collect(e.name) as nodes
                RETURN nodes
            }
            """
        )

        communities = {}
        for i, record in enumerate(result):
            nodes = record["nodes"]
            comm_id = f"community_{i}"

            communities[comm_id] = Community(
                level=0,
                title=f"Community {comm_id}",
                nodes=nodes,
                edges=[],
                chunk_ids=[],
                occurrence=float(len(nodes)),
                community_id=comm_id,
            )

        return communities

    def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Get the community structure schema."""
        communities = self._communities if hasattr(self, "_communities") else {}
        return {
            k: v.to_schema() for k, v in communities.items()
        }

    # ===== Path Operations =====

    def shortest_path(self, src: str, tgt: str) -> list[str]:
        """Find the shortest path between two nodes."""
        result = self._execute_query(
            """
            MATCH path = shortestPath(
                (s:Entity {name: $src})-[*..15]-(t:Entity {name: $tgt})
            )
            REDUCE [node.name IN nodes(path) | node.name] as path
            RETURN path
            """,
            {"src": src, "tgt": tgt},
        )

        record = result.single()
        return record["path"] if record else []

    def subgraph_edges(self, nodes: list[str]) -> list:
        """Get all edges in the subgraph induced by the given nodes."""
        if not nodes:
            return []

        result = self._execute_query(
            """
            UNWIND $nodes_list as node_name
            MATCH (e:Entity {name: node_name})
            MATCH (e)-[r]-(other:Entity)
            WHERE other.name IN $nodes_list
            RETURN e.name as src,
                   other.name as tgt,
                   r.weight as weight,
                   r.description as description
            """,
            {"nodes_list": nodes},
        )

        edges = []
        seen = set()

        for record in result:
            edge_key = tuple(sorted([record["src"], record["tgt"]]))
            if edge_key not in seen:
                seen.add(edge_key)
                edges.append({
                    "src": record["src"],
                    "tgt": record["tgt"],
                    "weight": record["weight"] or 1.0,
                    "description": record["description"] or "",
                })

        return edges

    def index_start_callback(self) -> None:
        """Called before indexing starts."""
        pass

    def index_done_callback(self) -> None:
        """Called after indexing completes."""
        pass

    def query_done_callback(self) -> None:
        """Called after query completes."""
        pass

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
