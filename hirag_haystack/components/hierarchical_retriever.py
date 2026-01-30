"""Hierarchical retrieval component for HiRAG.

This component implements the multi-mode retrieval strategy, supporting:
- naive: Basic document chunk retrieval
- hi_local: Local entity and relationship knowledge
- hi_global: Global community report knowledge
- hi_bridge: Cross-community reasoning paths
- hi: Full hierarchical retrieval combining all modes
"""

import json
from typing import Any

from haystack import component

from hirag_haystack._logging import get_logger, trace
from hirag_haystack.core.query_param import QueryParam
from hirag_haystack.stores.base import GraphDocumentStore


@component
class EntityRetriever:
    """Retrieve entities relevant to a query using vector similarity."""

    def __init__(
        self,
        entity_store: Any = None,
        top_k: int = 20,
    ):
        """Initialize the EntityRetriever.

        Args:
            entity_store: Document store for entity embeddings.
            top_k: Number of entities to retrieve.
        """
        self.entity_store = entity_store
        self.top_k = top_k

    @component.output_types(entities=list, entity_names=list)
    def run(
        self,
        query_embedding: list[float] | None = None,
        query: str | None = None,
        top_k: int | None = None,
    ) -> dict:
        """Retrieve relevant entities.

        Args:
            query_embedding: Pre-computed query embedding.
            query: Query text (if embedding not provided).
            top_k: Override default top_k.

        Returns:
            Dictionary with retrieved entities and their names.
        """
        if self.entity_store is None:
            return {"entities": [], "entity_names": []}

        k = top_k or self.top_k

        # Query the entity store
        if query_embedding:
            results = self.entity_store.query_by_embedding(query_embedding, top_k=k)
        elif query:
            results = self.entity_store.query(query, top_k=k)
        else:
            return {"entities": [], "entity_names": []}

        entities = []
        entity_names = []

        for doc in results:
            # Handle both Document objects and dictionaries
            if isinstance(doc, dict):
                entity_name = doc.get("entity_name", doc.get("id", ""))
                content = doc.get("content", "")
                meta = doc.get("meta", {})
            else:
                entity_name = doc.meta.get("entity_name", doc.id)
                content = doc.content
                meta = doc.meta

            entity_names.append(entity_name)
            entities.append({
                "entity_name": entity_name,
                "description": content,
                "meta": meta,
            })

        return {"entities": entities, "entity_names": entity_names}


@component
class HierarchicalRetriever:
    """Multi-mode hierarchical knowledge retriever.

    Implements the HiRAG retrieval strategy with support for multiple modes:
    - naive: Basic RAG with document chunks
    - hi_local: Local knowledge (entities + relations + chunks)
    - hi_global: Global knowledge (community reports + chunks)
    - hi_bridge: Bridge knowledge (cross-community paths)
    - hi_nobridge: Hierarchical without bridge paths
    - hi: Full hierarchical (all of the above)
    """

    def __init__(
        self,
        graph_store: GraphDocumentStore | None = None,
        chunk_store: Any = None,
        top_k: int = 20,
        top_m: int = 10,
    ):
        """Initialize the HierarchicalRetriever.

        Args:
            graph_store: Graph store for entity/relation queries.
            chunk_store: Document store for text chunks.
            top_k: Number of entities to retrieve.
            top_m: Number of key entities per community.
        """
        self.graph_store = graph_store
        self.chunk_store = chunk_store
        self.top_k = top_k
        self.top_m = top_m

        # Logger
        self._logger = get_logger("hierarchical_retriever")

    @component.output_types(context=str)
    def run(
        self,
        query: str,
        retrieved_entities: list | None = None,
        communities: dict | None = None,
        community_reports: dict | None = None,
        mode: str = "hi",
        param: QueryParam | None = None,
    ) -> dict:
        """Retrieve context based on the specified mode.

        Args:
            query: User query string.
            retrieved_entities: List of entities retrieved from vector store.
            communities: Dictionary of communities.
            community_reports: Dictionary of community report strings.
            mode: Retrieval mode (naive, hi_local, hi_global, hi_bridge, hi).
            param: QueryParam with detailed configuration.

        Returns:
            Dictionary with assembled context string.
        """
        param = param or QueryParam(mode=mode)

        entity_count = len(retrieved_entities) if retrieved_entities else 0
        self._logger.info(f"Retrieval mode={mode}, entities={entity_count}")

        if mode == "naive":
            context = self._naive_retrieve(query, param)
            self._logger.debug(f"Naive context length={len(context)}")
            return {"context": context}
        elif mode == "hi_local":
            context = self._local_retrieve(query, retrieved_entities, param)
            self._logger.debug(f"Local context length={len(context)}")
            return {"context": context}
        elif mode == "hi_global":
            context = self._global_retrieve(
                query, retrieved_entities, communities, community_reports, param
            )
            self._logger.debug(f"Global context length={len(context)}")
            return {"context": context}
        elif mode == "hi_bridge":
            context = self._bridge_retrieve(
                query, retrieved_entities, communities, param
            )
            self._logger.debug(f"Bridge context length={len(context)}")
            return {"context": context}
        elif mode == "hi":
            context = self._hierarchical_retrieve(
                query, retrieved_entities, communities, community_reports, param
            )
            self._logger.debug(f"Hierarchical context length={len(context)}")
            return {"context": context}
        elif mode == "hi_nobridge":
            context = self._nobridge_retrieve(
                query, retrieved_entities, communities, community_reports, param
            )
            self._logger.debug(f"No-bridge context length={len(context)}")
            return {"context": context}
        else:
            return {"context": ""}

    def _naive_retrieve(self, query: str, param: QueryParam) -> str:
        """Basic RAG retrieval with document chunks."""
        if self.chunk_store is None:
            return ""

        results = self.chunk_store.query(query, top_k=param.top_k)

        chunks = []
        for doc in results:
            chunks.append(doc.content)

        # Join with separator
        return "\n\n---\n\n".join(chunks[:param.top_k])

    def _local_retrieve(
        self,
        query: str,
        retrieved_entities: list | None,
        param: QueryParam,
    ) -> str:
        """Local knowledge retrieval: entities + relations + chunks."""
        if not retrieved_entities or self.graph_store is None:
            return self._naive_retrieve(query, param)

        entity_names = [e.get("entity_name") for e in retrieved_entities[:param.top_k]]
        self._logger.debug(f"Local retrieve: {len(entity_names)} entities")

        # Get entity details
        entities_section = self._build_entities_section(entity_names)
        relations_section = self._build_relations_section(entity_names)
        text_units_section = self._build_text_units_section(entity_names)

        return f"""-----Entities-----
{entities_section}

-----Relations-----
{relations_section}

-----Sources-----
{text_units_section}"""

    def _global_retrieve(
        self,
        query: str,
        retrieved_entities: list | None,
        communities: dict | None,
        community_reports: dict | None,
        param: QueryParam,
    ) -> str:
        """Global knowledge retrieval: community reports + chunks."""
        if not retrieved_entities or not communities:
            return self._naive_retrieve(query, param)

        # Find relevant communities
        entity_names = [e.get("entity_name") for e in retrieved_entities[:param.top_k * 10]]
        relevant_communities = self._find_relevant_communities(
            entity_names, communities, param
        )
        self._logger.debug(f"Global retrieve: {len(relevant_communities)} relevant communities")

        # Build sections
        communities_section = self._build_communities_section(
            relevant_communities, community_reports
        )
        text_units_section = self._build_text_units_section(entity_names[:param.top_k])

        return f"""-----Backgrounds-----
{communities_section}

-----Source Documents-----
{text_units_section}"""

    def _bridge_retrieve(
        self,
        query: str,
        retrieved_entities: list | None,
        communities: dict | None,
        param: QueryParam,
    ) -> str:
        """Bridge knowledge retrieval: cross-community reasoning paths."""
        if not retrieved_entities or not communities or self.graph_store is None:
            return ""

        entity_names = [e.get("entity_name") for e in retrieved_entities[:param.top_k * 10]]

        # Find key entities per community
        key_entities = self._find_key_entities_per_community(
            entity_names, communities, param
        )
        self._logger.debug(f"Bridge retrieve: {len(key_entities)} key entities")

        # Find reasoning path
        path = self._find_reasoning_path(key_entities)
        trace(self._logger, f"Reasoning path length: {len(path)}")
        path_section = self._build_path_section(path)

        text_units_section = self._build_text_units_section(entity_names[:param.top_k])

        return f"""-----Reasoning Path-----
{path_section}

-----Source Documents-----
{text_units_section}"""

    def _nobridge_retrieve(
        self,
        query: str,
        retrieved_entities: list | None,
        communities: dict | None,
        community_reports: dict | None,
        param: QueryParam,
    ) -> str:
        """Hierarchical retrieval without bridge paths.

        Combines local and global knowledge but excludes cross-community
        reasoning paths. This is useful when the query is focused on
        understanding entities within their community context.
        """
        if not retrieved_entities or not communities or self.graph_store is None:
            return self._naive_retrieve(query, param)

        entity_names = [e.get("entity_name") for e in retrieved_entities[:param.top_k * 10]]
        top_entities = entity_names[:param.top_k]

        # Get relevant communities
        relevant_communities = self._find_relevant_communities(
            top_entities, communities, param
        )

        # Build sections (no reasoning path)
        entities_section = self._build_entities_section(top_entities)
        relations_section = self._build_relations_section(top_entities)
        communities_section = self._build_communities_section(
            relevant_communities, community_reports
        )
        text_units_section = self._build_text_units_section(top_entities)

        return f"""-----Reports-----
{communities_section}

-----Entities-----
{entities_section}

-----Relationships-----
{relations_section}

-----Sources-----
{text_units_section}"""

    def _hierarchical_retrieve(
        self,
        query: str,
        retrieved_entities: list | None,
        communities: dict | None,
        community_reports: dict | None,
        param: QueryParam,
    ) -> str:
        """Full hierarchical retrieval combining all knowledge sources."""
        if not retrieved_entities or not communities or self.graph_store is None:
            return self._naive_retrieve(query, param)

        entity_names = [e.get("entity_name") for e in retrieved_entities[:param.top_k * 10]]
        top_entities = entity_names[:param.top_k]

        # Get relevant communities
        relevant_communities = self._find_relevant_communities(
            top_entities, communities, param
        )

        # Find reasoning path through key entities
        key_entities = self._find_key_entities_per_community(
            entity_names, relevant_communities, param
        )
        path = self._find_reasoning_path(key_entities)

        # Build all sections
        entities_section = self._build_entities_section(top_entities)
        communities_section = self._build_communities_section(
            relevant_communities, community_reports
        )
        path_section = self._build_path_section(path)
        text_units_section = self._build_text_units_section(top_entities)

        return f"""-----Backgrounds-----
{communities_section}

-----Reasoning Path-----
{path_section}

-----Detail Entity Information-----
{entities_section}

-----Source Documents-----
{text_units_section}"""

    # ===== Helper Methods =====

    def _build_entities_section(self, entity_names: list[str]) -> str:
        """Build CSV-formatted entities section."""
        lines = ["id,entity,type,description,rank"]

        for i, name in enumerate(entity_names[:50]):
            if self.graph_store is None:
                break
            node_data = self.graph_store.get_node(name)
            if node_data:
                rank = self.graph_store.node_degree(name)
                lines.append(
                    f"{i},{name},{node_data.get('entity_type', 'UNKNOWN')},"
                    f'"{node_data.get("description", "")[:50]}",{rank}'
                )

        return "\n".join(lines)

    def _build_relations_section(self, entity_names: list[str]) -> str:
        """Build CSV-formatted relations section."""
        if self.graph_store is None:
            return ""

        lines = ["id,source,target,description,weight,rank"]
        edges_seen = set()

        for i, name in enumerate(entity_names[:20]):
            node_edges = self.graph_store.get_node_edges(name)
            for src, tgt in node_edges:
                edge_key = tuple(sorted((src, tgt)))
                if edge_key in edges_seen:
                    continue
                edges_seen.add(edge_key)

                edge_data = self.graph_store.get_edge(src, tgt)
                if edge_data:
                    rank = self.graph_store.edge_degree(src, tgt)
                    weight = edge_data.get("weight", 1.0)
                    desc = edge_data.get("description", "")[:50]
                    lines.append(f"{len(edges_seen)},{src},{tgt},\"{desc}\",{weight},{rank}")

            if len(edges_seen) >= 50:
                break

        return "\n".join(lines)

    def _build_text_units_section(self, entity_names: list[str]) -> str:
        """Build text units section from entity sources."""
        if self.chunk_store is None or self.graph_store is None:
            return ""

        lines = ["id,content"]
        chunk_ids_seen = set()

        for name in entity_names[:20]:
            node_data = self.graph_store.get_node(name)
            if not node_data:
                continue

            source_ids = node_data.get("source_id", "").split("|")
            for chunk_id in source_ids:
                if chunk_id in chunk_ids_seen:
                    continue
                chunk_ids_seen.add(chunk_id)

                # Get chunk content
                doc = self.chunk_store.get_document_by_chunk_id(chunk_id)
                if doc:
                    content = doc.content[:200].replace("\n", " ")
                    lines.append(f"{len(chunk_ids_seen)},\"{content}\"")

            if len(chunk_ids_seen) >= 10:
                break

        return "\n".join(lines)

    def _build_communities_section(
        self,
        communities: dict,
        reports: dict | None,
    ) -> str:
        """Build communities section."""
        lines = ["id,content"]

        for i, (comm_id, community) in enumerate(list(communities.items())[:5]):
            report = reports.get(comm_id, "") if reports else community.report_string
            content = report[:300].replace("\n", " ")
            lines.append(f"{i},\"{content}\"")

        return "\n".join(lines)

    def _build_path_section(self, path: list[str]) -> str:
        """Build reasoning path section."""
        if not path or self.graph_store is None:
            return "id,source,target,description,weight,rank\n"

        lines = ["id,source,target,description,weight,rank"]

        # Get edges along the path
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            edge_data = self.graph_store.get_edge(src, tgt)
            if edge_data:
                rank = self.graph_store.edge_degree(src, tgt)
                desc = edge_data.get("description", "")[:50]
                weight = edge_data.get("weight", 1.0)
                lines.append(f"{i},{src},{tgt},\"{desc}\",{weight},{rank}")
            else:
                lines.append(f"{i},{src},{tgt},\"\",1.0,0")

        return "\n".join(lines)

    def _find_relevant_communities(
        self,
        entity_names: list[str],
        communities: dict,
        param: QueryParam,
    ) -> dict:
        """Find communities relevant to the retrieved entities."""
        community_counts = {}

        for name in entity_names:
            if self.graph_store is None:
                continue
            node_data = self.graph_store.get_node(name)
            if not node_data:
                continue

            # Get cluster assignments
            clusters_str = node_data.get("clusters", "[]")
            try:
                clusters = json.loads(clusters_str) if isinstance(clusters_str, str) else clusters_str
                for cluster in clusters:
                    if cluster.get("level", 0) <= param.level:
                        comm_id = cluster.get("cluster", "")
                        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        # Sort by count and return top communities
        sorted_communities = sorted(
            community_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            comm_id: communities[comm_id]
            for comm_id, _ in sorted_communities
            if comm_id in communities
        }

    def _find_key_entities_per_community(
        self,
        entity_names: list[str],
        communities: dict,
        param: QueryParam,
    ) -> list[str]:
        """Find key entities within each relevant community."""
        key_entities = []

        for community in communities.values():
            # Get top-m entities from this community
            community_entities = [
                e for e in entity_names
                if e in community.nodes
            ][:param.top_m]
            key_entities.extend(community_entities)

        return list(set(key_entities))

    def _find_reasoning_path(self, key_entities: list[str]) -> list[str]:
        """Find a reasoning path connecting key entities."""
        if not key_entities or self.graph_store is None:
            return []

        # Simple approach: shortest path through first and last entities
        if len(key_entities) < 2:
            return key_entities

        start = key_entities[0]
        end = key_entities[-1]

        if len(key_entities) == 2:
            return self.graph_store.shortest_path(start, end)

        # Path through intermediate entities
        full_path = []
        current = start

        for next_entity in key_entities[1:]:
            segment = self.graph_store.shortest_path(current, next_entity)
            if full_path:
                full_path.extend(segment[1:])  # Avoid duplicate
            else:
                full_path.extend(segment)
            current = next_entity

        return full_path


@component
class ContextBuilder:
    """Build final context for prompt generation.

    Combines retrieved information into a formatted context string.
    """

    @component.output_types(context=str)
    def run(
        self,
        entities_context: str = "",
        relations_context: str = "",
        communities_context: str = "",
        reasoning_path_context: str = "",
        text_units_context: str = "",
        mode: str = "hi",
    ) -> dict:
        """Build final context based on mode and available sections."""
        sections = []

        if mode == "naive":
            sections.append(text_units_context)
        elif mode == "hi_local":
            if entities_context:
                sections.append(f"-----Entities-----\n{entities_context}")
            if relations_context:
                sections.append(f"-----Relations-----\n{relations_context}")
            if text_units_context:
                sections.append(f"-----Sources-----\n{text_units_context}")
        elif mode == "hi_global":
            if communities_context:
                sections.append(f"-----Backgrounds-----\n{communities_context}")
            if text_units_context:
                sections.append(f"-----Sources-----\n{text_units_context}")
        elif mode == "hi_bridge":
            if reasoning_path_context:
                sections.append(f"-----Reasoning Path-----\n{reasoning_path_context}")
            if text_units_context:
                sections.append(f"-----Sources-----\n{text_units_context}")
        else:  # hi
            if communities_context:
                sections.append(f"-----Backgrounds-----\n{communities_context}")
            if reasoning_path_context:
                sections.append(f"-----Reasoning Path-----\n{reasoning_path_context}")
            if entities_context:
                sections.append(f"-----Entities-----\n{entities_context}")
            if text_units_context:
                sections.append(f"-----Sources-----\n{text_units_context}")

        return {"context": "\n\n".join(sections)}
