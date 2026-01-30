"""Hierarchical entity extraction component for HiRAG.

This module implements the two-stage entity extraction strategy from HiRAG:
1. Extract entities from chunks
2. Extract relations using entity embeddings
3. Perform hierarchical clustering
4. Merge clustered entities

This is the core innovation that differentiates HiRAG from other GraphRAG approaches.
"""

import json
import re
from typing import Any, Callable, List, Optional

from haystack import component
from haystack.dataclasses import Document
from haystack.dataclasses.chat_message import ChatMessage

from hirag_haystack.core.graph import Entity, Relation, NodeType
from hirag_haystack.prompts import (
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
)
from hirag_haystack.utils.token_utils import (
    split_string_by_multi_markers,
    compute_mdhash_id,
)


@component
class HierarchicalClusterDetector:
    """Detect hierarchical clusters in entities using embeddings.

    This component performs clustering on entity embeddings to identify
    groups of related entities, which form the basis for hierarchical knowledge.
    """

    def __init__(
        self,
        embedding_func: Callable[[List[str]], List[List[float]]] | None = None,
        cluster_algorithm: str = "agglomerative",
        n_clusters: int = 5,
        max_cluster_size: int = 10,
    ):
        """Initialize the hierarchical cluster detector.

        Args:
            embedding_func: Function to generate embeddings.
            cluster_algorithm: Clustering algorithm ("agglomerative" or "kmeans").
            n_clusters: Number of clusters for K-means (if applicable).
            max_cluster_size: Maximum size for a cluster before splitting.
        """
        self.embedding_func = embedding_func
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        self.max_cluster_size = max_cluster_size

        # Lazy imports
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            from sklearn.cluster import AgglomerativeClustering, KMeans
            from sklearn.metrics.pairwise import cosine_distances
            return True
        except ImportError:
            return False

    @component.output_types(
        clustered_entities=list,
        cluster_assignments=dict,
    )
    def run(
        self,
        entities: List[Entity],
        global_config: dict | None = None,
    ) -> dict:
        """Perform hierarchical clustering on entities.

        Args:
            entities: List of Entity objects with embeddings.
            global_config: Optional global configuration.

        Returns:
            Dictionary with:
                - clustered_entities: List of entities with cluster assignments
                - cluster_assignments: Dict mapping cluster_id to entity names
        """
        if not entities:
            return {"clustered_entities": [], "cluster_assignments": {}}

        if not self._sklearn_available:
            # Fallback: assign all to single cluster
            for entity in entities:
                entity.add_cluster(level=0, cluster_id="cluster_0")
            return {
                "clustered_entities": entities,
                "cluster_assignments": {"cluster_0": [e.entity_name for e in entities]},
            }

        # Extract embeddings and entity names
        entity_names = [e.entity_name for e in entities]
        embeddings = []

        for entity in entities:
            if entity.embedding:
                embeddings.append(entity.embedding)
            elif self.embedding_func:
                # Generate embedding on the fly
                emb_result = self.embedding_func([entity.description])
                embeddings.append(emb_result[0])
                entity.embedding = emb_result[0]
            else:
                # Fallback: use zero embedding
                embeddings.append([0.0] * 1536)

        # Perform clustering
        clusters = self._perform_clustering(embeddings)

        # Assign clusters to entities
        cluster_assignments = {}

        for entity, cluster_id in zip(entities, clusters):
            entity.add_cluster(level=0, cluster_id=f"cluster_{cluster_id}")

            if f"cluster_{cluster_id}" not in cluster_assignments:
                cluster_assignments[f"cluster_{cluster_id}"] = []
            cluster_assignments[f"cluster_{cluster_id}"].append(entity.entity_name)

        return {
            "clustered_entities": entities,
            "cluster_assignments": cluster_assignments,
        }

    def _perform_clustering(self, embeddings: List[List[float]]) -> List[int]:
        """Perform clustering on embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            List of cluster assignments.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances
        import numpy as np

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Compute distance matrix (cosine distance)
        distance_matrix = cosine_distances(embeddings_array)

        # Perform hierarchical clustering
        if self.cluster_algorithm == "agglomerative":
            # Determine number of clusters based on max size
            n_clusters = max(1, len(embeddings) // self.max_cluster_size)

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
            return clustering.fit_predict(distance_matrix)
        else:
            # K-means fallback
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            return kmeans.fit_predict(embeddings_array)


@component
class HierarchicalEntityExtractor:
    """Two-stage entity extractor with hierarchical clustering.

    Stage 1: Extract entities from each text chunk
    Stage 2: Extract relations using entity context
    Stage 3: Cluster entities hierarchically
    Stage 4: Merge and deduplicate entities
    """

    def __init__(
        self,
        generator: Any = None,
        entity_types: List[str] | None = None,
        max_gleaning: int = 1,
        enable_clustering: bool = True,
        embedding_func: Callable[[List[str]], List[List[float]]] | None = None,
    ):
        """Initialize the hierarchical entity extractor.

        Args:
            generator: LLM generator for extraction.
            entity_types: List of entity types to extract.
            max_gleaning: Maximum gleaning iterations.
            enable_clustering: Whether to perform hierarchical clustering.
            embedding_func: Function for generating embeddings.
        """
        self.generator = generator
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.max_gleaning = max_gleaning
        self.enable_clustering = enable_clustering
        self.embedding_func = embedding_func

        # Delimiters
        self.tuple_delimiter = DEFAULT_TUPLE_DELIMITER
        self.record_delimiter = DEFAULT_RECORD_DELIMITER
        self.completion_delimiter = DEFAULT_COMPLETION_DELIMITER

    @component.output_types(
        entities=list,
        relations=list,
        clustered_entities=list,
    )
    def run(
        self,
        documents: List[Document],
        global_config: dict | None = None,
    ) -> dict:
        """Extract entities and relations with hierarchical clustering.

        Args:
            documents: List of Documents (text chunks).
            global_config: Optional global configuration.

        Returns:
            Dictionary with:
                - entities: List of all extracted entities
                - relations: List of all extracted relations
                - clustered_entities: Entities with cluster assignments
        """
        if not documents:
            return {
                "entities": [],
                "relations": [],
                "clustered_entities": [],
            }

        all_entities = []
        all_relations = []
        chunk_entity_map = {}  # chunk_id -> entities in that chunk

        # Stage 1: Extract entities from each chunk
        for doc in documents:
            chunk_id = doc.id or f"chunk_{hash(doc.content)}"
            content = doc.content

            # Extract entities
            entities = self._extract_entities_from_text(content, chunk_id)
            all_entities.extend(entities)
            chunk_entity_map[chunk_id] = entities

        # Deduplicate and merge entities
        unique_entities = self._merge_entities(all_entities)

        # Generate embeddings for entities
        if self.embedding_func:
            unique_entities = self._generate_embeddings(unique_entities)

        # Stage 2: Extract relations using entity context
        for doc in documents:
            chunk_id = doc.id or f"chunk_{hash(doc.content)}"
            content = doc.content
            entities_in_chunk = chunk_entity_map.get(chunk_id, [])
            entity_names = [e.entity_name for e in entities_in_chunk]

            if entity_names:
                relations = self._extract_relations_from_text(
                    content,
                    chunk_id,
                    entity_names,
                )
                all_relations.extend(relations)

        # Deduplicate and merge relations
        unique_relations = self._merge_relations(all_relations)

        # Stage 3: Hierarchical clustering
        if self.enable_clustering:
            cluster_detector = HierarchicalClusterDetector(
                embedding_func=self.embedding_func,
            )
            cluster_result = cluster_detector.run(entities=unique_entities)
            clustered_entities = cluster_result.get("clustered_entities", unique_entities)
        else:
            clustered_entities = unique_entities

        return {
            "entities": unique_entities,
            "relations": unique_relations,
            "clustered_entities": clustered_entities,
        }

    def _extract_entities_from_text(
        self,
        text: str,
        chunk_id: str,
    ) -> List[Entity]:
        """Extract entities from text using LLM.

        Args:
            text: Input text.
            chunk_id: Source chunk ID.

        Returns:
            List of extracted Entity objects.
        """
        prompt = self._get_entity_extraction_prompt().format(
            entity_types=",".join(self.entity_types),
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            input_text=text[:4000],
        )

        result = self._call_llm(prompt)
        entities = self._parse_entities(result, chunk_id)

        # Gleaning
        if entities and self.max_gleaning > 0:
            entities = self._gleaning_entities(text, chunk_id, entities)

        return entities

    def _extract_relations_from_text(
        self,
        text: str,
        chunk_id: str,
        entity_names: List[str],
    ) -> List[Relation]:
        """Extract relations from text using LLM.

        Args:
            text: Input text.
            chunk_id: Source chunk ID.
            entity_names: Known entity names.

        Returns:
            List of extracted Relation objects.
        """
        if not entity_names:
            return []

        prompt = self._get_relation_extraction_prompt().format(
            entities=",".join(entity_names[:50]),
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            input_text=text[:4000],
        )

        result = self._call_llm(prompt)
        relations = self._parse_relations(result, chunk_id)

        # Gleaning
        if relations and self.max_gleaning > 0:
            relations = self._gleaning_relations(text, chunk_id, relations, entity_names)

        return relations

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with prompt."""
        if self.generator is None:
            raise ValueError("Generator not configured")

        # Wrap prompt in a ChatMessage for Haystack 2.x compatibility
        message = ChatMessage.from_user(prompt)
        response = self.generator.run(messages=[message])
        if hasattr(response, "replies"):
            return response.replies[0].text if response.replies else ""
        return str(response)

    def _parse_entities(self, text: str, chunk_id: str) -> List[Entity]:
        """Parse entities from LLM output."""
        entities = []
        records = split_string_by_multi_markers(
            text,
            [self.record_delimiter, self.completion_delimiter],
        )

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            fields = self._split_tuple(match.group(1))
            if len(fields) < 4 or fields[0] != '"entity"':
                continue

            entity_name = self._clean_str(fields[1]).upper()
            if not entity_name:
                continue

            entity_type = self._clean_str(fields[2]).upper()
            description = self._clean_str(fields[3]) if len(fields) > 3 else ""

            entities.append(Entity(
                entity_name=entity_name,
                entity_type=entity_type,
                description=description,
                source_id=chunk_id,
            ))

        return entities

    def _parse_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Parse relations from LLM output."""
        relations = []
        records = split_string_by_multi_markers(
            text,
            [self.record_delimiter, self.completion_delimiter],
        )

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            fields = self._split_tuple(match.group(1))
            if len(fields) < 4 or fields[0] != '"relationship"':
                continue

            src = self._clean_str(fields[1]).upper()
            tgt = self._clean_str(fields[2]).upper()
            description = self._clean_str(fields[3]) if len(fields) > 3 else ""

            weight = 1.0
            if len(fields) > 4:
                try:
                    weight = float(fields[-1])
                except ValueError:
                    pass

            relations.append(Relation(
                src_id=src,
                tgt_id=tgt,
                weight=weight,
                description=description,
                source_id=chunk_id,
            ))

        return relations

    def _split_tuple(self, tuple_str: str) -> List[str]:
        """Split tuple string by delimiter."""
        return [s.strip() for s in tuple_str.split(self.tuple_delimiter)]

    def _clean_str(self, s: str) -> str:
        """Clean extracted string."""
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        return s

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities."""
        merged = {}

        for entity in entities:
            name = entity.entity_name

            if name in merged:
                existing = merged[name]

                # Combine source IDs
                existing_sources = set(existing.source_id.split("|"))
                new_sources = set(entity.source_id.split("|"))
                merged[name].source_id = "|".join(existing_sources | new_sources)

                # Merge descriptions
                if entity.description and entity.description not in existing.description:
                    if existing.description:
                        merged[name].description += " | " + entity.description
                    else:
                        merged[name].description = entity.description
            else:
                merged[name] = entity

        return list(merged.values())

    def _merge_relations(self, relations: List[Relation]) -> List[Relation]:
        """Merge duplicate relations."""
        merged = {}

        for relation in relations:
            key = relation.sorted_pair

            if key in merged:
                existing = merged[key]
                existing.weight += relation.weight

                # Combine source IDs
                existing_sources = set(existing.source_id.split("|"))
                new_sources = set(relation.source_id.split("|"))
                existing.source_id = "|".join(existing_sources | new_sources)

                # Merge descriptions
                if relation.description and relation.description not in existing.description:
                    if existing.description:
                        existing.description += " | " + relation.description
                    else:
                        existing.description = relation.description
            else:
                merged[key] = relation

        return list(merged.values())

    def _generate_embeddings(self, entities: List[Entity]) -> List[Entity]:
        """Generate embeddings for entities."""
        if not self.embedding_func:
            return entities

        # Prepare descriptions
        descriptions = [e.description or e.entity_name for e in entities]

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            embeddings = self.embedding_func(batch)
            all_embeddings.extend(embeddings)

        # Assign embeddings
        for entity, embedding in zip(entities, all_embeddings):
            entity.embedding = embedding

        return entities

    def _gleaning_entities(
        self,
        text: str,
        chunk_id: str,
        initial_entities: List[Entity],
    ) -> List[Entity]:
        """Perform gleaning for missed entities."""
        current_entities = initial_entities

        for _ in range(self.max_gleaning):
            check_prompt = self._get_if_loop_prompt().format(
                extracted_count=len(current_entities)
            )
            result = self._call_llm(check_prompt)

            if "no" in result.lower().strip('"\' '):
                break

            continue_prompt = self._get_continue_prompt()
            result = self._call_llm(continue_prompt)
            new_entities = self._parse_entities(result, chunk_id)

            if not new_entities:
                break

            current_entities.extend(new_entities)

        return current_entities

    def _gleaning_relations(
        self,
        text: str,
        chunk_id: str,
        initial_relations: List[Relation],
        entity_names: List[str],
    ) -> List[Relation]:
        """Perform gleaning for missed relations."""
        current_relations = initial_relations

        for _ in range(self.max_gleaning):
            check_prompt = self._get_if_loop_prompt().format(
                extracted_count=len(current_relations)
            )
            result = self._call_llm(check_prompt)

            if "no" in result.lower().strip('"\' '):
                break

            continue_prompt = self._get_continue_prompt()
            result = self._call_llm(continue_prompt)
            new_relations = self._parse_relations(result, chunk_id)

            if not new_relations:
                break

            current_relations.extend(new_relations)

        return current_relations

    def _get_entity_extraction_prompt(self) -> str:
        """Get entity extraction prompt."""
        from hirag_haystack.prompts import HI_ENTITY_EXTRACTION_PROMPT
        return HI_ENTITY_EXTRACTION_PROMPT

    def _get_relation_extraction_prompt(self) -> str:
        """Get relation extraction prompt."""
        from hirag_haystack.prompts import HI_RELATION_EXTRACTION_PROMPT
        return HI_RELATION_EXTRACTION_PROMPT

    def _get_if_loop_prompt(self) -> str:
        """Get if-loop prompt."""
        from hirag_haystack.prompts import ENTITY_IF_LOOP_EXTRACTION
        return ENTITY_IF_LOOP_EXTRACTION

    def _get_continue_prompt(self) -> str:
        """Get continue prompt."""
        from hirag_haystack.prompts import ENTITY_CONTINUE_EXTRACTION
        return ENTITY_CONTINUE_EXTRACTION

    def _generate_summary_entities(
        self,
        community_entities: list[Entity],
        community_relations: list[Relation],
        level: int = 1,
    ) -> tuple[list[Entity], list[Relation]]:
        """Generate summary entities for hierarchical knowledge graph.

        This implements the meta-summary entity generation from the HiRAG paper,
        where we create higher-level abstractions (𝒳 concept sets) to guide
        cross-document reasoning.

        Args:
            community_entities: Entities in the community.
            community_relations: Relations in the community.
            level: Current hierarchy level.

        Returns:
            Tuple of (summary_entities, hierarchical_relations).
        """
        if not community_entities:
            return [], []

        # Prepare entity information for the prompt
        entities_info = "\n".join([
            f"- {e.entity_name} ({e.entity_type}): {e.description}"
            for e in community_entities[:50]  # Limit for token constraints
        ])

        # Prepare relation information
        relations_info = "\n".join([
            f"- {r.src_id} -> {r.tgt_id}: {r.description}"
            for r in community_relations[:50]
        ])

        # Get meta concepts for this level
        from hirag_haystack.prompts import (
            SUMMARY_ENTITY_EXTRACTION_PROMPT,
            META_SUMMARY_CONCEPTS,
            DEFAULT_TUPLE_DELIMITER,
            DEFAULT_RECORD_DELIMITER,
            DEFAULT_COMPLETION_DELIMITER,
        )

        meta_concepts = META_SUMMARY_CONCEPTS.get(
            "GENERAL", []
        ) + META_SUMMARY_CONCEPTS.get("RELATIONAL", [])

        prompt = SUMMARY_ENTITY_EXTRACTION_PROMPT.format(
            entities_info=entities_info,
            relations_info=relations_info,
            meta_concepts=", ".join(meta_concepts[:10]),
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        )

        try:
            result = self._call_llm(prompt)
            summary_entities = self._parse_summary_entities(result, level)
        except Exception:
            summary_entities = []

        # Generate hierarchical relations (connecting summary to detailed entities)
        hierarchical_relations = self._generate_hierarchical_relations(
            summary_entities,
            community_entities,
            level,
        )

        return summary_entities, hierarchical_relations

    def _parse_summary_entities(
        self,
        text: str,
        level: int,
    ) -> list[Entity]:
        """Parse summary entities from LLM output."""
        entities = []
        records = split_string_by_multi_markers(
            text,
            [self.record_delimiter, self.completion_delimiter],
        )

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            fields = self._split_tuple(match.group(1))
            if len(fields) < 3 or fields[0] != '"summary_entity"':
                continue

            entity_name = self._clean_str(fields[1]).upper()
            if not entity_name:
                continue

            entity_type = self._clean_str(fields[2]).upper()
            description = self._clean_str(fields[3]) if len(fields) > 3 else ""

            # Add level prefix to avoid conflicts
            prefixed_name = f"META_L{level}_{entity_name}"

            entities.append(Entity(
                entity_name=prefixed_name,
                entity_type=f"SUMMARY_{entity_type}",
                description=f"[Level {level}] {description}",
                source_id=f"meta_level_{level}",
            ))

        return entities

    def _generate_hierarchical_relations(
        self,
        summary_entities: list[Entity],
        detailed_entities: list[Entity],
        level: int,
    ) -> list[Relation]:
        """Generate hierarchical relations between summary and detailed entities.

        Args:
            summary_entities: Generated summary entities.
            detailed_entities: Original detailed entities.
            level: Current hierarchy level.

        Returns:
            List of hierarchical relations.
        """
        if not summary_entities or not detailed_entities:
            return []

        relations = []

        # Create relations from each summary entity to relevant detailed entities
        summary_names = [e.entity_name for e in summary_entities]
        detailed_names = [e.entity_name for e in detailed_entities]

        # Simple heuristic: connect summary to entities that appear in its description
        for summary in summary_entities:
            for detailed in detailed_entities:
                # Check if detailed entity name appears in summary description
                if detailed.entity_name in summary.description:
                    relations.append(Relation(
                        src_id=summary.entity_name,
                        tgt_id=detailed.entity_name,
                        weight=0.5,
                        description=f"generalizes",
                        source_id=f"meta_level_{level}",
                    ))

        return relations
