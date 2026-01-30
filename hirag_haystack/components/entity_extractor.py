"""Entity and relationship extraction component.

This component uses LLMs to extract entities and relationships from text chunks,
supporting the hierarchical extraction approach used in HiRAG.
"""

import re
from dataclasses import dataclass
from typing import Any

from haystack import component
from haystack.dataclasses import Document
from haystack.dataclasses.chat_message import ChatMessage

from hirag_haystack._logging import get_logger, trace
from hirag_haystack.core.graph import Entity, Relation


# Default entity types to extract
DEFAULT_ENTITY_TYPES = [
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
    "TECHNICAL_TERM",
]

# Delimiters for parsing LLM output
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##SPLITTER##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETION|>"


@component
class EntityExtractor:
    """Extract entities and relationships from documents using LLM.

    This component implements the entity extraction strategy from HiRAG,
    including:
    - Entity extraction with type classification
    - Relationship extraction between entities
    - Gleaning mechanism to catch missed entities
    - Entity description summarization
    """

    def __init__(
        self,
        generator: Any = None,
        entity_types: list[str] | None = None,
        max_gleaning: int = 1,
        summary_max_tokens: int = 500,
        tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
        record_delimiter: str = DEFAULT_RECORD_DELIMITER,
        completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
    ):
        """Initialize the EntityExtractor.

        Args:
            generator: Haystack ChatGenerator for LLM calls.
            entity_types: List of entity types to extract.
            max_gleaning: Maximum number of gleaning iterations.
            summary_max_tokens: Maximum tokens for entity summaries.
            tuple_delimiter: Delimiter for tuple fields in LLM output.
            record_delimiter: Delimiter between records in LLM output.
            completion_delimiter: Delimiter marking end of output.
        """
        self.generator = generator
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.max_gleaning = max_gleaning
        self.summary_max_tokens = summary_max_tokens
        self.tuple_delimiter = tuple_delimiter
        self.record_delimiter = record_delimiter
        self.completion_delimiter = completion_delimiter

        # Logger
        self._logger = get_logger("entity_extractor")

    @component.output_types(entities=list, relations=list)
    def run(
        self,
        documents: list[Document],
        prompt_template: str | None = None,
    ) -> dict:
        """Extract entities and relations from documents.

        Args:
            documents: List of Documents to process.
            prompt_template: Optional custom prompt template.

        Returns:
            Dictionary with:
                - entities: List of Entity objects
                - relations: List of Relation objects
        """
        if not documents:
            return {"entities": [], "relations": []}

        self._logger.info(f"Extracting from {len(documents)} chunks")

        all_entities = []
        all_relations = []

        for doc in documents:
            content = doc.content
            chunk_id = doc.id or f"chunk_{hash(content)}"

            # Extract entities first
            entities = self._extract_entities(content, chunk_id, prompt_template)
            all_entities.extend(entities)

            # Extract relations using known entities
            entity_names = [e.entity_name for e in entities]
            relations = self._extract_relations(
                content, chunk_id, entity_names, prompt_template
            )
            all_relations.extend(relations)

        self._logger.debug(f"Extracted {len(all_entities)} entities, {len(all_relations)} relations")

        return {
            "entities": all_entities,
            "relations": all_relations,
        }

    def _extract_entities(
        self,
        text: str,
        chunk_id: str,
        prompt_template: str | None = None,
    ) -> list[Entity]:
        """Extract entities from text using LLM.

        Args:
            text: Input text.
            chunk_id: Source chunk ID.
            prompt_template: Optional custom prompt.

        Returns:
            List of extracted Entity objects.
        """
        prompt = prompt_template or self._get_entity_extraction_prompt()

        entity_types_str = ",".join(self.entity_types)
        prompt = prompt.format(
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            entity_types=entity_types_str,
            input_text=text[:4000],  # Truncate for context window
        )

        result = self._call_llm(prompt)
        entities = self._parse_entities(result, chunk_id)

        # Gleaning: check for missed entities
        if entities and self.max_gleaning > 0:
            entities = self._gleaning_entities(text, chunk_id, entities)

        return entities

    def _extract_relations(
        self,
        text: str,
        chunk_id: str,
        known_entities: list[str],
        prompt_template: str | None = None,
    ) -> list[Relation]:
        """Extract relationships from text using LLM.

        Args:
            text: Input text.
            chunk_id: Source chunk ID.
            known_entities: List of known entity names.
            prompt_template: Optional custom prompt.

        Returns:
            List of extracted Relation objects.
        """
        if not known_entities:
            return []

        prompt = prompt_template or self._get_relation_extraction_prompt()

        entities_str = ",".join(known_entities[:50])  # Limit for context
        prompt = prompt.format(
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            entities=entities_str,
            input_text=text[:4000],
        )

        result = self._call_llm(prompt)
        relations = self._parse_relations(result, chunk_id)

        # Gleaning: check for missed relations
        if relations and self.max_gleaning > 0:
            relations = self._gleaning_relations(text, chunk_id, relations, known_entities)

        return relations

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if self.generator is None:
            raise ValueError("Generator not configured. Provide a ChatGenerator.")

        self._logger.debug(f"Calling LLM (prompt_len={len(prompt)})")

        # Wrap prompt in a ChatMessage for Haystack 2.x compatibility
        message = ChatMessage.from_user(prompt)
        response = self.generator.run(messages=[message])
        # Extract text from response based on generator type
        if hasattr(response, "replies"):
            result = response.replies[0].text if response.replies else ""
            self._logger.debug(f"LLM response (len={len(result)})")
            return result
        return str(response)

    def _parse_entities(self, text: str, chunk_id: str) -> list[Entity]:
        """Parse entities from LLM output."""
        entities = []

        # Escape delimiters for regex and split by record delimiter
        escaped_record = re.escape(self.record_delimiter)
        escaped_completion = re.escape(self.completion_delimiter)
        records = re.split(
            f"({escaped_record}|{escaped_completion})",
            text
        )

        for record in records:
            record = record.strip()
            if not record or self.record_delimiter in record:
                continue

            # Match tuple pattern
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            tuple_content = match.group(1)
            fields = self._split_tuple(tuple_content)

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

    def _parse_relations(self, text: str, chunk_id: str) -> list[Relation]:
        """Parse relations from LLM output."""
        relations = []

        # Escape delimiters for regex and split by record delimiter
        escaped_record = re.escape(self.record_delimiter)
        escaped_completion = re.escape(self.completion_delimiter)
        records = re.split(
            f"({escaped_record}|{escaped_completion})",
            text
        )

        for record in records:
            record = record.strip()
            if not record or self.record_delimiter in record:
                continue

            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            tuple_content = match.group(1)
            fields = self._split_tuple(tuple_content)

            if len(fields) < 4 or fields[0] != '"relationship"':
                continue

            src = self._clean_str(fields[1]).upper()
            tgt = self._clean_str(fields[2]).upper()
            description = self._clean_str(fields[3]) if len(fields) > 3 else ""

            # Parse weight from last field if present
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

    def _split_tuple(self, tuple_str: str) -> list[str]:
        """Split a tuple string by the tuple delimiter."""
        return [s.strip() for s in tuple_str.split(self.tuple_delimiter)]

    def _clean_str(self, s: str) -> str:
        """Clean extracted string values."""
        s = s.strip()
        # Remove quotes if present
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        return s

    def _gleaning_entities(
        self,
        text: str,
        chunk_id: str,
        initial_entities: list[Entity],
    ) -> list[Entity]:
        """Perform gleaning to catch missed entities."""
        current_entities = initial_entities

        for i in range(self.max_gleaning):
            trace(self._logger, f"Gleaning entities round {i+1}, current={len(current_entities)}")
            # Check if more entities should be extracted
            entity_names = [e.entity_name for e in current_entities]
            check_prompt = self._get_if_loop_prompt().format(
                extracted_entities=",".join(entity_names)
            )

            result = self._call_llm(check_prompt)
            if "no" in result.lower().strip('"\' '):
                trace(self._logger, "No more entities to extract")
                break

            # Extract more entities
            continue_prompt = self._get_continue_prompt()
            result = self._call_llm(continue_prompt)
            new_entities = self._parse_entities(result, chunk_id)

            if not new_entities:
                trace(self._logger, "No new entities found in gleaning")
                break

            trace(self._logger, f"Found {len(new_entities)} new entities")
            current_entities.extend(new_entities)

        return current_entities

    def _gleaning_relations(
        self,
        text: str,
        chunk_id: str,
        initial_relations: list[Relation],
        known_entities: list[str],
    ) -> list[Relation]:
        """Perform gleaning to catch missed relations."""
        current_relations = initial_relations

        for i in range(self.max_gleaning):
            trace(self._logger, f"Gleaning relations round {i+1}, current={len(current_relations)}")
            check_prompt = self._get_if_loop_prompt().format(
                extracted_relations=str(len(current_relations))
            )

            result = self._call_llm(check_prompt)
            if "no" in result.lower().strip('"\' '):
                trace(self._logger, "No more relations to extract")
                break

            continue_prompt = self._get_continue_prompt()
            result = self._call_llm(continue_prompt)
            new_relations = self._parse_relations(result, chunk_id)

            if not new_relations:
                trace(self._logger, "No new relations found in gleaning")
                break

            trace(self._logger, f"Found {len(new_relations)} new relations")
            current_relations.extend(new_relations)

        return current_relations

    # ===== Prompt Templates =====

    def _get_entity_extraction_prompt(self) -> str:
        """Get the entity extraction prompt template."""
        return """You are an expert at extracting entities from text.

Extract all entities from the following text and categorize them by type.

Entity Types: {entity_types}

Output Format:
- Start each record with ("entity", ...
- Use {tuple_delimiter} to separate fields within a record
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("entity"<|>"ENTITY_NAME"<|>"TYPE"<|>"Description of the entity"){record_delimiter}

Text to analyze:
{input_text}

Output:"""

    def _get_relation_extraction_prompt(self) -> str:
        """Get the relation extraction prompt template."""
        return """You are an expert at extracting relationships between entities.

Extract relationships between the following entities from the text.

Known Entities: {entities}

Output Format:
- Start each record with ("relationship", ...
- Use {tuple_delimiter} to separate fields
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("relationship"<|>"SOURCE_ENTITY"<|>"TARGET_ENTITY"<|>"Description of the relationship"<|>weight){record_delimiter}

Text to analyze:
{input_text}

Output:"""

    def _get_continue_prompt(self) -> str:
        """Get the prompt for continuing extraction."""
        return """Continue extracting. Output additional records following the same format:"""

    def _get_if_loop_prompt(self) -> str:
        """Get the prompt for checking if more extraction is needed."""
        return """Are there any more entities/relationships that were missed?

If you believe there are more items to extract, respond with "yes".
If you have exhausted all possibilities, respond with "no".

Your response:"""


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from a single chunk."""

    chunk_id: str
    entities: list[Entity]
    relations: list[Relation]


def merge_entities(
    existing: dict[str, Entity],
    new_entities: list[Entity],
) -> dict[str, Entity]:
    """Merge new entities with existing ones.

    Args:
        existing: Dictionary mapping entity names to Entity objects.
        new_entities: List of new Entity objects to merge.

    Returns:
        Updated dictionary of entities.
    """
    for entity in new_entities:
        name = entity.entity_name

        if name in existing:
            # Merge with existing entity
            existing_entity = existing[name]

            # Combine source IDs
            existing_sources = set(existing_entity.source_id.split("|"))
            new_sources = set(entity.source_id.split("|"))
            merged_sources = existing_sources | new_sources

            # Keep most common type
            existing_entity.source_id = "|".join(merged_sources)

            # Append description if different
            if entity.description and entity.description not in existing_entity.description:
                if existing_entity.description:
                    existing_entity.description += " | " + entity.description
                else:
                    existing_entity.description = entity.description
        else:
            existing[name] = entity

    return existing


def merge_relations(
    existing: dict[tuple[str, str], Relation],
    new_relations: list[Relation],
) -> dict[tuple[str, str], Relation]:
    """Merge new relations with existing ones.

    Args:
        existing: Dictionary mapping (src, tgt) tuples to Relation objects.
        new_relations: List of new Relation objects to merge.

    Returns:
        Updated dictionary of relations.
    """
    for relation in new_relations:
        key = relation.sorted_pair

        if key in existing:
            # Merge with existing relation
            existing_relation = existing[key]

            # Accumulate weight
            existing_relation.weight += relation.weight

            # Combine source IDs
            existing_sources = set(existing_relation.source_id.split("|"))
            new_sources = set(relation.source_id.split("|"))
            merged_sources = existing_sources | new_sources
            existing_relation.source_id = "|".join(merged_sources)

            # Append description if different
            if relation.description and relation.description not in existing_relation.description:
                if existing_relation.description:
                    existing_relation.description += " | " + relation.description
                else:
                    existing_relation.description = relation.description

            # Update order to minimum
            existing_relation.order = min(existing_relation.order, relation.order)
        else:
            existing[key] = relation

    return existing
