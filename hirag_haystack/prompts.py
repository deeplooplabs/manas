"""Prompt templates for HiRAG.

This module contains all prompt templates used for LLM interactions,
migrated from the original HiRAG implementation.
"""

from typing import Dict, List


# ===== Delimiters =====
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##SPLITTER##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETION|>"
GRAPH_FIELD_SEP = "|"

# ===== Entity Types =====
DEFAULT_ENTITY_TYPES = [
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
    "TECHNICAL_TERM",
]

META_ENTITY_TYPES = [
    "ORGANIZATION",
    "PERSON",
    "GEO_LOCATION",
    "DATE_TIME",
    "EVENT",
    "CONCEPT",
    "TECHNICAL_TERM",
    "NUMBER",
]

# ===== Entity Extraction Prompts =====

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting entities from text.

Extract all entities from the following text and categorize them by type.

Entity Types: {entity_types}

Output Format:
- Start each record with ("entity", ...
- Use {tuple_delimiter} to separate fields within a record
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("entity"{tuple_delimiter}"ENTITY_NAME"{tuple_delimiter}"TYPE"{tuple_delimiter}"Description of the entity"){record_delimiter}

Text to analyze:
{input_text}

Output:"""

HI_ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting entities from text for building a knowledge graph.

Extract all entities from the following text and categorize them by type.

Entity Types: {entity_types}

Output Format:
- Start each record with ("entity", ...
- Use {tuple_delimiter} to separate fields within a record
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("entity"{tuple_delimiter}"ENTITY_NAME"{tuple_delimiter}"TYPE"{tuple_delimiter}"Description of the entity"){record_delimiter}

Text to analyze:
{input_text}

Output:"""

# ===== Relation Extraction Prompts =====

RELATION_EXTRACTION_PROMPT = """You are an expert at extracting relationships between entities.

Extract relationships between the following entities from the text.

Known Entities: {entities}

Output Format:
- Start each record with ("relationship", ...
- Use {tuple_delimiter} to separate fields
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("relationship"{tuple_delimiter}"SOURCE_ENTITY"{tuple_delimiter}"TARGET_ENTITY"{tuple_delimiter}"Description of the relationship"{tuple_delimiter}weight){record_delimiter}

Text to analyze:
{input_text}

Output:"""

HI_RELATION_EXTRACTION_PROMPT = """You are an expert at extracting relationships between entities in a knowledge graph.

Extract relationships between the following entities from the text.
Focus on direct, meaningful relationships.

Known Entities: {entities}

Output Format:
- Start each record with ("relationship", ...
- Use {tuple_delimiter} to separate fields
- Use {record_delimiter} to separate records
- End with {completion_delimiter}

Example:
("relationship"{tuple_delimiter}"SOURCE_ENTITY"{tuple_delimiter}"TARGET_ENTITY"{tuple_delimiter}"Description of the relationship"{tuple_delimiter}1.0){record_delimiter}

Text to analyze:
{input_text}

Output:"""

# ===== Community Report Prompts =====

COMMUNITY_REPORT_PROMPT = """You are analyzing a community of related entities in a knowledge graph.

Based on the following information, generate a comprehensive report that:
1. Identifies the main theme/topic of this community
2. Summarizes the key entities and their roles
3. Describes important relationships

{input_text}

Generate a JSON response with this structure:
{{
    "title": "Brief descriptive title",
    "summary": "2-3 sentence overview of the community",
    "findings": [
        {{"summary": "Key point 1", "explanation": "Detailed explanation"}},
        {{"summary": "Key point 2", "explanation": "Detailed explanation"}}
    ]
}}

Response:"""

# ===== Entity Summarization Prompts =====

SUMMARIZE_ENTITY_DESCRIPTIONS = """The following is a description of the entity "{entity_name}":

{description_list}

The description is too long. Please summarize it in fewer than {max_tokens} tokens
while preserving the key information about this entity.

Summary:"""

# ===== Gleaning / Iteration Prompts =====

ENTITY_CONTINUE_EXTRACTION = """Continue extracting. There may be more entities or relationships
that were missed in the previous extraction.

Output additional records following the same format:"""

RELATION_CONTINUE_EXTRACTION = """Continue extracting. There may be more relationships
that were missed in the previous extraction.

Output additional records following the same format:"""

ENTITY_IF_LOOP_EXTRACTION = """Based on the entities and relationships extracted so far,
do you believe there are more items that should be extracted from the text?

Respond with "yes" if you believe more extraction is needed.
Respond with "no" if you have exhausted all possibilities.

Your response:"""

# ===== Response Generation Prompts =====

NAIVE_RAG_RESPONSE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{content_data}

Query: {query}

Please provide a comprehensive answer. If the context doesn't contain relevant information, say so.

Response format: {response_type}

Answer:"""

LOCAL_RAG_RESPONSE = """You are a knowledgeable assistant that answers questions based on retrieved context.

The context includes:
- Entity descriptions and relationships
- Community reports providing high-level summaries
- Source documents for detailed information
- Reasoning paths showing how concepts connect

Context:
{context_data}

Query: {query}

Synthesize information from multiple sources to provide a comprehensive answer.
If the context doesn't contain relevant information, say so.

Response format: {response_type}

Answer:"""

# ===== Utility Prompts =====

FAIL_RESPONSE = """I apologize, but I couldn't find relevant information to answer your question
in the available knowledge base. Please try rephrasing your question or providing more context."""

# ===== Process Tickers =====
PROCESS_TICKERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ===== Default Text Separators =====
DEFAULT_TEXT_SEPARATOR = [
    "\n\n",
    "\n",
    ". ",
    ", ",
    "; ",
    ": ",
    " ",
]

# ===== Prompt Management =====

class PromptManager:
    """Manager for accessing and customizing prompts."""

    def __init__(self):
        self._custom_prompts: Dict[str, str] = {}

    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt by name."""
        return self._custom_prompts.get(
            prompt_name,
            globals().get(prompt_name, "")
        )

    def set_prompt(self, prompt_name: str, template: str) -> None:
        """Set a custom prompt template."""
        self._custom_prompts[prompt_name] = template

    def format_prompt(
        self,
        prompt_name: str,
        **kwargs
    ) -> str:
        """Format a prompt with provided variables."""
        template = self.get_prompt(prompt_name)
        return template.format(**kwargs)

    def get_entity_extraction_prompt(
        self,
        entity_types: List[str] | None = None,
        tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
        record_delimiter: str = DEFAULT_RECORD_DELIMITER,
        completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
        hierarchical: bool = False,
    ) -> str:
        """Get the entity extraction prompt with configured options."""
        base_prompt = HI_ENTITY_EXTRACTION_PROMPT if hierarchical else ENTITY_EXTRACTION_PROMPT
        types = entity_types or DEFAULT_ENTITY_TYPES

        return base_prompt.format(
            entity_types=",".join(types),
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
        )

    def get_relation_extraction_prompt(
        self,
        tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
        record_delimiter: str = DEFAULT_RECORD_DELIMITER,
        completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
        hierarchical: bool = False,
    ) -> str:
        """Get the relation extraction prompt with configured options."""
        base_prompt = HI_RELATION_EXTRACTION_PROMPT if hierarchical else RELATION_EXTRACTION_PROMPT

        return base_prompt.format(
            entities="{entities}",
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
            input_text="{input_text}",
        )


# Global prompt manager instance
prompt_manager = PromptManager()


def get_prompts() -> Dict[str, str]:
    """Get all available prompts as a dictionary."""
    return {
        "ENTITY_EXTRACTION_PROMPT": ENTITY_EXTRACTION_PROMPT,
        "HI_ENTITY_EXTRACTION_PROMPT": HI_ENTITY_EXTRACTION_PROMPT,
        "RELATION_EXTRACTION_PROMPT": RELATION_EXTRACTION_PROMPT,
        "HI_RELATION_EXTRACTION_PROMPT": HI_RELATION_EXTRACTION_PROMPT,
        "COMMUNITY_REPORT_PROMPT": COMMUNITY_REPORT_PROMPT,
        "SUMMARIZE_ENTITY_DESCRIPTIONS": SUMMARIZE_ENTITY_DESCRIPTIONS,
        "ENTITY_CONTINUE_EXTRACTION": ENTITY_CONTINUE_EXTRACTION,
        "ENTITY_IF_LOOP_EXTRACTION": ENTITY_IF_LOOP_EXTRACTION,
        "NAIVE_RAG_RESPONSE": NAIVE_RAG_RESPONSE,
        "LOCAL_RAG_RESPONSE": LOCAL_RAG_RESPONSE,
        "FAIL_RESPONSE": FAIL_RESPONSE,
    }


PROMPTS = get_prompts()
