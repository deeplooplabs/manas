"""Query pipeline for HiRAG.

This module implements the query pipeline that:
1. Retrieves relevant entities
2. Builds hierarchical context
3. Generates answers using LLM
"""

import time
from typing import Any

from haystack import Pipeline, component
from haystack.dataclasses.chat_message import ChatMessage

from hirag_haystack._logging import get_logger
from hirag_haystack.core.query_param import QueryParam
from hirag_haystack.components.hierarchical_retriever import (
    EntityRetriever,
    HierarchicalRetriever,
)
from hirag_haystack.stores.base import GraphDocumentStore


@component
class PromptBuilder:
    """Build prompts for LLM generation based on retrieved context."""

    def __init__(
        self,
        system_prompt: str | None = None,
        response_type: str = "Multiple Paragraphs",
    ):
        """Initialize the prompt builder.

        Args:
            system_prompt: Optional custom system prompt.
            response_type: Expected response format.
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.response_type = response_type

    @component.output_types(prompt=str)
    def run(
        self,
        query: str,
        context: str,
        response_type: str | None = None,
    ) -> dict:
        """Build prompt from query and retrieved context.

        Args:
            query: User query.
            context: Retrieved context.
            response_type: Override default response type.

        Returns:
            Dictionary with formatted prompt.
        """
        rt = response_type or self.response_type

        prompt = f"""{self.system_prompt}

Context:
{context}

Query: {query}

Please provide a comprehensive answer in the following format: {rt}

Answer:"""

        return {"prompt": prompt}

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a knowledgeable assistant that answers questions based on the provided context.

Use the context information which includes:
- Entity descriptions and relationships
- Community reports providing high-level summaries
- Source documents for detailed information
- Reasoning paths showing how concepts connect

Synthesize information from multiple sources to provide a comprehensive answer.
If the context doesn't contain relevant information, say so."""


@component
class QueryModeRouter:
    """Route queries based on retrieval mode."""

    def __init__(
        self,
        default_mode: str = "hi",
    ):
        """Initialize the router.

        Args:
            default_mode: Default retrieval mode.
        """
        self.default_mode = default_mode

    @component.output_types(mode=str)
    def run(
        self,
        query: str,
        mode: str | None = None,
        param: QueryParam | None = None,
    ) -> dict:
        """Determine the retrieval mode.

        Args:
            query: User query.
            mode: Explicit mode override.
            param: QueryParam with mode.

        Returns:
            Dictionary with determined mode.
        """
        if param:
            determined_mode = param.mode
        elif mode:
            determined_mode = mode
        else:
            determined_mode = self.default_mode

        return {"mode": determined_mode, "query": query}


class HiRAGQueryPipeline:
    """Pipeline for querying the HiRAG system.

    Supports multiple retrieval modes:
    - naive: Basic RAG
    - hi_local: Local entity knowledge
    - hi_global: Global community knowledge
    - hi_bridge: Cross-community reasoning
    - hi: Full hierarchical retrieval
    """

    def __init__(
        self,
        graph_store: GraphDocumentStore,
        entity_store: Any = None,
        chunk_store: Any = None,
        generator: Any = None,
        default_mode: str = "hi",
        top_k: int = 20,
        top_m: int = 10,
    ):
        """Initialize the query pipeline.

        Args:
            graph_store: Graph store for entity/relation queries.
            entity_store: Vector store for entity retrieval.
            chunk_store: Document store for text chunks.
            generator: LLM generator for answer generation.
            default_mode: Default retrieval mode.
            top_k: Number of entities to retrieve.
            top_m: Key entities per community for path finding.
        """
        self.graph_store = graph_store
        self.entity_store = entity_store
        self.chunk_store = chunk_store
        self.generator = generator
        self.default_mode = default_mode
        self.top_k = top_k
        self.top_m = top_m

        # Initialize components
        self.entity_retriever = EntityRetriever(
            entity_store=entity_store,
            top_k=top_k,
        )
        self.hierarchical_retriever = HierarchicalRetriever(
            graph_store=graph_store,
            chunk_store=chunk_store,
            top_k=top_k,
            top_m=top_m,
        )
        self.prompt_builder = PromptBuilder()

        # Logger
        self._logger = get_logger("query")

    def query(
        self,
        query: str,
        mode: str | None = None,
        param: QueryParam | None = None,
    ) -> dict:
        """Execute a query.

        Args:
            query: User query string.
            mode: Retrieval mode override.
            param: QueryParam with detailed configuration.

        Returns:
            Dictionary with answer and metadata.
        """
        # Determine mode
        if param:
            query_mode = param.mode
        elif mode:
            query_mode = mode
        else:
            query_mode = self.default_mode

        self._logger.info(f"Query mode={query_mode}, query_len={len(query)}")

        # Build query param if not provided
        if param is None:
            param = QueryParam(mode=query_mode, top_k=self.top_k, top_m=self.top_m)
        elif mode:
            param = QueryParam(**{**param.__dict__, "mode": mode})

        # Retrieve entities if using non-naive modes
        retrieved_entities = None
        if query_mode != "naive" and self.entity_store:
            result = self.entity_retriever.run(query=query)
            retrieved_entities = result.get("entities", [])
            self._logger.debug(f"Retrieved {len(retrieved_entities)} entities")

        # Get communities and reports from graph store
        communities = getattr(self.graph_store, "_communities", {})
        reports = getattr(self.graph_store, "_reports", {})

        # Build context using hierarchical retriever
        context_result = self.hierarchical_retriever.run(
            query=query,
            retrieved_entities=retrieved_entities,
            communities=communities,
            community_reports=reports,
            mode=query_mode,
            param=param,
        )
        context = context_result.get("context", "")
        self._logger.debug(f"Context length={len(context)}")

        # If only context needed, return early
        if param.only_need_context:
            return {
                "answer": context,
                "context": context,
                "mode": query_mode,
            }

        # Build prompt and generate answer
        if self.generator:
            prompt_result = self.prompt_builder.run(
                query=query,
                context=context,
                response_type=param.response_type,
            )
            prompt = prompt_result.get("prompt", "")

            answer = self._generate_answer(prompt)
        else:
            answer = context

        return {
            "answer": answer,
            "context": context,
            "mode": query_mode,
        }

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using the LLM.

        Args:
            prompt: Formatted prompt.

        Returns:
            Generated answer.
        """
        if self.generator is None:
            return "No generator configured."

        self._logger.debug(f"Generating answer (prompt_len={len(prompt)})")
        start_time = time.time()

        # Wrap prompt in a ChatMessage for Haystack 2.x compatibility
        message = ChatMessage.from_user(prompt)
        response = self.generator.run(messages=[message])

        # Extract text from response
        if hasattr(response, "replies"):
            replies = response.replies
            if replies and len(replies) > 0:
                answer = replies[0].text if hasattr(replies[0], "text") else str(replies[0])
                elapsed = time.time() - start_time
                self._logger.debug(f"LLM response (len={len(answer)}, time={elapsed:.2f}s)")
                return answer

        # Fallback for unexpected response format
        return str(response)

    @property
    def communities(self) -> dict:
        """Get communities from graph store."""
        return getattr(self.graph_store, "_communities", {})

    @property
    def reports(self) -> dict:
        """Get reports from graph store."""
        return getattr(self.graph_store, "_reports", {})


def build_query_pipeline(
    graph_store: GraphDocumentStore,
    entity_store: Any = None,
    chunk_store: Any = None,
    generator: Any = None,
) -> Pipeline:
    """Build a Haystack Pipeline for HiRAG querying.

    Args:
        graph_store: Graph store for knowledge graph.
        entity_store: Vector store for entity retrieval.
        chunk_store: Document store for chunks.
        generator: LLM generator for answers.

    Returns:
        Configured Haystack Pipeline.
    """
    pipeline = Pipeline()

    # Components would be added here for a full Haystack pipeline
    # This is simplified for the initial implementation

    return pipeline
