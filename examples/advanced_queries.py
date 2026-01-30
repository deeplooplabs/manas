"""Advanced query examples for HiRAG-Haystack.

This example demonstrates:
1. Using QueryParam for fine-grained control
2. Context-only retrieval
3. Different retrieval modes and their use cases
"""

import os

from dotenv import load_dotenv

from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.dataclasses import Document

from hirag_haystack import HiRAG, QueryParam
from hirag_haystack.stores import EntityVectorStore, ChunkVectorStore


def main():
    """Run advanced HiRAG query examples."""

    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    base_url = os.getenv("OPENAI_BASE_URL")  # Optional
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Allow custom model

    # Initialize generator with custom base URL if provided
    if base_url:
        generator = OpenAIGenerator(model=model, api_base_url=base_url, timeout=120.0)
    else:
        generator = OpenAIGenerator(model=model, timeout=120.0)

    # Set up stores for entity and chunk retrieval
    chunk_store = ChunkVectorStore(working_dir="./hirag_data")
    entity_store = EntityVectorStore(working_dir="./hirag_data")

    hirag = HiRAG(
        working_dir="./hirag_data",
        generator=generator,
        entity_store=entity_store,
        chunk_store=chunk_store,
        top_k=20,
        top_m=10,
    )

    # Sample documents for indexing
    documents = """
    # Artificial Intelligence

    Artificial Intelligence (AI) is a branch of computer science focused on creating
    systems capable of performing tasks that typically require human intelligence.
    These tasks include learning, reasoning, problem-solving, perception, and
    language understanding.

    ## Machine Learning

    Machine Learning (ML) is a subset of AI that focuses on algorithms that can
    learn from data. Key approaches include supervised learning, unsupervised
    learning, and reinforcement learning. Deep Learning, a subset of ML, uses
    neural networks with multiple layers.

    ## Natural Language Processing

    Natural Language Processing (NLP) is another important area of AI. It deals
    with the interaction between computers and human language. Applications include
    machine translation, sentiment analysis, and question answering systems.

    Large Language Models (LLMs) like GPT have revolutionized NLP by demonstrating
    impressive capabilities in text generation, understanding, and reasoning.

    ## Knowledge Graphs

    Knowledge graphs represent information as a network of entities and their
    relationships. They are used in various applications including search engines,
    recommendation systems, and AI reasoning. GraphRAG and HiRAG are approaches
    that combine knowledge graphs with retrieval-augmented generation.
    """

    print("Indexing documents...")
    result = hirag.index([Document(content=documents)])
    print(f"Indexed: {result}\n")

    # Example 1: Get context without generation
    print("=" * 60)
    print("Example 1: Context-only retrieval")
    print("=" * 60)

    param = QueryParam(
        mode="hi",
        only_need_context=True,
        top_k=10,
    )

    result = hirag.query(
        "What concepts are related to neural networks?",
        param=param,
    )
    print(f"Context length: {len(result['answer'])} characters")
    print(f"Context preview:\n{result['answer'][:500]}...")

    # Example 2: Naive RAG comparison
    print("\n" + "=" * 60)
    print("Example 2: Naive RAG vs Hierarchical")
    print("=" * 60)

    query = "How does deep learning relate to AI?"

    naive_result = hirag.query(query, mode="naive")
    hi_result = hirag.query(query, mode="hi")

    print(f"\nNaive RAG Answer:\n{naive_result['answer'][:400]}...")
    print(f"\nHierarchical Answer:\n{hi_result['answer'][:400]}...")

    # Example 3: Bridge reasoning
    print("\n" + "=" * 60)
    print("Example 3: Bridge reasoning between distant concepts")
    print("=" * 60)

    result = hirag.query(
        "How are machine learning and knowledge graphs connected?",
        mode="hi_bridge",
    )
    print(f"Bridge Answer:\n{result['answer'][:500]}...")

    # Example 4: Global overview
    print("\n" + "=" * 60)
    print("Example 4: Global community overview")
    print("=" * 60)

    result = hirag.query(
        "What are the main themes across all documents?",
        mode="hi_global",
    )
    print(f"Global Answer:\n{result['answer'][:500]}...")

    # Example 5: Local entity-focused
    print("\n" + "=" * 60)
    print("Example 5: Local entity and relationship focus")
    print("=" * 60)

    param = QueryParam.hi_local(
        top_k=15,
        max_token_for_local_context=10000,
    )

    result = hirag.query(
        "Tell me about neural networks and their components.",
        param=param,
    )
    print(f"Local Answer:\n{result['answer'][:500]}...")

    # Example 6: Token limits
    print("\n" + "=" * 60)
    print("Example 6: Controlling token limits")
    print("=" * 60)

    param = QueryParam(
        mode="hi",
        max_token_for_text_unit=5000,
        max_token_for_community_report=5000,
        max_token_for_bridge_knowledge=3000,
    )

    result = hirag.query(
        "Summarize the key information about AI applications.",
        param=param,
    )
    print(f"Controlled Answer:\n{result['answer'][:500]}...")


if __name__ == "__main__":
    main()
