"""Basic usage example for HiRAG-Haystack.

This example demonstrates:
1. Setting up HiRAG with OpenAI
2. Indexing documents
3. Querying with different retrieval modes
"""

import os

from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack.document_stores import InMemoryDocumentStore

from hirag_haystack import HiRAG, QueryParam


def main():
    """Run basic HiRAG example."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize components
    generator = OpenAIGenerator(model="gpt-4o-mini", api_key=api_key)

    # Set up stores
    chunk_store = InMemoryDocumentStore()
    entity_store = InMemoryDocumentStore()

    # Initialize HiRAG
    hirag = HiRAG(
        working_dir="./hirag_data",
        generator=generator,
        entity_store=entity_store,
        chunk_store=chunk_store,
        top_k=20,
        top_m=10,
    )

    # Sample documents
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
    result = hirag.index(documents)
    print(f"Indexed: {result}")

    # Query examples with different modes
    queries = [
        ("What is the relationship between AI and Machine Learning?", "hi"),
        ("What are the main areas of AI?", "hi_global"),
        ("How do LLMs relate to NLP?", "hi_bridge"),
        ("Explain knowledge graphs and their applications.", "hi_local"),
    ]

    print("\n" + "=" * 60)
    print("Query Examples")
    print("=" * 60)

    for query, mode in queries:
        print(f"\n--- Mode: {mode} ---")
        print(f"Query: {query}")

        result = hirag.query(query, mode=mode)
        print(f"Answer: {result['answer'][:300]}...")

    # Query with custom parameters
    print("\n" + "=" * 60)
    print("Custom Query Parameters")
    print("=" * 60)

    param = QueryParam(
        mode="hi",
        top_k=10,
        top_m=5,
        response_type="Single Paragraph",
    )

    result = hirag.query(
        "What are the key concepts in AI?",
        param=param,
    )
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
