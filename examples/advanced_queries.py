"""Advanced query examples for HiRAG-Haystack.

This example demonstrates:
1. Using QueryParam for fine-grained control
2. Context-only retrieval
3. Different retrieval modes and their use cases
"""

import os

from haystack.components.generators import OpenAIGenerator
from haystack.document_stores import InMemoryDocumentStore

from hirag_haystack import HiRAG, QueryParam


def main():
    """Run advanced HiRAG query examples."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    generator = OpenAIGenerator(model="gpt-4o-mini", api_key=api_key)

    hirag = HiRAG(
        working_dir="./hirag_data",
        generator=generator,
        top_k=20,
        top_m=10,
    )

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
