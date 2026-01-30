"""Demo ManasRAG functionality with mock generator.

This demonstrates the full workflow without requiring a real OpenAI API key.
"""

import os
from dotenv import load_dotenv

from haystack.dataclasses import Document

from manasrag import ManasRAG, QueryParam
from manasrag.stores import EntityVectorStore, ChunkVectorStore
from manasrag.components.entity_extractor import EntityExtractor
from manasrag.core.graph import Entity, Relation


class MockGenerator:
    """Mock generator for demonstration purposes."""

    def __init__(self):
        self.call_count = 0

    def run(self, prompt, **kwargs):
        """Generate mock responses for entity extraction."""
        self.call_count += 1

        # Mock entity extraction response
        if "entity" in prompt.lower()[:200]:
            return MockResponse([
                '(\"entity\"<|>\"ARTIFICIAL_INTELLIGENCE\"<|>\"CONCEPT\"<|>\"A branch of computer science\"<|>\"doc1\")##SPLITTER##',
                '(\"entity\"<|>\"MACHINE_LEARNING\"<|>\"CONCEPT\"<|>\"Subset of AI focusing on algorithms\"<|>\"doc1\")##SPLITTER##',
                '(\"entity\"<|>\"NEURAL_NETWORK\"<|>\"TECHNICAL_TERM\"<|>\"Computing system inspired by biological neurons\"<|>\"doc1\")##SPLITTER##',
                '<|COMPLETION|>'
            ])

        # Mock relation extraction response
        if "relationship" in prompt.lower()[:200]:
            return MockResponse([
                '(\"relationship\"<|>\"MACHINE_LEARNING\"<|>\"ARTIFICIAL_INTELLIGENCE\"<|>\"Subset of\"<|>\"1.0\")##SPLITTER##',
                '(\"relationship\"<|>\"NEURAL_NETWORK\"<|>\"MACHINE_LEARNING\"<|>\"Used in\"<|>\"1.0\")##SPLITTER##',
                '<|COMPLETION|>'
            ])

        # Mock report generation
        if "community" in prompt.lower()[:200] or "report" in prompt.lower()[:200]:
            return MockResponse([
                '{"title": "AI Concepts", "summary": "Core concepts in artificial intelligence and machine learning", "findings": [{"summary": "AI is a broad field", "explanation": "Includes ML and neural networks"}]}'
            ])

        # Default response
        return MockResponse(["This is a mock response for demonstration purposes."])


class MockResponse:
    """Mock response object."""

    def __init__(self, replies):
        self.replies = [MockReply(text) for text in replies]


class MockReply:
    """Mock reply object."""

    def __init__(self, text):
        self.text = text


def main():
    """Run ManasRAG demo with mock generator."""

    print("=" * 70)
    print("ManasRAG Demo (Mock Mode)")
    print("=" * 70)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "mock-key")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"\n📝 Configuration:")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Model: {model}")

    # Initialize mock generator
    generator = MockGenerator()
    print(f"   Generator: Mock (for demonstration)")

    # Set up stores
    print(f"\n📦 Initializing stores...")
    chunk_store = ChunkVectorStore(working_dir="./manas_demo_data")
    entity_store = EntityVectorStore(working_dir="./manas_demo_data")

    print(f"   ✓ ChunkVectorStore: {chunk_store.count} chunks")
    print(f"   ✓ EntityVectorStore: {entity_store.count} entities")

    # Initialize ManasRAG
    print(f"\n🚀 Initializing ManasRAG...")
    manas = ManasRAG(
        working_dir="./manas_demo_data",
        generator=generator,
        entity_store=entity_store,
        chunk_store=chunk_store,
        top_k=20,
        top_m=10,
    )
    print(f"   ✓ ManasRAG initialized")

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

    ## Neural Networks

    Neural networks are computing systems inspired by biological neural networks.
    They consist of interconnected nodes (neurons) organized in layers. Deep neural
    networks have multiple hidden layers and are particularly effective for pattern
    recognition and complex tasks.

    ## Applications

    AI applications include natural language processing, computer vision, robotics,
    autonomous vehicles, and game playing. Modern language models like GPT have
    demonstrated impressive capabilities in text generation and understanding.
    """

    # Index documents
    print(f"\n📚 Indexing documents...")
    print(f"   Document size: {len(documents)} characters")

    result = manas.index([Document(content=documents)])

    print(f"\n✅ Indexing completed:")
    print(f"   - Documents: {result.get('documents_count', 0)}")
    print(f"   - Chunks: {result.get('chunks_count', 0)}")
    print(f"   - Entities: {result.get('entities_count', 0)}")
    print(f"   - Relations: {result.get('relations_count', 0)}")
    print(f"   - Communities: {result.get('communities_count', 0)}")
    print(f"   - Generator calls: {generator.call_count}")

    # Check graph state
    print(f"\n📊 Graph state:")
    print(f"   - Total entities in graph: {len(list(manas.graph_store._graph.nodes()))}")
    print(f"   - Total relations in graph: {len(list(manas.graph_store._graph.edges()))}")

    # Display detected communities
    communities = manas.communities
    if communities:
        print(f"\n🏘️  Detected Communities ({len(communities)}):")
        for comm_id, community in list(communities.items())[:3]:
            node_count = len(community.nodes) if hasattr(community, 'nodes') else 0
            print(f"   - {comm_id}: {node_count} entities")

    # Display community reports
    reports = manas.reports
    if reports:
        print(f"\n📋 Community Reports ({len(reports)}):")
        for comm_id, report in list(reports.items())[:2]:
            report_str = str(report)[:100]
            print(f"   - {comm_id}: {report_str}...")

    # Test different retrieval modes
    print(f"\n" + "=" * 70)
    print("Testing Retrieval Modes")
    print("=" * 70)

    test_queries = [
        ("What is AI?", "naive"),
        ("Explain machine learning", "local"),
        ("How are neural networks related to AI?", "global"),
    ]

    for query, mode in test_queries:
        print(f"\n🔍 Mode: {mode}")
        print(f"   Query: {query}")

        try:
            result = manas.query(query, mode=mode)

            # Show context length instead of full answer (since it's mock)
            context = result.get('context', '')
            if context:
                lines = context.split('\n')
                print(f"   Context preview: {len(lines)} lines")
                print(f"   First few lines:")
                for line in lines[:3]:
                    if line.strip():
                        print(f"      {line[:80]}...")
        except Exception as e:
            print(f"   ⚠ Error: {e}")

    # Test custom query parameters
    print(f"\n🎯 Custom Query Parameters:")
    param = QueryParam(
        mode="hi",
        top_k=10,
        top_m=5,
        response_type="Single Paragraph",
    )
    print(f"   - Mode: {param.mode}")
    print(f"   - Top K: {param.top_k}")
    print(f"   - Top M: {param.top_m}")
    print(f"   - Response type: {param.response_type}")

    print(f"\n" + "=" * 70)
    print("✅ Demo Completed Successfully!")
    print("=" * 70)
    print(f"\n💡 To use with real OpenAI API:")
    print(f"   1. Set OPENAI_API_KEY in .env file")
    print(f"   2. Run: python examples/basic_usage.py")
    print(f"\n📁 Demo data saved to: ./manas_demo_data/")


if __name__ == "__main__":
    main()
