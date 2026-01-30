# ManasRAG

> Hierarchical Retrieval-Augmented Generation with Haystack

This project implements [HiRAG](https://github.com/hhy-huang/HiRAG) using the [Haystack](https://github.com/deepset-ai/haystack) framework. ManasRAG is a hierarchical knowledge retrieval approach that combines knowledge graphs with community-based summarization for improved RAG systems.

## Features

- **Hierarchical Knowledge Structure**: Uses Leiden clustering to build multi-level community hierarchies
- **Multiple Retrieval Modes**:
  - `naive`: Basic RAG with document chunks
  - `hi_local`: Local entity and relationship knowledge
  - `hi_global`: Global community report knowledge
  - `hi_bridge`: Cross-community reasoning paths
  - `hi`: Full hierarchical retrieval combining all modes
- **Flexible Storage**: Supports NetworkX (in-memory) and Neo4j graph databases
- **Haystack Integration**: Built on Haystack's component and pipeline architecture

## Installation

```bash
# Basic installation
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Neo4j support
pip install -e ".[neo4j]"

# All optional dependencies
pip install -e ".[all]"
```

## Configuration

### Environment Variables

The project supports loading environment variables from a `.env` file. Copy the example file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Custom API base URL
# OPENAI_BASE_URL=https://api.openai.com/v1
```

The examples will automatically load environment variables from the `.env` file.

## Quick Start

```python
from manasrag import ManasRAG
from haystack.components.generators import OpenAIGenerator
import os

# Initialize with OpenAI
manas = ManasRAG(
    working_dir="./manas_data",
    generator=OpenAIGenerator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
)

# Index documents
documents = """
# Machine Learning

Machine Learning is a subset of Artificial Intelligence focused on
algorithms that can learn from data...

# Neural Networks

Neural networks are computing systems inspired by biological neurons...
"""

manas.index(documents)

# Query with different modes
result = manas.query(
    "How are neural networks related to machine learning?",
    mode="hi"  # Full hierarchical retrieval
)

print(result["answer"])
```

## Retrieval Modes

| Mode | Description | Components |
|------|-------------|------------|
| `naive` | Basic RAG | Document chunks only |
| `hi_local` | Local knowledge | Entities + Relations + Chunks |
| `hi_global` | Global knowledge | Community reports + Chunks |
| `hi_bridge` | Bridge knowledge | Cross-community reasoning paths |
| `hi` | Full hierarchical | All components combined |

## Advanced Usage

### Custom Query Parameters

```python
from manasrag import QueryParam

param = QueryParam(
    mode="hi",
    top_k=20,           # Number of entities to retrieve
    top_m=10,           # Key entities per community
    max_token_for_text_unit=20000,
    response_type="Multiple Paragraphs",
)

result = manas.query("Your query here", param=param)
```

### Using Custom LLM

```python
from haystack.components.generators import HuggingFaceLocalGenerator

generator = HuggingFaceLocalGenerator(
    model="HuggingFaceH4/zephyr-7b-beta"
)

manas = ManasRAG(generator=generator)
```

### Accessing Communities

```python
# After indexing, access detected communities
for comm_id, community in manas.communities.items():
    print(f"Community: {community.title}")
    print(f"Entities: {len(community.nodes)}")
    print(f"Report: {community.report_string[:200]}...")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Indexing Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Documents → Splitter → EntityExtractor → GraphDocumentStore │
│                                    ↓                          │
│                          CommunityDetector                     │
│                                    ↓                          │
│                       CommunityReportGenerator                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Query Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  Query → EntityRetriever → HierarchicalRetriever            │
│                            ↓                                 │
│                      ContextBuilder                          │
│                            ↓                                 │
│                       PromptBuilder                          │
│                            ↓                                 │
│                       ChatGenerator → Answer                  │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
manasrag/
├── core/           # Core data structures
├── stores/         # Graph storage backends
├── components/     # Haystack components
├── pipelines/      # Indexing and query pipelines
└── __init__.py     # High-level API
```

## References

- [HiRAG Paper](https://arxiv.org/abs/2503.10150)
- [HiRAG GitHub](https://github.com/hhy-huang/HiRAG)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)

## License

MIT

## Acknowledgments

Based on [HiRAG](https://github.com/hhy-huang/HiRAG) by Haoyu Huang et al.
