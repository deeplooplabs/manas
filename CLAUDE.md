# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiRAG-Haystack implements the [HiRAG paper](https://arxiv.org/abs/2503.10150) (Hierarchical Retrieval-Augmented Generation) using the [Haystack](https://docs.haystack.deepset.ai/) framework. It builds knowledge graphs from documents via LLM entity extraction, detects communities with Louvain/Leiden clustering, generates community reports, and supports multiple retrieval modes.

## Commands

```bash
# Install dependencies
uv sync

# Run examples
uv run python examples/basic_usage.py

# Lint
ruff check hirag_haystack/

# Format
ruff format hirag_haystack/

# Run tests (test files are in the project root, not tests/)
uv run pytest test_*.py
```

Note: `pyproject.toml` declares `testpaths = ["tests"]` but no `tests/` directory exists. Test files live in the project root as `test_*.py`.

## Architecture

### Layer Overview

```
HiRAG (facade)  →  Pipelines  →  Components  →  Stores
  __init__.py       pipelines/     components/     stores/
```

- **`HiRAG`** (`__init__.py`): Facade class — the main user-facing API. Wires together stores, components, and pipelines. All public methods (`index()`, `query()`, `query_local()`, etc.) delegate to pipelines.
- **Pipelines** (`pipelines/`): `HiRAGIndexingPipeline` and `HiRAGQueryPipeline` orchestrate Haystack components into end-to-end workflows.
- **Components** (`components/`): Haystack `@component`-decorated classes with `run()` methods returning dicts. Each component does one job (extract entities, detect communities, retrieve, build context, etc.).
- **Stores** (`stores/`): Storage backends. `GraphDocumentStore` is the ABC; `NetworkXGraphStore` (in-memory) and `Neo4jGraphStore` (production, optional import) implement it. `EntityVectorStore`, `ChunkVectorStore`, and `KVStore` handle embeddings and metadata.
- **Core** (`core/`): Pure data structures — `Entity`, `Relation`, `NodeType` enum, `Community`, `QueryParam`, `RetrievalMode` enum. No business logic.

### Data Flow

**Indexing:** Documents → `DocumentSplitter` (token-based chunking) → `EntityExtractor` (LLM with multi-pass gleaning) → `GraphIndexer` (upsert to graph store) → `CommunityDetector` (Louvain level 0, then optional hierarchical clustering with sklearn) → `CommunityReportGenerator` (LLM summaries) → vector stores

**Query:** Query → `EntityRetriever` (semantic search on entity embeddings) → `HierarchicalRetriever` (mode-specific: local entities, global community reports, bridge cross-community paths) → `ContextBuilder` (assemble hierarchical context) → `PromptBuilder` → `ChatGenerator` → answer

### Retrieval Modes

| Mode | What it retrieves |
|------|-------------------|
| `naive` | Document chunks only |
| `hi_local` | Entities + relations + chunks |
| `hi_global` | Community reports + chunks |
| `hi_bridge` | Cross-community reasoning paths |
| `hi_nobridge` | Local + global combined (no paths) |
| `hi` | All: local + global + bridge |

### Storage Abstraction

`GraphDocumentStore` ABC defines the interface: node CRUD (`has_node`, `get_node`, `upsert_node`, `node_degree`, `get_node_edges`), edge CRUD, community operations (`clustering`, `community_schema`), and path operations (`shortest_path`, `subgraph_edges`). Both NetworkX and Neo4j backends implement this interface.

## Conventions

- **Haystack components**: Use `@component` decorator, declare outputs with `@component.output_types(...)`, return dicts from `run()` methods
- **Types**: Python 3.10+ union syntax (`X | None`), `@dataclass` for data classes, `TypedDict` for typed dicts
- **Naming**: `PascalCase` classes, `snake_case` functions, `_prefix` private attrs, `UPPER_SNAKE_CASE` constants grouped with `# ===== NAME =====` headers
- **Imports**: `flake8: noqa` in `__init__.py` files; grouped stdlib → third-party → local
- **Neo4j**: Optional import with try/except in `stores/__init__.py`; lazy import in `HiRAG.__init__`
- **Ruff**: line-length 100, target Python 3.10

## Environment

Requires `OPENAI_API_KEY` (or compatible) in `.env` file. Optional `OPENAI_BASE_URL` for custom endpoints. See `.env.example`.

## Dependencies

Core: `haystack-ai>=2.6`, `networkx`, `python-louvain`, `tiktoken`, `python-dotenv`

Optional groups: `openai`, `neo4j`, `scikit-learn` (hierarchical clustering), `visualization` (pyvis, plotly), `dev` (pytest), `all`
