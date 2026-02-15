# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ManasRAG implements the [HiRAG paper](https://arxiv.org/abs/2503.10150) (Hierarchical Retrieval-Augmented Generation) using the [Haystack](https://docs.haystack.deepset.ai/) framework. It builds knowledge graphs from documents via LLM entity extraction, detects communities with Louvain/Leiden clustering, generates community reports, and supports multiple retrieval modes.

## Commands

```bash
# Install dependencies
uv sync

# Run examples
uv run python examples/basic_usage.py

# CLI usage
uv run manas add-documents document.pdf          # Index documents
uv run manas query "What are the main themes?"   # Query
uv run manas serve --port 8000                   # Start REST API server
uv run manas visualize                           # Generate visualizations

# Lint
ruff check manasrag/

# Format
ruff format manasrag/

# Run tests
uv run pytest
```

Note: `pyproject.toml` declares `testpaths = ["tests"]` but no tests directory exists yet.

## Architecture

### Layer Overview

```
ManasRAG (facade)  →  Pipelines  →  Components  →  Stores
  __init__.py         pipelines/     components/     stores/
```

- **`ManasRAG`** (`__init__.py`): Facade class — the main user-facing API. Wires together stores, components, and pipelines. All public methods (`index()`, `query()`, `query_local()`, etc.) delegate to pipelines.
- **Pipelines** (`pipelines/`): `ManasRAGIndexingPipeline` and `ManasRAGQueryPipeline` orchestrate Haystack components into end-to-end workflows.
- **Components** (`components/`): Haystack `@component`-decorated classes with `run()` methods returning dicts. Each component does one job (extract entities, detect communities, retrieve, build context, etc.).
- **Stores** (`stores/`): Storage backends. `GraphDocumentStore` is the ABC; `NetworkXGraphStore` (in-memory) and `Neo4jGraphStore` (production, optional import) implement it. `EntityVectorStore`, `ChunkVectorStore`, and `KVStore` handle embeddings and metadata.
- **Core** (`core/`): Pure data structures — `Entity`, `Relation`, `NodeType` enum, `Community`, `QueryParam`, `RetrievalMode` enum. No business logic.
- **Haystack** (`haystack/`): Custom Haystack components. `MinerUToDocument` converts PDFs to Markdown using MinerU.

### Data Flow

**Indexing:** Documents → `DocumentSplitter` (token-based chunking) → `EntityExtractor` (LLM with multi-pass gleaning) → `GraphIndexer` (upsert to graph store) → `CommunityDetector` (Louvain level 0, then optional hierarchical clustering with sklearn) → `CommunityReportGenerator` (LLM summaries) → vector stores

**Query:** Query → `EntityRetriever` (semantic search on entity embeddings) → `HierarchicalRetriever` (mode-specific: local entities, global community reports, bridge cross-community paths) → `ContextBuilder` (assemble hierarchical context) → `PromptBuilder` → `ChatGenerator` → answer

**Document Management:** `index()`, `delete()`, `update()`, `list_documents()`, `has_document()` for CRUD operations. Documents use external `doc_id` for tracking.

### Retrieval Modes

| Mode | What it retrieves |
|------|-------------------|
| `naive` | Document chunks only |
| `local` | Entities + relations + chunks |
| `global` | Community reports + chunks |
| `bridge` | Cross-community reasoning paths |
| `nobridge` | Local + global combined (no paths) |
| `hi` | All: local + global + bridge |

### Storage Abstraction

`GraphDocumentStore` ABC defines the interface: node CRUD (`has_node`, `get_node`, `upsert_node`, `node_degree`, `get_node_edges`), edge CRUD, community operations (`clustering`, `community_schema`), and path operations (`shortest_path`, `subgraph_edges`). Both NetworkX and Neo4j backends implement this interface.

## Conventions

- **Haystack components**: Use `@component` decorator, declare outputs with `@component.output_types(...)`, return dicts from `run()` methods
- **Types**: Python 3.10+ union syntax (`X | None`), `@dataclass` for data classes, `TypedDict` for typed dicts
- **Naming**: `PascalCase` classes, `snake_case` functions, `_prefix` private attrs, `UPPER_SNAKE_CASE` constants grouped with `# ===== NAME =====` headers
- **Imports**: `flake8: noqa` in `__init__.py` files; grouped stdlib → third-party → local
- **Neo4j**: Optional import with try/except in `stores/__init__.py`; lazy import in `ManasRAG.__init__`
- **Ruff**: line-length 100, target Python 3.10

## Environment

Requires `OPENAI_API_KEY` (or compatible) in `.env` file. Optional `OPENAI_BASE_URL` for custom endpoints. See `.env.example`.

## CLI and API

**CLI** (`manas` command): `add-documents`, `query`, `serve`, `visualize`, `visualize-path`, `default-config`. Config file: `manas.yaml` or `~/.manas.yaml`.

**REST API** (`manas serve`): Native endpoints at `/api/*` and OpenAI-compatible endpoints at `/v1/chat/completions` for integration with tools like Open WebUI. Install with `pip install -e ".[api]"`.

## Multi-Project Isolation

The `project_id` parameter isolates data per project. Each project gets its own subdirectory under `working_dir` with independent graph stores, vector stores, and communities.

```python
manas.index(documents, project_id="project_a")
manas.query("query", project_id="project_a")
```

## Dependencies

Core: `haystack-ai>=2.6`, `networkx`, `python-louvain`, `tiktoken`, `python-dotenv`, `mineru` (PDF parsing), `torch`

Optional groups: `openai`, `neo4j`, `scikit-learn` (hierarchical clustering), `visualization` (pyvis, plotly), `cli` (document converters), `api` (FastAPI), `webui` (Streamlit), `dev` (pytest), `all`
