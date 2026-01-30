# AGENTS.md

**Generated:** 2026-01-30
**Project:** HiRAG-Haystack - Hierarchical RAG with Haystack

## Overview

Python library implementing [HiRAG](https://github.com/hhy-huang/HiRAG) using Haystack framework. Builds hierarchical knowledge graphs from documents using LLM entity extraction and Leiden/Louvain community detection.

## Structure

```
hirag_haystack/
├── __init__.py           # HiRAG facade class (26 public APIs)
├── prompts.py            # LLM prompt templates
├── core/                 # Data structures
├── stores/               # Graph storage backends (NetworkX, Neo4j)
├── components/           # Haystack components
├── pipelines/            # Indexing/query pipelines
└── utils/                # Token, color utilities
```

## Where to Look

| Task | Location | Key Types |
|------|----------|-----------|
| Entity extraction | `components/entity_extractor.py` | `EntityExtractor` |
| Community detection | `components/community_detector.py` | `CommunityDetector` |
| Graph storage | `stores/networkx_store.py` | `NetworkXGraphStore` |
| Query pipeline | `pipelines/query.py` | `HiRAGQueryPipeline` |

## Conventions

- **Imports**: `flake8: noqa`, grouped stdlib → third-party → local
- **Types**: Python 3.10+ union syntax (`X | None`), `@dataclass`, `TypedDict`
- **Naming**: `PascalCase` classes, `snake_case` functions, `_prefix` private attrs
- **Haystack**: `@component` decorator, `@component.output_types(...)`, return dicts from `run()`
- **Constants**: Module-level `UPPER_SNAKE_CASE`, grouped with `# ===== NAME =====`

## Anti-Patterns (THIS PROJECT)

- NO tests directory - pyproject.toml has pytest config but no `tests/`
- `nul` file in root - Windows artifact, should be deleted
- `.env` may contain secrets - ensure not committed

## Commands

```bash
# Install
uv sync

# Run examples
uv run python examples/basic_usage.py

# Lint
ruff check hirag_haystack/
```

## Notes

- No CLI entry points - use Python API: `from hirag_haystack import HiRAG`
- Neo4j store has optional import with try/except fallback
- Utils module (`token_utils`, `color_utils`) not exported in main `__init__.py`
