# components/AGENTS.md

Haystack components for HiRAG pipeline stages.

## Components Overview

| Component | Purpose | Key Method |
|-----------|---------|------------|
| `EntityExtractor` | LLM-based entity/relation extraction | `run(documents)` |
| `CommunityDetector` | Louvain/Leiden community detection | `run(graph)` |
| `CommunityReportGenerator` | Generate community summary reports | `run(community)` |
| `HierarchicalRetriever` | Multi-mode knowledge retrieval | `run(query, ...)` |
| `PathFinder` | Cross-community path finding | `run(src, tgt)` |

## Component Pattern

```python
@component
class ComponentName:
    """One-line description."""

    @component.output_types(output_type=OutputType)
    def run(self, input: InputType) -> dict:
        """Process input, return dict matching output_types."""
        return {"output": result}
```

## Key Files

- `entity_extractor.py` - Entity/relation extraction with gleaning
- `community_detector.py` - Louvain clustering (python-louvain)
- `hierarchical_retriever.py` - Retrieval modes: naive, hi_local, hi_global, hi_bridge
- `path_finder.py` - NetworkX shortest_path for cross-community reasoning

## Dependencies

- `haystack.components.*` - Haystack component base classes
- `community` (python-louvain) - Community detection
- `networkx` - Graph algorithms
