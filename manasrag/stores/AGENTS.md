# stores/AGENTS.md

Graph storage backends for ManasRAG knowledge graphs.

## Stores Overview

| Store | Backend | Notes |
|-------|---------|-------|
| `NetworkXGraphStore` | In-memory NetworkX | Default, uses Louvain |
| `Neo4jGraphStore` | Neo4j database | Production, uses GDS |

## Base Class

`GraphDocumentStore` (ABC) defines the interface:
- **Node ops**: `has_node`, `get_node`, `upsert_node`, `node_degree`
- **Edge ops**: `has_edge`, `get_edge`, `upsert_edge`
- **Community ops**: `clustering(algorithm)`, `community_schema()`
- **Path ops**: `shortest_path(src, tgt)`, `subgraph_edges(nodes)`

## Persistence

NetworkX store saves to JSON:
- `{namespace}_graph.json` - NetworkX node-link format
- `{namespace}_communities.json` - Community data

Loaded automatically on instantiation if files exist.

## Neo4j Import Pattern

```python
try:
    from manasrag.stores.neo4j_store import Neo4jGraphStore
except ImportError:
    # neo4j not installed
    pass
```

Neo4j is optional; fallback to NetworkX if import fails.
