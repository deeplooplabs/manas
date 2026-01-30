# flake8: noqa
from .base import GraphDocumentStore
from .networkx_store import NetworkXGraphStore
from .vector_store import EntityVectorStore, ChunkVectorStore, KVStore, DocIdIndex

# Neo4j is optional
try:
    from .neo4j_store import Neo4jGraphStore
    _neo4j_available = True
except ImportError:
    _neo4j_available = False
    Neo4jGraphStore = None

__all__ = [
    "GraphDocumentStore",
    "NetworkXGraphStore",
    "EntityVectorStore",
    "ChunkVectorStore",
    "KVStore",
    "DocIdIndex",
]

if _neo4j_available:
    __all__.append("Neo4jGraphStore")
