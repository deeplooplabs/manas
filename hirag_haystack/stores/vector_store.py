"""Vector store implementations for HiRAG.

This module provides specialized vector stores for entities and chunks,
built on top of Haystack's document store abstraction.
"""

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class EntityVectorStore:
    """Vector store for entity embeddings.

    Provides efficient similarity search for entities based on
    their descriptions and metadata.
    """

    def __init__(
        self,
        embedding_func: Callable[[List[str]], List[List[float]]] | None = None,
        meta_fields: set | None = None,
        working_dir: str = "./hirag_cache",
    ):
        """Initialize the entity vector store.

        Args:
            embedding_func: Function to generate embeddings for text.
                Should accept a list of strings and return a list of embeddings.
            meta_fields: Metadata fields to store with each entity.
            working_dir: Directory for persistent storage.
        """
        self.embedding_func = embedding_func
        self.meta_fields = meta_fields or {"entity_name"}
        self.working_dir = working_dir

        self._store = InMemoryDocumentStore()
        self._entity_index: dict[str, str] = {}  # entity_name -> doc_id

        # Load from disk if available
        self._load_from_disk()

    def upsert(
        self,
        entities: dict[str, dict],
    ) -> None:
        """Upsert entities into the vector store.

        Args:
            entities: Dictionary mapping entity hash to entity data.
                Each entity data should have:
                - content: Text content for embedding
                - entity_name: Name of the entity
                - Other metadata fields
        """
        documents = []

        for entity_hash, entity_data in entities.items():
            entity_name = entity_data.get("entity_name", "")

            # Create document
            doc = Document(
                id=entity_hash,
                content=entity_data.get("content", entity_name),
                meta={
                    "entity_name": entity_name,
                    **{k: v for k, v in entity_data.items() if k != "content"}
                },
            )
            documents.append(doc)

            # Update index
            self._entity_index[entity_name] = entity_hash

        if documents:
            self._store.write_documents(documents)

    def query(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> List[dict]:
        """Query for similar entities.

        Args:
            query: Query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching entities with metadata.
        """
        if not query:
            return []

        # Use BM25 retrieval for text-based search
        results = self._store.bm25_retrieval(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        return [
            {
                "id": r.id,
                "entity_name": r.meta.get("entity_name", r.id),
                "content": r.content,
                "score": getattr(r, "score", 0.0),
                "meta": r.meta,
            }
            for r in results
        ]

    def query_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> List[dict]:
        """Query by pre-computed embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching entities.
        """
        results = self._store.query_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        return [
            {
                "id": r.id,
                "entity_name": r.meta.get("entity_name", r.id),
                "content": r.content,
                "score": getattr(r, "score", 0.0),
                "meta": r.meta,
            }
            for r in results
        ]

    def get_entity_ids(self, entity_names: List[str]) -> dict[str, str]:
        """Get document IDs for entity names.

        Args:
            entity_names: List of entity names.

        Returns:
            Dictionary mapping entity names to document IDs.
        """
        return {
            name: self._entity_index.get(name, "")
            for name in entity_names
            if name in self._entity_index
        }

    def delete_entities(self, entity_names: List[str]) -> None:
        """Delete entities from the store.

        Args:
            entity_names: List of entity names to delete.
        """
        doc_ids = [
            self._entity_index[name]
            for name in entity_names
            if name in self._entity_index
        ]

        if doc_ids:
            self._store.delete_documents(doc_ids)

            # Remove from index
            for name in entity_names:
                self._entity_index.pop(name, None)

    def _load_from_disk(self) -> None:
        """Load index from disk."""
        index_path = Path(self.working_dir) / "entity_index.json"

        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._entity_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._entity_index = {}

    def save_to_disk(self) -> None:
        """Save index to disk."""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        index_path = Path(self.working_dir) / "entity_index.json"

        with open(index_path, "w") as f:
            json.dump(self._entity_index, f)

    @property
    def count(self) -> int:
        """Get the number of entities in the store."""
        return len(self._entity_index)


class ChunkVectorStore:
    """Vector store for text chunks.

    Provides efficient similarity search for document chunks
    in naive RAG mode.
    """

    def __init__(
        self,
        embedding_func: Callable[[List[str]], List[List[float]]] | None = None,
        working_dir: str = "./hirag_cache",
    ):
        """Initialize the chunk vector store.

        Args:
            embedding_func: Function to generate embeddings.
            working_dir: Directory for persistent storage.
        """
        self.embedding_func = embedding_func
        self.working_dir = working_dir

        self._store = InMemoryDocumentStore()
        self._chunk_index: dict[str, dict] = {}  # chunk_id -> chunk_data

        # Load from disk if available
        self._load_from_disk()

    def upsert(
        self,
        chunks: dict[str, dict],
    ) -> None:
        """Upsert chunks into the vector store.

        Args:
            chunks: Dictionary mapping chunk hash to chunk data.
                Each chunk should have:
                - content: Text content
                - full_doc_id: Source document ID
                - chunk_order_index: Order within document
        """
        documents = []

        for chunk_hash, chunk_data in chunks.items():
            doc = Document(
                id=chunk_hash,
                content=chunk_data.get("content", ""),
                meta={
                    "full_doc_id": chunk_data.get("full_doc_id", ""),
                    "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                    "tokens": chunk_data.get("tokens", 0),
                },
            )
            documents.append(doc)

            # Update index
            self._chunk_index[chunk_hash] = chunk_data

        if documents:
            self._store.write_documents(documents)

    def query(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> List[dict]:
        """Query for similar chunks.

        Args:
            query: Query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching chunks.
        """
        if not query:
            return []

        results = self._store.bm25_retrieval(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        return [
            {
                "id": r.id,
                "content": r.content,
                "score": getattr(r, "score", 0.0),
                "meta": r.meta,
            }
            for r in results
        ]

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Get a chunk by its ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            Chunk data or None if not found.
        """
        if chunk_id in self._chunk_index:
            return self._chunk_index[chunk_id]

        # Try to get from document store using filter_documents
        docs = self._store.filter_documents(filters={"field": "id", "operator": "==", "value": chunk_id})
        if docs:
            doc = docs[0]
            return {
                "id": doc.id,
                "content": doc.content,
                "meta": doc.meta,
            }

        return None

    @staticmethod
    def _get_document_by_chunk_id(store: InMemoryDocumentStore, chunk_id: str) -> Optional[Document]:
        """Get a document by its chunk ID from the store.

        Args:
            store: InMemoryDocumentStore instance.
            chunk_id: The chunk ID.

        Returns:
            Document or None if not found.
        """
        docs = store.filter_documents(filters={"field": "id", "operator": "==", "value": chunk_id})
        return docs[0] if docs else None

    def get_by_ids(self, chunk_ids: List[str]) -> List[Optional[dict]]:
        """Get multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs.

        Returns:
            List of chunk data (None for missing chunks).
        """
        return [self.get_by_id(cid) for cid in chunk_ids]

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from the store.

        Args:
            chunk_ids: List of chunk IDs to delete.
        """
        self._store.delete_documents(chunk_ids)

        for chunk_id in chunk_ids:
            self._chunk_index.pop(chunk_id, None)

    def _load_from_disk(self) -> None:
        """Load index from disk."""
        index_path = Path(self.working_dir) / "chunk_index.json"

        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._chunk_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._chunk_index = {}

    def save_to_disk(self) -> None:
        """Save index to disk."""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        index_path = Path(self.working_dir) / "chunk_index.json"

        with open(index_path, "w") as f:
            json.dump(self._chunk_index, f)

    @property
    def count(self) -> int:
        """Get the number of chunks in the store."""
        return len(self._chunk_index)

    def get_document_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        """Get a chunk by its chunk ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            Chunk data or None if not found.
        """
        return self.get_by_id(chunk_id)

    def get_chunks_by_doc_id(self, doc_id: str) -> List[dict]:
        """Get all chunks belonging to a document by its full_doc_id metadata.

        Args:
            doc_id: The source document ID (stored as full_doc_id in chunk metadata).

        Returns:
            List of chunk data dicts.
        """
        results = []
        for chunk_id, chunk_data in self._chunk_index.items():
            if chunk_data.get("full_doc_id") == doc_id:
                results.append({"id": chunk_id, **chunk_data})
        return results


class DocIdIndex:
    """Reverse index mapping external doc_ids to their chunks and entities.

    Tracks which chunks and entities belong to each document,
    enabling efficient deletion and update operations.
    """

    def __init__(self, working_dir: str = "./hirag_cache"):
        self.working_dir = working_dir
        # doc_id -> list of chunk_ids
        self._doc_chunks: dict[str, list[str]] = {}
        # doc_id -> set of entity_names (stored as list for JSON serialization)
        self._doc_entities: dict[str, list[str]] = {}

        self._load_from_disk()

    def add_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        """Register chunk_ids for a doc_id."""
        existing = self._doc_chunks.get(doc_id, [])
        merged = list(dict.fromkeys(existing + chunk_ids))  # deduplicate, preserve order
        self._doc_chunks[doc_id] = merged

    def add_entities(self, doc_id: str, entity_names: list[str]) -> None:
        """Register entity_names for a doc_id."""
        existing = set(self._doc_entities.get(doc_id, []))
        existing.update(entity_names)
        self._doc_entities[doc_id] = sorted(existing)

    def get_chunks(self, doc_id: str) -> list[str]:
        """Return chunk_ids belonging to doc_id."""
        return list(self._doc_chunks.get(doc_id, []))

    def get_entities(self, doc_id: str) -> list[str]:
        """Return entity_names belonging to doc_id."""
        return list(self._doc_entities.get(doc_id, []))

    def list_doc_ids(self) -> list[str]:
        """Return all registered doc_ids."""
        return sorted(self._doc_chunks.keys())

    def has_doc(self, doc_id: str) -> bool:
        """Check if a doc_id is registered."""
        return doc_id in self._doc_chunks

    def remove_doc(self, doc_id: str) -> None:
        """Remove all records for a doc_id."""
        self._doc_chunks.pop(doc_id, None)
        self._doc_entities.pop(doc_id, None)

    def all_entity_doc_ids(self, entity_name: str) -> list[str]:
        """Return all doc_ids that reference a given entity_name."""
        return [
            did for did, enames in self._doc_entities.items()
            if entity_name in enames
        ]

    def _load_from_disk(self) -> None:
        index_path = Path(self.working_dir) / "doc_id_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                self._doc_chunks = data.get("doc_chunks", {})
                self._doc_entities = data.get("doc_entities", {})
            except (json.JSONDecodeError, IOError):
                self._doc_chunks = {}
                self._doc_entities = {}

    def save_to_disk(self) -> None:
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        index_path = Path(self.working_dir) / "doc_id_index.json"
        with open(index_path, "w") as f:
            json.dump(
                {"doc_chunks": self._doc_chunks, "doc_entities": self._doc_entities},
                f,
            )


class KVStore:
    """Key-value store for structured data.

    Used for storing community reports, LLM cache, and other
    structured data that doesn't require vector search.
    """

    def __init__(
        self,
        namespace: str = "default",
        working_dir: str = "./hirag_cache",
    ):
        """Initialize the KV store.

        Args:
            namespace: Namespace for this store.
            working_dir: Directory for persistent storage.
        """
        self.namespace = namespace
        self.working_dir = working_dir
        self._data: dict = {}

        self._load_from_disk()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        return self._data.get(key, default)

    def get_all(self) -> dict:
        """Get all stored data.

        Returns:
            Dictionary of all key-value pairs.
        """
        return self._data.copy()

    def set(self, key: str, value: Any) -> None:
        """Set a value.

        Args:
            key: The key to set.
            value: The value to store.
        """
        self._data[key] = value

    def set_batch(self, data: dict) -> None:
        """Set multiple values.

        Args:
            data: Dictionary of key-value pairs to set.
        """
        self._data.update(data)

    def delete(self, key: str) -> None:
        """Delete a key.

        Args:
            key: The key to delete.
        """
        self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

    def get_document_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        """Get a document by its chunk ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            Document data or None if not found.
        """
        return self.get(chunk_id)

    def filter_keys(self, keys: List[str]) -> List[str]:
        """Filter out keys that already exist.

        Args:
            keys: List of keys to check.

        Returns:
            List of keys that don't exist in the store.
        """
        return [k for k in keys if k not in self._data]

    def _load_from_disk(self) -> None:
        """Load data from disk."""
        file_path = Path(self.working_dir) / f"{self.namespace}_kv.json"

        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def save_to_disk(self) -> None:
        """Save data to disk."""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        file_path = Path(self.working_dir) / f"{self.namespace}_kv.json"

        with open(file_path, "w") as f:
            json.dump(self._data, f)

    @property
    def count(self) -> int:
        """Get the number of items in the store."""
        return len(self._data)
