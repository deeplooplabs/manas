"""Core graph data structures for HiRAG."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """Common entity types extracted from text."""

    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    TECHNICAL_TERM = "TECHNICAL_TERM"
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    """Represents an entity extracted from documents.

    Attributes:
        entity_name: The unique name/identifier of the entity.
        entity_type: The type classification of the entity.
        description: A textual description of the entity.
        source_id: IDs of source chunks where this entity appears.
        clusters: List of community assignments with level information.
        embedding: Vector embedding of the entity description.
    """

    entity_name: str
    entity_type: str
    description: str
    source_id: str = ""
    clusters: list[dict] = field(default_factory=list)
    embedding: Optional[list[float]] = None

    def add_cluster(self, level: int, cluster_id: str) -> None:
        """Add a community cluster assignment."""
        self.clusters.append({"level": level, "cluster": cluster_id})

    def get_clusters_at_level(self, level: int) -> list[str]:
        """Get all cluster IDs at a specific level."""
        return [c["cluster"] for c in self.clusters if c["level"] == level]


@dataclass
class Relation:
    """Represents a relationship between two entities.

    Attributes:
        src_id: Source entity name.
        tgt_id: Target entity name.
        weight: Strength/frequency of the relationship.
        description: Textual description of the relationship.
        source_id: IDs of source chunks where this relation appears.
        order: Order of appearance (for DSPy predictions).
    """

    src_id: str
    tgt_id: str
    weight: float = 1.0
    description: str = ""
    source_id: str = ""
    order: int = 1

    @property
    def sorted_pair(self) -> tuple[str, str]:
        """Return the entity pair sorted alphabetically."""
        return tuple(sorted((self.src_id, self.tgt_id)))

    def to_tuple(self) -> tuple:
        """Convert to tuple representation."""
        return (self.src_id, self.tgt_id, self.weight, self.description, self.source_id)
