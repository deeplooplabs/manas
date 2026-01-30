"""Community and clustering data structures for HiRAG."""

from dataclasses import dataclass, field
from typing import TypedDict


class SingleCommunitySchema(TypedDict):
    """Schema for a single community without report."""

    level: int
    title: str
    edges: list[tuple[str, str]]
    nodes: list[str]
    chunk_ids: list[str]
    occurrence: float
    sub_communities: list[str]


class CommunitySchema(SingleCommunitySchema):
    """Schema for a community with generated report."""

    report_string: str
    report_json: dict


@dataclass
class Community:
    """Represents a community of entities in the knowledge graph.

    A community is a cluster of closely related entities detected by
    the Leiden algorithm. Communities form a hierarchical structure
    where higher-level communities contain lower-level ones.

    Attributes:
        level: Hierarchical level (0 = lowest, most specific).
        title: Human-readable title/description of the community.
        nodes: List of entity names in this community.
        edges: List of (source, target) tuples representing relationships.
        chunk_ids: Source document chunk IDs.
        occurrence: Frequency score of this community.
        sub_communities: IDs of sub-communities at lower levels.
        report_string: Generated summary report text.
        report_json: Structured report data with rating and findings.
        community_id: Unique identifier for this community.
    """

    level: int
    title: str
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    occurrence: float = 0.0
    sub_communities: list[str] = field(default_factory=list)
    report_string: str = ""
    report_json: dict = field(default_factory=dict)
    community_id: str = ""

    def to_schema(self) -> CommunitySchema:
        """Convert to CommunitySchema TypedDict."""
        return CommunitySchema(
            level=self.level,
            title=self.title,
            edges=self.edges,
            nodes=self.nodes,
            chunk_ids=self.chunk_ids,
            occurrence=self.occurrence,
            sub_communities=self.sub_communities,
            report_string=self.report_string,
            report_json=self.report_json,
        )

    def has_entity(self, entity_name: str) -> bool:
        """Check if an entity is in this community."""
        return entity_name in self.nodes

    def add_edge(self, src: str, tgt: str) -> None:
        """Add an edge to the community."""
        edge = tuple(sorted((src, tgt)))
        if edge not in self.edges:
            self.edges.append(edge)
