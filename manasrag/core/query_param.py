"""Query parameters for HiRAG retrieval modes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class RetrievalMode(str, Enum):
    """Supported retrieval modes."""

    NAIVE = "naive"  # Basic RAG with document chunks
    LOCAL = "local"  # Local knowledge: entities + relations + chunks
    GLOBAL = "global"  # Global knowledge: community reports + chunks
    BRIDGE = "bridge"  # Bridge knowledge: cross-community paths
    HI = "hi"  # Full hierarchical: all of the above
    NOBRIDGE = "nobridge"  # Hierarchical without bridge paths


@dataclass
class QueryParam:
    """Parameters for configuring retrieval behavior.

    Attributes:
        mode: The retrieval mode to use.
        only_need_context: If True, return context without generating answer.
        response_type: Expected response format.
        level: Maximum community level to retrieve.
        top_k: Number of top entities to retrieve.
        top_m: Number of key entities per community.
        max_token_for_text_unit: Max tokens for text units.
        max_token_for_local_context: Max tokens for local context.
        max_token_for_bridge_knowledge: Max tokens for bridge paths.
        max_token_for_community_report: Max tokens for community reports.
        naive_max_token_for_text_unit: Max tokens for naive mode text units.
        community_single_one: If True, only use top community.
    """

    mode: Literal["hi", "local", "global", "bridge", "nobridge", "naive"] = "hi"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    top_m: int = 10

    # Token limits
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    naive_max_token_for_text_unit: int = 10000

    community_single_one: bool = False

    @classmethod
    def naive(cls, **kwargs) -> "QueryParam":
        """Create params for naive RAG mode."""
        return cls(mode="naive", **kwargs)

    @classmethod
    def local(cls, **kwargs) -> "QueryParam":
        """Create params for local knowledge mode."""
        return cls(mode="local", **kwargs)

    @classmethod
    def global_mode(cls, **kwargs) -> "QueryParam":
        """Create params for global knowledge mode."""
        return cls(mode="global", **kwargs)

    @classmethod
    def bridge(cls, **kwargs) -> "QueryParam":
        """Create params for bridge knowledge mode."""
        return cls(mode="bridge", **kwargs)

    @classmethod
    def hi(cls, **kwargs) -> "QueryParam":
        """Create params for full hierarchical mode."""
        return cls(mode="hi", **kwargs)
