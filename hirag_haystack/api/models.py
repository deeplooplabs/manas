"""Pydantic models for HiRAG REST API.

This module defines request/response schemas for both native HiRAG endpoints
and OpenAI-compatible endpoints.
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Native HiRAG API Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class QueryRequest(BaseModel):
    """Native HiRAG query request."""

    query: str = Field(..., description="The query string")
    mode: Literal["naive", "hi_local", "hi_global", "hi_bridge", "hi_nobridge", "hi"] = Field(
        default="hi",
        description="Retrieval mode",
    )
    top_k: int = Field(default=20, ge=1, description="Number of entities to retrieve")
    top_m: int = Field(default=10, ge=1, description="Key entities per community")
    response_type: str = Field(
        default="Multiple Paragraphs",
        description="Expected response format",
    )
    only_need_context: bool = Field(
        default=False,
        description="If true, return only context without generating answer",
    )


class QueryResponse(BaseModel):
    """Native HiRAG query response."""

    query: str
    mode: str
    answer: str = ""
    context: str = ""


class DocumentInput(BaseModel):
    """Document input for indexing."""

    id: str | None = Field(default=None, description="Optional document ID")
    content: str = Field(
        ...,
        min_length=1,
        description="Document content (must not be empty)",
    )
    meta: dict | None = Field(default=None, description="Optional metadata")


class DocumentUrlInput(BaseModel):
    """Document URL input for indexing.

    Loads document content from a URL (http/https).
    Supports PDF, DOCX, HTML, Markdown, XLSX, CSV, TXT files.
    """

    url: str = Field(
        ...,
        description="Document URL (http/https)",
    )
    meta: dict | None = Field(default=None, description="Optional metadata")


class AddDocumentsRequest(BaseModel):
    """Request for adding documents to the knowledge graph."""

    documents: list[DocumentInput | DocumentUrlInput] = Field(
        ..., min_length=1, description="Documents to add (content or URL)"
    )
    incremental: bool = Field(default=True, description="Skip already-indexed documents")
    force_reindex: bool = Field(default=False, description="Force reindex all documents")


class AddDocumentsResponse(BaseModel):
    """Response for document addition."""

    status: str
    documents_count: int | None = None
    new_documents: int | None = None
    chunks_count: int | None = None
    new_chunks: int | None = None
    entities_count: int | None = None
    relations_count: int | None = None
    communities_count: int | None = None


# Backward compatibility aliases
IndexRequest = AddDocumentsRequest
IndexResponse = AddDocumentsResponse


class GraphStatsResponse(BaseModel):
    """Graph statistics response."""

    entities_count: int
    relations_count: int
    communities_count: int
    chunks_count: int


# ============================================================================
# Document Management Models
# ============================================================================


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    doc_ids: list[str]
    count: int


class DocumentUpdateRequest(BaseModel):
    """Request for updating a document."""

    content: str = Field(..., min_length=1, description="New document content")
    meta: dict | None = Field(default=None, description="Optional metadata")


class BatchDeleteRequest(BaseModel):
    """Request for batch deleting documents."""

    doc_ids: list[str] = Field(..., min_length=1, description="Document IDs to delete")


class BatchDeleteResponse(BaseModel):
    """Response for batch delete operation."""

    deleted_count: int
    doc_ids: list[str]


# ============================================================================
# Visualization Models
# ============================================================================


class VisualizeRequest(BaseModel):
    """Request for generating visualizations."""

    kind: Literal["graph", "communities", "stats", "all"] = Field(
        default="all",
        description="Type of visualization to generate",
    )
    layout: str | None = Field(
        default=None,
        description="Layout algorithm (force, hierarchical, circular)",
    )
    color_by: str | None = Field(
        default=None,
        description="Coloring scheme (entity_type, community, degree)",
    )


class VisualizeResponse(BaseModel):
    """Response containing visualization file paths."""

    files: dict[str, str] = Field(
        ...,
        description="Mapping of visualization name to file path",
    )


# ============================================================================
# OpenAI-Compatible API Models
# ============================================================================

# Model-to-mode mapping
MODEL_MODE_MAPPING: dict[str, str] = {
    "hirag": "hi",
    "hirag-hi": "hi",
    "hirag-local": "hi_local",
    "hirag-global": "hi_global",
    "hirag-bridge": "hi_bridge",
    "hirag-nobridge": "hi_nobridge",
    "hirag-naive": "naive",
}

# Available model names for /v1/models
AVAILABLE_MODELS = list(MODEL_MODE_MAPPING.keys())


class OpenAIModel(BaseModel):
    """OpenAI model object."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "hirag"


class OpenAIModelList(BaseModel):
    """OpenAI model list response."""

    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    HiRAG-specific extensions (top_k, top_m, response_type) are optional
    and will be silently ignored by standard OpenAI clients.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model name (maps to retrieval mode)")
    messages: list[ChatMessage] = Field(..., min_length=1, description="Chat messages")
    stream: bool = Field(default=False, description="Enable streaming response")
    temperature: float | None = Field(default=None, description="Ignored, for compatibility")
    max_tokens: int | None = Field(default=None, description="Ignored, for compatibility")

    # HiRAG-specific extensions
    top_k: int = Field(default=20, ge=1, description="Number of entities to retrieve")
    top_m: int = Field(default=10, ge=1, description="Key entities per community")
    response_type: str = Field(
        default="Multiple Paragraphs",
        description="Expected response format",
    )


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


# Streaming response models


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice in streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# Error response


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail
