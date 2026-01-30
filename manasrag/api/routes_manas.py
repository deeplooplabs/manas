"""Native ManasRAG API routes.

Endpoints:
- GET /api/health - Health check
- POST /api/query - Query the knowledge graph
- POST /api/documents - Add documents
- GET /api/graph/stats - Get graph statistics
- GET /api/documents - List all documents
- HEAD /api/documents/{doc_id} - Check if document exists
- PUT /api/documents/{doc_id} - Update a document
- DELETE /api/documents/{doc_id} - Delete a document
- POST /api/documents/batch-delete - Batch delete documents
- POST /api/visualize - Generate visualizations
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from haystack.dataclasses import Document

from manasrag import ManasRAG, QueryParam
from manasrag.document_loader import DocumentLoader
from manasrag.api.dependencies import (
    get_manasrag,
    run_in_executor,
    run_index_with_lock,
)
from manasrag.api.models import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    BatchDeleteRequest,
    BatchDeleteResponse,
    DocumentInput,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentUrlInput,
    GraphStatsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    VisualizeRequest,
    VisualizeResponse,
)

router = APIRouter(prefix="/api", tags=["manas"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns {"status": "ok"} if the server is running.
    """
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> QueryResponse:
    """Query the ManasRAG knowledge graph.

    Supports multiple retrieval modes:
    - naive: Simple chunk-based retrieval
    - local: Entity-level retrieval only
    - global: Community report retrieval only
    - bridge: Cross-community path finding
    - nobridge: Local + global without paths
    - hi: Full hierarchical (local + global + bridge)

    Use project_id query parameter for multi-project isolation.
    """
    # Build query parameters
    param = QueryParam(
        mode=request.mode,
        top_k=request.top_k,
        top_m=request.top_m,
        response_type=request.response_type,
        only_need_context=request.only_need_context,
    )

    # Use project_id if provided, else default
    pid = project_id or "default"

    try:
        # Run query in thread executor to avoid blocking
        result = await run_in_executor(
            lambda: manas.query(query=request.query, mode=request.mode, param=param, project_id=pid)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        query=request.query,
        mode=request.mode,
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    )


@router.post("/documents", response_model=AddDocumentsResponse)
async def add_documents(
    request: AddDocumentsRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> AddDocumentsResponse:
    """Add documents to the ManasRAG knowledge graph.

    Documents are processed to extract entities and relations,
    detect communities, and generate community reports.

    Supports both content-based documents (DocumentInput) and URL-based
    documents (DocumentUrlInput). URLs can point to PDF, DOCX, HTML,
    Markdown, XLSX, CSV, or TXT files.

    Use project_id query parameter for multi-project isolation.
    """
    # Convert request documents to Haystack Documents
    documents: list[Document] = []
    urls_to_load: list[str] = []

    for doc in request.documents:
        if isinstance(doc, DocumentUrlInput):
            # Collect URLs for batch loading
            urls_to_load.append(doc.url)
        else:
            # DocumentInput: create Document directly
            meta = doc.meta or {}
            documents.append(Document(id=doc.id, content=doc.content, meta=meta))

    # Load documents from URLs if any
    if urls_to_load:
        loader = DocumentLoader()
        for url in urls_to_load:
            try:
                loaded_docs = loader.load([url])
                # Find the corresponding input to get metadata
                for doc_input in request.documents:
                    if isinstance(doc_input, DocumentUrlInput) and doc_input.url == url:
                        for loaded_doc in loaded_docs:
                            loaded_doc.id = doc_input.id
                            if doc_input.meta:
                                if loaded_doc.meta is None:
                                    loaded_doc.meta = {}
                                loaded_doc.meta.update(doc_input.meta)
                        break
                documents.extend(loaded_docs)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load URL '{url}': {str(e)}")

    if not documents:
        raise HTTPException(status_code=400, detail="No valid documents to index")

    # Use project_id if provided, else default
    pid = project_id or "default"

    try:
        # Run indexing with lock to prevent concurrent writes
        result = await run_index_with_lock(
            lambda: manas.index(
                documents=documents,
                incremental=request.incremental,
                force_reindex=request.force_reindex,
                project_id=pid,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return IndexResponse(
        status=result.get("status", "unknown"),
        documents_count=result.get("documents_count"),
        new_documents=result.get("new_documents"),
        chunks_count=result.get("chunks_count"),
        new_chunks=result.get("new_chunks"),
        entities_count=result.get("entities_count"),
        relations_count=result.get("relations_count"),
        communities_count=result.get("communities_count"),
    )


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> GraphStatsResponse:
    """Get statistics about the knowledge graph.

    Returns counts of entities, relations, communities, and chunks.

    Use project_id query parameter for multi-project isolation.
    """
    # Use project_id if provided, else default
    pid = project_id or "default"

    def _get_stats() -> tuple[int, int, int, int]:
        """Get graph statistics (runs in executor)."""
        # Get project-specific graph store
        graph_store = manas.get_graph_store(pid)

        # Entity count
        entities = (
            graph_store.get_all_entities() if hasattr(graph_store, "get_all_entities") else []
        )
        entities_count = len(entities)

        # Relation count
        relations = (
            graph_store.get_all_relations() if hasattr(graph_store, "get_all_relations") else []
        )
        relations_count = len(relations)

        # Community count - use public property if available
        communities_count = 0
        if hasattr(graph_store, "get_community_count"):
            communities_count = graph_store.get_community_count()
        elif hasattr(graph_store, "_communities"):
            communities_count = len(graph_store._communities)

        # Chunk count
        chunks_count = 0
        if manas.chunk_store and hasattr(manas.chunk_store, "count_documents"):
            chunks_count = manas.chunk_store.count_documents()

        return entities_count, relations_count, communities_count, chunks_count

    try:
        # Run in executor to avoid blocking on potentially slow operations
        entities_count, relations_count, communities_count, chunks_count = await run_in_executor(
            _get_stats
        )
        return GraphStatsResponse(
            entities_count=entities_count,
            relations_count=relations_count,
            communities_count=communities_count,
            chunks_count=chunks_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# Document Management Endpoints
# ============================================================================


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> DocumentListResponse:
    """List all document IDs in the system.

    Returns a list of external document IDs that have been indexed.

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    try:
        doc_ids = await run_in_executor(lambda: manas.list_documents(project_id=pid))
        return DocumentListResponse(doc_ids=doc_ids, count=len(doc_ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.head("/documents/{doc_id}")
async def check_document(
    doc_id: str,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> Response:
    """Check if a document exists.

    Returns 200 if document exists, 404 if not.

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    try:
        exists = await run_in_executor(lambda: manas.has_document(doc_id, project_id=pid))
        if exists:
            return Response(status_code=200)
        return Response(status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check document: {str(e)}")


@router.put("/documents/{doc_id}", response_model=AddDocumentsResponse)
async def update_document(
    doc_id: str,
    request: DocumentUpdateRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> AddDocumentsResponse:
    """Update a document's content.

    Deletes the existing document and re-indexes with new content.

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    try:
        result = await run_index_with_lock(
            lambda: manas.update(doc_id=doc_id, content=request.content, project_id=pid)
        )
        return AddDocumentsResponse(
            status=result.get("status", "updated"),
            documents_count=result.get("documents_count"),
            new_documents=result.get("new_documents"),
            chunks_count=result.get("chunks_count"),
            new_chunks=result.get("new_chunks"),
            entities_count=result.get("entities_count"),
            relations_count=result.get("relations_count"),
            communities_count=result.get("communities_count"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> dict:
    """Delete a single document by ID.

    Removes the document and all associated data (chunks, entities, relations).

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    try:
        result = await run_index_with_lock(lambda: manas.delete(doc_ids=doc_id, project_id=pid))
        return {
            "status": "deleted",
            "doc_id": doc_id,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/documents/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_documents(
    request: BatchDeleteRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> BatchDeleteResponse:
    """Batch delete multiple documents.

    Removes multiple documents and all associated data.

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    try:
        result = await run_index_with_lock(
            lambda: manas.delete(doc_ids=request.doc_ids, project_id=pid)
        )
        return BatchDeleteResponse(
            deleted_count=result.get("deleted_count", len(request.doc_ids)),
            doc_ids=request.doc_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch delete: {str(e)}")


# ============================================================================
# Visualization Endpoints
# ============================================================================


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize(
    request: VisualizeRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    manas: ManasRAG = Depends(get_manasrag),
) -> VisualizeResponse:
    """Generate knowledge graph visualizations.

    Creates interactive HTML visualizations for:
    - graph: Knowledge graph with entities and relations
    - communities: Community structure visualization
    - stats: Entity statistics charts
    - all: Generate all visualizations

    Use project_id query parameter for multi-project isolation.
    """
    pid = project_id or "default"

    # Build kwargs from optional parameters
    kwargs = {}
    if request.layout:
        kwargs["layout"] = request.layout
    if request.color_by:
        kwargs["color_by"] = request.color_by

    try:
        files = await run_in_executor(
            lambda: manas.visualize(kind=request.kind, project_id=pid, **kwargs)
        )
        return VisualizeResponse(files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")
