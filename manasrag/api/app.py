"""FastAPI application factory for ManasRAG REST API.

This module provides the create_app() factory function that sets up
the FastAPI application with proper lifespan management for ManasRAG
initialization and cleanup.
"""

import os
import threading
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI

from manasrag.api.dependencies import (
    set_executor,
    set_manasrag_instance,
    set_index_lock,
)
from manasrag.api.routes_manas import router as manas_router
from manasrag.api.routes_openai import router as openai_router

if TYPE_CHECKING:
    from manasrag import ManasRAG


@dataclass
class AppConfig:
    """Configuration for the ManasRAG API application.

    Attributes:
        working_dir: Working directory for ManasRAG data.
        model: LLM model name.
        api_key: OpenAI API key.
        base_url: API base URL for custom endpoints.
        graph_backend: Graph backend ("networkx" or "neo4j").
        chunk_size: Token chunk size for document splitting.
        chunk_overlap: Overlap between chunks.
        top_k: Number of entities to retrieve.
        top_m: Key entities per community.
        max_workers: Number of thread pool workers.
    """

    working_dir: str = "./manas_cache"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    graph_backend: str = "networkx"
    chunk_size: int = 1200
    chunk_overlap: int = 100
    top_k: int = 20
    top_m: int = 10
    max_workers: int = 4


def create_app(
    config: AppConfig | None = None,
    manas: "ManasRAG | None" = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration. Used to build ManasRAG if manas is None.
        manas: Pre-configured ManasRAG instance. If provided, config is ignored
               for ManasRAG initialization.

    Returns:
        Configured FastAPI application.

    Example:
        ```python
        from manasrag.api import create_app, AppConfig

        # Using config
        config = AppConfig(
            working_dir="./my_data",
            model="gpt-4o",
            api_key="sk-...",
        )
        app = create_app(config=config)

        # Using pre-configured ManasRAG
        manas = ManasRAG(working_dir="./data", generator=my_generator)
        app = create_app(manas=manas)

        # Run with uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        ```
    """
    config = config or AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage application lifespan.

        Handles ManasRAG initialization on startup and cleanup on shutdown.
        """
        # Initialize thread pool executor
        executor = ThreadPoolExecutor(max_workers=config.max_workers)
        set_executor(executor)

        # Initialize indexing lock
        index_lock = threading.Lock()
        set_index_lock(index_lock)

        # Initialize ManasRAG
        if manas is not None:
            # Use provided ManasRAG instance
            app.state.manas = manas
        else:
            # Build ManasRAG from config
            from manasrag.cli import _build_manasrag, ManasRAGConfigError

            try:
                app.state.manas = _build_manasrag(
                    working_dir=config.working_dir,
                    model=config.model,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    graph_backend=config.graph_backend,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    top_k=config.top_k,
                    top_m=config.top_m,
                )
            except ManasRAGConfigError as e:
                # Re-raise config errors to be handled at startup
                raise RuntimeError(f"ManasRAG configuration error: {e}") from e

        set_manasrag_instance(app.state.manas)

        yield

        # Cleanup
        executor.shutdown(wait=True)
        app.state.manas = None

    app = FastAPI(
        title="ManasRAG API",
        description=(
            "REST API for ManasRAG (Hierarchical Retrieval-Augmented Generation). "
            "Provides native ManasRAG endpoints and OpenAI-compatible chat completions."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routers
    app.include_router(manas_router)
    app.include_router(openai_router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "ManasRAG API",
            "version": "0.1.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "native": ["/api/health", "/api/query", "/api/index", "/api/graph/stats"],
                "openai": ["/v1/models", "/v1/chat/completions"],
            },
        }

    return app


def _create_dev_app() -> FastAPI:
    """Create app for development mode with reload.

    Reads configuration from environment variables set by the CLI.
    This is used when running with --reload flag via uvicorn --factory.
    """
    config = AppConfig(
        working_dir=os.environ.get("_MANAS_WORKING_DIR", "./manas_cache"),
        model=os.environ.get("_MANAS_MODEL") or None,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        graph_backend=os.environ.get("_MANAS_GRAPH_BACKEND", "networkx"),
    )
    return create_app(config=config)
