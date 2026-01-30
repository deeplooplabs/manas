"""FastAPI dependencies for HiRAG API.

This module provides dependency injection for accessing the HiRAG instance
and other shared resources across route handlers.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from fastapi import Request

if TYPE_CHECKING:
    from hirag_haystack import HiRAG


T = TypeVar("T")

# Module-level state (FastAPI runs in a single process, so this is safe)
_hirag_instance: "HiRAG | None" = None
_executor: ThreadPoolExecutor | None = None
_index_lock: threading.Lock | None = None


def set_hirag_instance(hirag: "HiRAG") -> None:
    """Set the global HiRAG instance.

    Called during app startup/lifespan.
    """
    global _hirag_instance
    _hirag_instance = hirag


def get_hirag_instance() -> "HiRAG | None":
    """Get the global HiRAG instance.

    Returns None if not initialized.
    """
    return _hirag_instance


def set_executor(executor: ThreadPoolExecutor) -> None:
    """Set the thread pool executor."""
    global _executor
    _executor = executor


def get_executor() -> ThreadPoolExecutor | None:
    """Get the thread pool executor."""
    return _executor


def set_index_lock(lock: threading.Lock) -> None:
    """Set the indexing lock."""
    global _index_lock
    _index_lock = lock


def get_index_lock() -> threading.Lock | None:
    """Get the indexing lock."""
    return _index_lock


def get_hirag(request: Request) -> "HiRAG":
    """FastAPI dependency to get HiRAG instance.

    This is used in route handlers via Depends(get_hirag).

    Args:
        request: FastAPI request object (for accessing app state).

    Returns:
        HiRAG instance.

    Raises:
        RuntimeError: If HiRAG is not initialized.
    """
    hirag = request.app.state.hirag
    if hirag is None:
        raise RuntimeError("HiRAG instance not initialized")
    return hirag


async def run_in_executor(func: Callable[..., T], *args: Any) -> T:
    """Run a synchronous function in the thread pool executor.

    Args:
        func: Synchronous function to run.
        *args: Arguments to pass to the function.

    Returns:
        Result of the function.

    Raises:
        RuntimeError: If executor is not initialized.
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()
    if executor is None:
        raise RuntimeError("Thread pool executor not initialized")
    return await loop.run_in_executor(executor, func, *args)


async def run_index_with_lock(func: Callable[..., T], *args: Any) -> T:
    """Run an indexing operation with lock protection.

    Indexing operations are serialized to prevent concurrent writes.

    Args:
        func: Indexing function to run.
        *args: Arguments to pass to the function.

    Returns:
        Result of the function.

    Raises:
        RuntimeError: If index lock is not initialized.
    """
    lock = get_index_lock()
    if lock is None:
        raise RuntimeError("Index lock not initialized. Ensure app lifespan is properly started.")

    def locked_func() -> T:
        with lock:
            return func(*args)

    return await run_in_executor(locked_func)
