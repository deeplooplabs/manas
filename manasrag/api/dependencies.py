"""FastAPI dependencies for ManasRAG API.

This module provides dependency injection for accessing the ManasRAG instance
and other shared resources across route handlers.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from fastapi import Request

if TYPE_CHECKING:
    from manasrag import ManasRAG


T = TypeVar("T")

# Module-level state (FastAPI runs in a single process, so this is safe)
_manas_instance: "ManasRAG | None" = None
_executor: ThreadPoolExecutor | None = None
_index_lock: threading.Lock | None = None


def set_manasrag_instance(manas: "ManasRAG") -> None:
    """Set the global ManasRAG instance.

    Called during app startup/lifespan.
    """
    global _manas_instance
    _manas_instance = manas


def get_manasrag_instance() -> "ManasRAG | None":
    """Get the global ManasRAG instance.

    Returns None if not initialized.
    """
    return _manas_instance


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


def get_manasrag(request: Request) -> "ManasRAG":
    """FastAPI dependency to get ManasRAG instance.

    This is used in route handlers via Depends(get_manasrag).

    Args:
        request: FastAPI request object (for accessing app state).

    Returns:
        ManasRAG instance.

    Raises:
        RuntimeError: If ManasRAG is not initialized.
    """
    manas = request.app.state.manas
    if manas is None:
        raise RuntimeError("ManasRAG instance not initialized")
    return manas


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
