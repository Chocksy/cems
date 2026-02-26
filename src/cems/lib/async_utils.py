"""Async utilities shared across CEMS modules."""

import asyncio


def run_async(coro):
    """Run an async coroutine in a sync context.

    NOTE: This is for sync contexts only (CLI, MCP stdio).
    For async contexts (HTTP server), use the async methods directly.

    Raises RuntimeError if called from an async context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version (e.g., add_async instead of add)."
        )
    else:
        return asyncio.run(coro)


def run_async_in_thread(coro):
    """Run an async coroutine from a background thread (e.g., scheduler).

    Creates a fresh event loop, runs the coroutine, and closes the loop.
    Use this from threads that don't have an event loop (like APScheduler jobs).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
