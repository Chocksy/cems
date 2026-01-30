"""CEMS Memory package.

This package provides unified memory management using PostgreSQL with pgvector.
The main class CEMSMemory is re-exported here for backward compatibility.

Usage:
    from cems.memory import CEMSMemory

    memory = CEMSMemory()
    memory.add("User prefers TypeScript for frontend")
    results = memory.search("coding preferences")
"""

from cems.memory.core import CEMSMemory, _run_async

__all__ = ["CEMSMemory", "_run_async"]
