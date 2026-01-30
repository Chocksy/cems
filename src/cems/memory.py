"""CEMS Memory module (backward compatibility).

This module re-exports from cems.memory.core for backward compatibility.
New code should import from cems.memory directly:

    from cems.memory import CEMSMemory

The CEMSMemory class provides unified memory management using PostgreSQL
with pgvector for both vector embeddings and metadata storage.
"""

# Re-export everything from the memory package for backward compatibility
from cems.memory.core import CEMSMemory, _run_async

__all__ = ["CEMSMemory", "_run_async"]
