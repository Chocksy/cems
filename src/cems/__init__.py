"""CEMS - Continuous Evolving Memory System.

A dual-layer memory system (personal + shared) with scheduled maintenance,
built on top of Mem0 and exposed as an MCP server.
"""

from cems.config import CEMSConfig
from cems.memory import CEMSMemory

__version__ = "0.1.0"
__all__ = ["CEMSMemory", "CEMSConfig", "__version__"]
