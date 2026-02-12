"""CEMS - Continuous Evolving Memory System.

A dual-layer memory system (personal + shared) with scheduled maintenance,
built on top of Mem0 and exposed as an MCP server.
"""

from importlib.metadata import version

from cems.config import CEMSConfig
from cems.memory import CEMSMemory

__version__ = version("cems")
__all__ = ["CEMSMemory", "CEMSConfig", "__version__"]
