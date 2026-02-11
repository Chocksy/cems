"""LLM utilities for CEMS maintenance operations.

Uses OpenRouter exclusively for all LLM calls. OpenRouter acts as a unified
gateway supporting any model (OpenAI, Anthropic, Google, etc.) through the
OpenAI SDK interface.

Environment Variables:
    OPENROUTER_API_KEY: Required. Your OpenRouter API key.
    CEMS_LLM_MODEL: Optional. Model in OpenRouter format (default: anthropic/claude-3-haiku)
    CEMS_OPENROUTER_SITE_URL: Optional. Attribution URL for OpenRouter dashboard.
    CEMS_OPENROUTER_SITE_NAME: Optional. Attribution name for OpenRouter dashboard.

Example:
    from cems.llm import get_client, summarize_memories

    client = get_client()
    response = client.complete("Summarize these items: ...")

    summary = summarize_memories(memories, category="preferences")
"""

# Client exports
from cems.llm.client import (
    FAST_PROVIDERS,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS,
    OpenRouterClient,
    get_client,
    get_llm_client,  # Deprecated, kept for backwards compatibility
)

# Summarization exports
from cems.llm.summarization import (
    merge_memory_contents,
    summarize_memories,
)

# Learning extraction exports
from cems.llm.learning_extraction import (
    LEARNING_TYPES,
    chunk_content,
    extract_session_learnings,
    extract_tool_learning,
    normalize_category,
)

# Observation extraction exports
from cems.llm.observation_extraction import (
    extract_observations,
)

__all__ = [
    # Client
    "OpenRouterClient",
    "get_client",
    "get_llm_client",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODELS",
    "FAST_PROVIDERS",
    # Summarization
    "summarize_memories",
    "merge_memory_contents",
    # Learning extraction
    "LEARNING_TYPES",
    "extract_session_learnings",
    "extract_tool_learning",
    "chunk_content",
    "normalize_category",
    # Observation extraction
    "extract_observations",
]
