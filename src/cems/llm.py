"""LLM utilities for CEMS maintenance operations.

Uses OpenRouter exclusively for all LLM calls. OpenRouter acts as a unified
gateway supporting any model (OpenAI, Anthropic, Google, etc.) through the
OpenAI SDK interface.

Environment Variables:
    OPENROUTER_API_KEY: Required. Your OpenRouter API key.
    CEMS_LLM_MODEL: Optional. Model in OpenRouter format (default: anthropic/claude-3-haiku)
    CEMS_OPENROUTER_SITE_URL: Optional. Attribution URL for OpenRouter dashboard.
    CEMS_OPENROUTER_SITE_NAME: Optional. Attribution name for OpenRouter dashboard.

Note:
    Both Mem0 and CEMS maintenance use OPENROUTER_API_KEY. Mem0 uses the native
    OpenRouter LLM provider for fact extraction, and the OpenAI embedder with
    openai_base_url pointing to OpenRouter for embeddings.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model mapping for OpenRouter (provider/model format)
OPENROUTER_MODELS = {
    # Default models for maintenance tasks
    "default": "anthropic/claude-3-haiku",
    "fast": "anthropic/claude-3-haiku",
    "smart": "anthropic/claude-sonnet-4",
    # Explicit mappings from common model names
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
}


class OpenRouterClient:
    """OpenRouter LLM client for CEMS maintenance operations.

    This is the central LLM interface for all CEMS operations. Uses OpenRouter
    as a unified gateway to access any LLM provider.

    Example:
        client = OpenRouterClient()
        response = client.complete("Summarize these items: ...")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            model: Model in OpenRouter format. Defaults to CEMS_LLM_MODEL or anthropic/claude-3-haiku.
            site_url: Attribution URL for OpenRouter dashboard.
            site_name: Attribution name for OpenRouter dashboard.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = self._resolve_model(model or os.getenv("CEMS_LLM_MODEL"))
        self.site_url = site_url or os.getenv("CEMS_OPENROUTER_SITE_URL", "https://github.com/cems")
        self.site_name = site_name or os.getenv("CEMS_OPENROUTER_SITE_NAME", "CEMS Memory Server")

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
        )

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a model name to OpenRouter format.

        Args:
            model: Model name (can be standard or OpenRouter format)

        Returns:
            Model name in OpenRouter format (provider/model)
        """
        if model is None:
            return OPENROUTER_MODELS["default"]

        # Already in OpenRouter format
        if "/" in model:
            return model

        # Map known models
        if model in OPENROUTER_MODELS:
            return OPENROUTER_MODELS[model]

        # Assume it's already valid
        return model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0-2)
            model: Override model for this call

        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


# Module-level client (lazy initialization)
_client: OpenRouterClient | None = None


def get_client() -> OpenRouterClient:
    """Get the shared OpenRouter client instance.

    Returns:
        OpenRouterClient instance

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


def summarize_memories(
    memories: list[str],
    category: str,
    model: str | None = None,
) -> str:
    """Use LLM to summarize a list of memories into a coherent summary.

    Args:
        memories: List of memory content strings
        category: The category being summarized
        model: Optional model override (OpenRouter format)

    Returns:
        A markdown summary of the memories
    """
    if not memories:
        return f"No memories in category '{category}'."

    memories_text = "\n".join(f"- {m}" for m in memories)

    # Enhanced prompt with few-shot example
    system_prompt = """You are a Memory Summarization Specialist. Your job is to compress a list of individual memory items into a coherent, structured summary that can be used for future context retrieval.

## Output Format
Return a markdown document with:
1. A brief overview (1-2 sentences)
2. Key facts as bullet points (preserve specifics like names, numbers, versions)
3. Patterns or preferences identified
4. Any contradictions resolved (newer info takes precedence)

## Example Input
Category: preferences
- User prefers Python for backend development
- User likes FastAPI over Flask
- User mentioned they prefer dark mode in IDEs
- User prefers TypeScript over JavaScript for frontend
- User dislikes PHP

## Example Output
### Preferences Summary

Developer with strong opinions on technology choices and tooling.

**Languages & Frameworks:**
- Backend: Python (prefers FastAPI over Flask)
- Frontend: TypeScript (over JavaScript)
- Avoids: PHP

**Environment:**
- IDE: Dark mode preference

**Pattern:** Tends toward modern, type-safe technologies with good developer experience."""

    prompt = f"""Summarize these memories from category "{category}":

{memories_text}

Create a structured summary following the format shown in my instructions. Keep it under 500 words. Focus on actionable, retrievable information."""

    try:
        client = get_client()
        return client.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=1000,
            temperature=0.3,
            model=model,
        )
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return _fallback_summary(memories, category)


def merge_memory_contents(
    memories: list[dict],
    model: str | None = None,
) -> str:
    """Use LLM to merge duplicate memories into a single coherent memory.

    Args:
        memories: List of memory dicts with 'memory' key containing content
        model: Optional model override (OpenRouter format)

    Returns:
        Merged memory content
    """
    if not memories:
        return ""

    if len(memories) == 1:
        return memories[0].get("memory", "")

    contents = [m.get("memory", "") for m in memories if m.get("memory")]
    if not contents:
        return ""

    contents_text = "\n---\n".join(contents)

    # Enhanced prompt with conflict resolution guidance
    system_prompt = """You are a Memory Consolidation Specialist. Your job is to merge semantically similar memories into a single, comprehensive memory.

## Rules
1. **Combine unique information** - Include all distinct facts from both memories
2. **Resolve conflicts** - If memories contradict, prefer the more specific or recent-sounding version
3. **Remove redundancy** - Don't repeat the same information twice
4. **Preserve specifics** - Keep exact names, numbers, versions, URLs
5. **Keep it atomic** - Output should be a single fact/preference, not a list

## Example Input
Memory 1: "User prefers Python for backend work"
Memory 2: "User likes Python, especially FastAPI framework"
Memory 3: "User prefers Python over JavaScript for server-side code"

## Example Output
User prefers Python for backend/server-side development, especially using the FastAPI framework (over JavaScript)

## Example Input (with conflict)
Memory 1: "User works at Google"
Memory 2: "User works at OpenAI as an engineer"

## Example Output
User works at OpenAI as an engineer (previously at Google)"""

    prompt = f"""Merge these similar memories into ONE comprehensive memory:

{contents_text}

Output ONLY the merged memory content. No explanations, no markdown, just the merged fact."""

    try:
        client = get_client()
        return client.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=300,
            temperature=0.2,
            model=model,
        )
    except Exception as e:
        logger.error(f"LLM merge failed: {e}")
        return contents[0]


def _fallback_summary(memories: list[str], category: str) -> str:
    """Create a simple fallback summary without LLM.

    Args:
        memories: List of memory content strings
        category: The category name

    Returns:
        Simple bullet-point summary
    """
    summary = f"## {category.title()}\n\n"
    summary += f"Contains {len(memories)} memories:\n\n"

    # Take first few memories as examples
    for i, mem in enumerate(memories[:5]):
        truncated = mem[:100] + "..." if len(mem) > 100 else mem
        summary += f"- {truncated}\n"

    if len(memories) > 5:
        summary += f"\n... and {len(memories) - 5} more.\n"

    return summary


# Backwards compatibility - these functions work but use OpenRouter now
def get_llm_client(provider: str = "openrouter"):
    """Get the LLM client.

    DEPRECATED: Use get_client() instead. This function is kept for backwards
    compatibility but always returns an OpenRouter-based client.

    Args:
        provider: Ignored. OpenRouter is always used.

    Returns:
        The underlying OpenAI client configured for OpenRouter
    """
    logger.warning(
        "get_llm_client() is deprecated. Use get_client() which returns OpenRouterClient."
    )
    return get_client()._client


def _resolve_openrouter_model(model: str | None) -> str:
    """Resolve a model name to OpenRouter format.

    DEPRECATED: Use OpenRouterClient._resolve_model() instead.
    """
    client = OpenRouterClient.__new__(OpenRouterClient)
    return client._resolve_model(model)
