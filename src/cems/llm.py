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


# =============================================================================
# Session Learning Extraction
# =============================================================================

# Learning types that can be extracted from sessions
LEARNING_TYPES = [
    "WORKING_SOLUTION",  # Code patterns that worked
    "FAILED_APPROACH",   # What didn't work and why
    "USER_PREFERENCE",   # User's stated preferences
    "ERROR_FIX",         # How errors were resolved
    "DECISION",          # Architectural/design decisions made
]


def extract_session_learnings(
    transcript: str | list[dict],
    working_dir: str | None = None,
    tool_summary: dict | None = None,
    model: str | None = None,
) -> list[dict]:
    """Extract learnings from a session transcript using LLM.

    Analyzes the transcript and extracts significant learnings that should
    be stored for future reference.

    Args:
        transcript: Either a string transcript or list of message dicts
        working_dir: Optional working directory for project context
        tool_summary: Optional dict of tools used, files changed
        model: Optional model override (defaults to haiku for speed)

    Returns:
        List of learning dicts with keys:
        - type: One of LEARNING_TYPES
        - content: The learning content to store
        - confidence: 0-1 confidence score
        - category: Suggested category for storage
    """
    # Convert transcript to string if it's a list of messages
    if isinstance(transcript, list):
        transcript_text = _format_transcript(transcript)
    else:
        transcript_text = transcript

    # Skip very short transcripts
    if len(transcript_text) < 200:
        logger.debug("Transcript too short for learning extraction")
        return []

    # Build context section
    context_parts = []
    if working_dir:
        context_parts.append(f"Project directory: {working_dir}")
    if tool_summary:
        if tool_summary.get("files_changed"):
            context_parts.append(f"Files changed: {', '.join(tool_summary['files_changed'][:10])}")
        if tool_summary.get("tools_used"):
            context_parts.append(f"Tools used: {', '.join(tool_summary['tools_used'][:10])}")

    context_section = "\n".join(context_parts) if context_parts else "No additional context."

    system_prompt = """You are a Session Learning Extractor. Your job is to analyze AI assistant session transcripts and extract learnings worth remembering for future sessions.

## What to Extract

Extract ONLY significant learnings that would help in future sessions:

1. **WORKING_SOLUTION** - Code patterns, commands, or approaches that successfully solved a problem
2. **FAILED_APPROACH** - Approaches that didn't work and why (prevents repeating mistakes)
3. **USER_PREFERENCE** - User's stated preferences about tools, styles, or workflows
4. **ERROR_FIX** - How a specific error was diagnosed and resolved
5. **DECISION** - Architectural or design decisions made with reasoning

## What NOT to Extract

- Generic knowledge that any AI would know
- Temporary debugging steps without lasting value
- Trivial changes or routine operations
- Information already commonly known

## Output Format

Return a JSON array of learnings. Each learning must have:
- "type": One of the 5 types above
- "content": The learning in a clear, reusable format (2-4 sentences max)
- "confidence": 0.0-1.0 (how confident this is worth remembering)
- "category": Suggested category (e.g., "git", "python", "debugging", "preferences")

If no significant learnings, return an empty array: []

Example output:
```json
[
  {
    "type": "WORKING_SOLUTION",
    "content": "For FastAPI datetime serialization issues, use model_dump(mode='json') which automatically converts datetime objects to ISO strings.",
    "confidence": 0.9,
    "category": "python"
  },
  {
    "type": "USER_PREFERENCE",
    "content": "User prefers using uv for Python package management instead of pip.",
    "confidence": 0.8,
    "category": "preferences"
  }
]
```"""

    # Truncate transcript if too long (keep last 8000 chars for context window)
    max_transcript_len = 8000
    if len(transcript_text) > max_transcript_len:
        transcript_text = "...[truncated]...\n" + transcript_text[-max_transcript_len:]

    prompt = f"""Analyze this session transcript and extract learnings worth remembering.

## Context
{context_section}

## Session Transcript
{transcript_text}

## Instructions
Extract significant learnings from this session. Return ONLY a JSON array (no markdown, no explanation).
If nothing significant to learn, return: []"""

    try:
        client = get_client()
        response = client.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=2000,
            temperature=0.2,
            model=model or "anthropic/claude-3-haiku",  # Use haiku by default for speed/cost
        )

        # Parse JSON response
        learnings = _parse_learnings_response(response)

        # Filter by confidence threshold
        learnings = [l for l in learnings if l.get("confidence", 0) >= 0.6]

        logger.info(f"Extracted {len(learnings)} learnings from session")
        return learnings

    except Exception as e:
        logger.error(f"Failed to extract session learnings: {e}")
        return []


def _format_transcript(messages: list[dict]) -> str:
    """Format a list of message dicts into a readable transcript.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted transcript string
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle content that might be a list (tool calls, etc.)
        if isinstance(content, list):
            content_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_use":
                        content_parts.append(f"[Tool: {part.get('name', 'unknown')}]")
                else:
                    content_parts.append(str(part))
            content = " ".join(content_parts)

        # Truncate very long messages
        if len(content) > 1000:
            content = content[:1000] + "...[truncated]"

        if content.strip():
            lines.append(f"**{role.upper()}**: {content}")

    return "\n\n".join(lines)


def _parse_learnings_response(response: str) -> list[dict]:
    """Parse the LLM response into a list of learning dicts.

    Args:
        response: Raw LLM response text

    Returns:
        List of validated learning dicts
    """
    import json
    import re

    # Try to extract JSON from response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```"):
        # Find the end of the code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if match:
            response = match.group(1).strip()

    # Try to parse as JSON
    try:
        learnings = json.loads(response)
        if not isinstance(learnings, list):
            return []

        # Validate each learning
        valid_learnings = []
        for learning in learnings:
            if not isinstance(learning, dict):
                continue
            if learning.get("type") not in LEARNING_TYPES:
                continue
            if not learning.get("content"):
                continue

            valid_learnings.append({
                "type": learning["type"],
                "content": str(learning["content"])[:500],  # Limit content length
                "confidence": float(learning.get("confidence", 0.5)),
                "category": str(learning.get("category", "general"))[:50],
            })

        return valid_learnings

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse learnings JSON: {response[:200]}")
        return []
