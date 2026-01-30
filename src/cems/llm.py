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
    # Default models - using GPT-4o-mini for reliability and system prompt support
    "default": "openai/gpt-4o-mini",
    "fast": "openai/gpt-4o-mini",
    "smart": "anthropic/claude-sonnet-4",
    # Explicit mappings from common model names
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
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
5. **Allow appropriate length** - Simple facts should be 1-2 sentences, but detailed guidelines, workflows, or multi-step procedures can be 1-2 paragraphs

## Example Input (simple fact)
Memory 1: "User prefers Python for backend work"
Memory 2: "User likes Python, especially FastAPI framework"
Memory 3: "User prefers Python over JavaScript for server-side code"

## Example Output (simple fact)
User prefers Python for backend/server-side development, especially using the FastAPI framework (over JavaScript)

## Example Input (detailed guideline)
Memory 1: "For RSpec tests, always use 'its' helper instead of standalone 'it' blocks"
Memory 2: "RSpec preference: use is_expected.to instead of explicit expect() calls"
Memory 3: "In RSpec, prefer subject-based testing with its_block for complex assertions"

## Example Output (detailed guideline)
RSpec testing conventions: Always use 'its', 'its_block', and 'its_call' helpers instead of standalone 'it' blocks. Prefer is_expected.to over explicit expect() calls. For complex assertions, use subject-based testing with its_block helper.

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
    "GUIDELINE",         # Rules, conventions, best practices
]


def chunk_content(content: str, max_chunk_size: int = 6000) -> list[str]:
    """Split long content intelligently for processing.

    Strategy:
    1. If under max_chunk_size, return as-is
    2. Try splitting by markdown ## headers
    3. If chunks still too large, split by paragraphs
    4. Ensure reasonable chunk sizes

    Args:
        content: The content to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of content chunks
    """
    import re

    if len(content) <= max_chunk_size:
        return [content]

    chunks = []

    # Try splitting by markdown headers first (## or ###)
    sections = re.split(r"\n(?=##?\s)", content)

    current_chunk = ""
    for section in sections:
        # If adding this section would exceed limit, save current and start new
        if len(current_chunk) + len(section) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # If single section is too large, split by paragraphs
            if len(section) > max_chunk_size:
                paragraphs = section.split("\n\n")
                para_chunk = ""
                for para in paragraphs:
                    if len(para_chunk) + len(para) > max_chunk_size:
                        if para_chunk.strip():
                            chunks.append(para_chunk.strip())
                        para_chunk = para + "\n\n"
                    else:
                        para_chunk += para + "\n\n"
                current_chunk = para_chunk
            else:
                current_chunk = section
        else:
            current_chunk += section

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [content[:max_chunk_size]]


def _get_size_guidance(input_size: int) -> str:
    """Get extraction guidance based on input size."""
    if input_size > 10000:
        return (
            f"This is a LARGE document ({input_size:,} chars). "
            "Extract ALL distinct learnings, guidelines, rules, and conventions. "
            "Expect 15-50+ items. Each rule or guideline should be a separate learning."
        )
    elif input_size > 3000:
        return (
            f"This is a medium-sized input ({input_size:,} chars). "
            "Extract all significant learnings - expect 5-15 items."
        )
    else:
        return (
            f"This is a short input ({input_size:,} chars). "
            "Extract the key learnings - expect 1-5 items."
        )


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

    # Get adaptive guidance based on input size
    input_size = len(transcript_text)
    size_guidance = _get_size_guidance(input_size)

    system_prompt = """You are a Learning Extractor. Analyze ANY input (transcripts, summaries, notes, documents, or guidelines) and extract useful learnings to store for future reference.

## Learning Types

1. **WORKING_SOLUTION** - Code patterns, commands, configurations, or approaches that work
2. **FAILED_APPROACH** - What didn't work and why (helps avoid repeating mistakes)
3. **USER_PREFERENCE** - User preferences about tools, styles, workflows, naming conventions
4. **ERROR_FIX** - How specific errors were diagnosed and fixed
5. **DECISION** - Design/architectural decisions with their reasoning
6. **GUIDELINE** - Rules, conventions, best practices, or standards to follow

## Output Format

Return a JSON array. Each learning needs:
- "type": One of the 6 types above
- "content": Clear, actionable learning (can be 1-3 sentences for simple items, or longer for detailed guidelines)
- "confidence": 0.0-1.0 (how useful is this to remember)
- "category": Topic category (e.g., "docker", "deployment", "git", "python", "testing", "rspec", "preferences")

Extract ALL learnings you find - be generous, not restrictive. For documents with many rules/guidelines, extract EACH ONE as a separate item.

If the input is a summary that mentions learnings, extract each one as a separate item.
If truly nothing useful to extract, return: []

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
    "type": "GUIDELINE",
    "content": "In RSpec tests, ALWAYS use 'its', 'its_block', 'its_call' helpers instead of standalone 'it' blocks. Never use explicit expect() - prefer is_expected.to instead.",
    "confidence": 1.0,
    "category": "rspec"
  }
]
```"""

    # For very long content, process in chunks
    if input_size > 12000:
        return _extract_from_chunks(
            transcript_text,
            context_section,
            system_prompt,
            model,
        )

    prompt = f"""Analyze this input and extract ALL learnings worth remembering.

## Context
{context_section}

## Scale Guidance
{size_guidance}

## Input (transcript, summary, notes, or document)
{transcript_text}

## Instructions
Extract every useful learning from this input. The input might be a raw transcript, a summary, notes, or a full document with guidelines - extract learnings from whatever format is provided.

For documents with rules/guidelines/conventions, extract EACH distinct rule as a separate learning item.

Return ONLY a JSON array (no markdown, no explanation).
If truly nothing useful, return: []"""

    try:
        client = get_client()
        # Use more tokens for larger inputs
        max_tokens = 4000 if input_size > 5000 else 2000

        response = client.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Slightly higher for more creative extraction
            model=model or "openai/gpt-4o-mini",  # Fast, reliable, supports system prompts
        )

        # Parse JSON response
        learnings = _parse_learnings_response(response)

        # Lower confidence threshold - be generous with what we store
        learnings = [l for l in learnings if l.get("confidence", 0) >= 0.3]

        logger.info(f"Extracted {len(learnings)} learnings from session")
        return learnings

    except Exception as e:
        logger.error(f"Failed to extract session learnings: {e}")
        return []


def _extract_from_chunks(
    content: str,
    context_section: str,
    system_prompt: str,
    model: str | None = None,
) -> list[dict]:
    """Extract learnings from content by processing in chunks.

    Used for very long documents (> 12k chars) that need to be split.

    Args:
        content: The full content to process
        context_section: Context info to include in each chunk prompt
        system_prompt: The system prompt to use
        model: Optional model override

    Returns:
        Aggregated and deduplicated list of learnings
    """
    chunks = chunk_content(content, max_chunk_size=6000)
    all_learnings = []

    logger.info(f"Processing content in {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        chunk_guidance = _get_size_guidance(chunk_size)

        prompt = f"""Analyze this section (part {i + 1} of {len(chunks)}) and extract ALL learnings.

## Context
{context_section}

## Scale Guidance
{chunk_guidance}

## Content Section {i + 1}/{len(chunks)}
{chunk}

## Instructions
Extract every useful learning from this section. For documents with rules/guidelines, extract EACH distinct rule as a separate item.

Return ONLY a JSON array (no markdown, no explanation).
If nothing useful in this section, return: []"""

        try:
            client = get_client()
            response = client.complete(
                prompt=prompt,
                system=system_prompt,
                max_tokens=4000,
                temperature=0.3,
                model=model or "meta-llama/llama-3.1-8b-instruct",
            )

            chunk_learnings = _parse_learnings_response(response)
            chunk_learnings = [l for l in chunk_learnings if l.get("confidence", 0) >= 0.3]

            logger.info(f"Chunk {i + 1}/{len(chunks)}: extracted {len(chunk_learnings)} learnings")
            all_learnings.extend(chunk_learnings)

        except Exception as e:
            logger.warning(f"Failed to extract from chunk {i + 1}: {e}")
            continue

    # Deduplicate learnings by content similarity
    deduplicated = _deduplicate_learnings(all_learnings)
    logger.info(f"Total learnings after deduplication: {len(deduplicated)}")

    return deduplicated


def _deduplicate_learnings(learnings: list[dict]) -> list[dict]:
    """Remove duplicate learnings based on content similarity.

    Args:
        learnings: List of learning dicts

    Returns:
        Deduplicated list
    """
    if not learnings:
        return []

    seen_content = set()
    unique = []

    for learning in learnings:
        # Normalize content for comparison
        content = learning.get("content", "").lower().strip()
        # Use first 100 chars as fingerprint to catch near-duplicates
        fingerprint = content[:100]

        if fingerprint not in seen_content:
            seen_content.add(fingerprint)
            unique.append(learning)

    return unique


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


def extract_tool_learning(
    tool_context: str,
    conversation_snippet: str = "",
    working_dir: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Extract a single learning from tool usage context.

    This is a lightweight version of extract_session_learnings designed
    for incremental tool-based learning (SuperMemory-style).

    Args:
        tool_context: Description of the tool that was used (name, input, output)
        conversation_snippet: Recent conversation context (last few messages)
        working_dir: Optional working directory for project context
        model: Optional model override (defaults to haiku for speed)

    Returns:
        Learning dict with keys (type, content, category, confidence) or None if
        nothing worth storing.
    """
    # Skip if context is too brief
    total_context = tool_context + conversation_snippet
    if len(total_context) < 100:
        logger.debug("Tool context too brief for learning extraction")
        return None

    # Build context
    context_parts = []
    if working_dir:
        # Extract project name from path
        project_name = working_dir.split("/")[-1] if "/" in working_dir else working_dir
        context_parts.append(f"Project: {project_name}")

    context_section = "\n".join(context_parts) if context_parts else ""

    system_prompt = """You are a Tool Learning Extractor. Given a tool usage and conversation context, extract ONE meaningful learning if present.

## Learning Types
- WORKING_SOLUTION - A pattern, command, or approach that worked
- ERROR_FIX - How an error was diagnosed and fixed
- DECISION - A design/architecture decision with reasoning
- USER_PREFERENCE - A user preference about tools, styles, or workflows

## Output Format
Return a single JSON object OR null if nothing worth remembering.

{
  "type": "WORKING_SOLUTION",
  "content": "Clear, actionable learning (1-2 sentences)",
  "confidence": 0.7,
  "category": "topic"
}

Be selective - only extract if there's a genuine learning, not routine operations.
Return null for routine reads, searches, or trivial edits."""

    prompt = f"""Analyze this tool usage and extract ONE learning if significant.

{f"## Context\\n{context_section}\\n" if context_section else ""}
## Tool Usage
{tool_context}

{f"## Recent Conversation\\n{conversation_snippet[:1000]}\\n" if conversation_snippet else ""}
## Instructions
If this represents a meaningful pattern, fix, decision, or preference, extract it.
If it's routine (just reading files, simple searches), return null.

Return ONLY valid JSON (object or null)."""

    try:
        client = get_client()
        response = client.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=500,  # Short response expected
            temperature=0.2,  # Lower temperature for consistency
            model=model or "openai/gpt-4o-mini",
        )

        # Parse response
        response = response.strip()

        # Handle null response
        if response.lower() in ("null", "none", "{}"):
            return None

        # Remove markdown code blocks if present
        if response.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()

        import json
        learning = json.loads(response)

        if not learning or not isinstance(learning, dict):
            return None

        # Validate learning
        if learning.get("type") not in LEARNING_TYPES:
            return None
        if not learning.get("content"):
            return None
        if learning.get("confidence", 0) < 0.5:  # Higher threshold for tool learning
            return None

        return {
            "type": learning["type"],
            "content": str(learning["content"])[:500],
            "confidence": float(learning.get("confidence", 0.7)),
            "category": str(learning.get("category", "general"))[:50],
        }

    except Exception as e:
        logger.warning(f"Failed to extract tool learning: {e}")
        return None
