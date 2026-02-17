"""Session and tool learning extraction using LLM.

This module provides functions for extracting learnings from session
transcripts and tool usage contexts.
"""

import logging
import re

from cems.lib.json_parsing import extract_json_from_response, parse_json_list
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

# Learning types that can be extracted from sessions
LEARNING_TYPES = [
    "WORKING_SOLUTION",  # Code patterns that worked
    "FAILED_APPROACH",   # What didn't work and why
    "USER_PREFERENCE",   # User's stated preferences
    "ERROR_FIX",         # How errors were resolved
    "DECISION",          # Architectural/design decisions made
    "GUIDELINE",         # Rules, conventions, best practices
]

# Categories used as SQL filters in application logic (set by code, not LLMs).
# All other categories are free-text display labels from LLM extraction.
FUNCTIONAL_CATEGORIES = {
    "gate-rules",       # PreToolUse hook blocking
    "preferences",      # Session start profile
    "guidelines",       # Session start profile
    "observation",      # Observation reflection pipeline
    "session-summary",  # Session analyze upserts
    "project",          # Project-specific context
}

# Noise patterns to filter from extracted learnings
_NOISE_PATTERNS = [
    "/private/tmp/claude",
    "/tmp/claude-",
    "background command",
    "exit code 0",
    "task-notification",
    "task notification",
]

# Minimum content length for a learning to be stored
MIN_LEARNING_LENGTH = 80

# Minimum confidence threshold for storing learnings
MIN_CONFIDENCE = 0.6


def normalize_category(raw: str) -> str:
    """Normalize a category string to a clean lowercase-hyphenated form.

    Functional categories (gate-rules, preferences, etc.) are preserved as-is.
    All other categories are free-text from LLM output — just cleaned up for
    consistent display. No alias mapping or canonicalization.

    Args:
        raw: Raw category string from LLM output

    Returns:
        Cleaned category string (lowercase, hyphenated)
    """
    return raw.lower().strip().replace(" ", "-").replace("_", "-").replace("/", "-") or "general"


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
- "category": MUST be one of: api, ai, architecture, cems, configuration, css, database, debugging, deployment, design, development, email, environment, error-handling, frontend, general, git, hooks, infrastructure, preferences, project-management, refactoring, ruby, security, seo, task-management, testing, ui, workflow

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
        response = client.complete(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3,  # Slightly higher for more creative extraction
            model=model or "x-ai/grok-4.1-fast",
        )

        # Parse JSON response
        learnings = _parse_learnings_response(response)

        # Lower confidence threshold - be generous with what we store
        learnings = [l for l in learnings if l.get("confidence", 0) >= MIN_CONFIDENCE]

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

    # Cap chunks to prevent 200+ sequential LLM calls that cause OOM.
    # 10 chunks × 6KB = 60KB of content — more than enough for learning extraction.
    MAX_CHUNKS = 10
    if len(chunks) > MAX_CHUNKS:
        # Take first half + last half to capture setup and conclusions
        half = MAX_CHUNKS // 2
        original_count = len(chunks)
        chunks = chunks[:half] + chunks[-half:]
        logger.warning(
            f"Capped {original_count} chunks to {MAX_CHUNKS} to prevent OOM"
        )

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
                temperature=0.3,
                model=model or "meta-llama/llama-3.1-8b-instruct",
            )

            chunk_learnings = _parse_learnings_response(response)
            chunk_learnings = [l for l in chunk_learnings if l.get("confidence", 0) >= MIN_CONFIDENCE]

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
    # Parse JSON using shared utility
    learnings = parse_json_list(response, fallback=[])

    # Validate each learning
    valid_learnings = []
    for learning in learnings:
        if not isinstance(learning, dict):
            continue
        if learning.get("type") not in LEARNING_TYPES:
            continue
        if not learning.get("content"):
            continue

        content = str(learning["content"])[:500]

        # Skip short content — too vague to be useful
        if len(content) < MIN_LEARNING_LENGTH:
            continue

        # Skip noise content (tmp paths, background commands, exit codes)
        content_lower = content.lower()
        if any(pattern in content_lower for pattern in _NOISE_PATTERNS):
            continue

        # Normalize category to canonical vocabulary
        raw_category = normalize_category(str(learning.get("category", "general")))

        valid_learnings.append({
            "type": learning["type"],
            "content": content,
            "confidence": float(learning.get("confidence", 0.5)),
            "category": raw_category,
        })

    return valid_learnings


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
    import json

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
            temperature=0.2,  # Lower temperature for consistency
            model=model or "x-ai/grok-4.1-fast",
        )

        # Parse response
        response = response.strip()

        # Handle null response
        if response.lower() in ("null", "none", "{}"):
            return None

        # Use shared utility for extraction
        cleaned = extract_json_from_response(response)

        learning = json.loads(cleaned)

        if not learning or not isinstance(learning, dict):
            return None

        # Validate learning
        if learning.get("type") not in LEARNING_TYPES:
            return None
        if not learning.get("content"):
            return None
        if learning.get("confidence", 0) < MIN_CONFIDENCE:
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
