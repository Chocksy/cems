"""Memory summarization and merging using LLM.

This module provides functions for summarizing and merging memories
using LLM-powered text generation.
"""

import logging

from cems.llm.client import get_client

logger = logging.getLogger(__name__)


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
