"""Inference retrieval pipeline for CEMS.

Implements the 5-stage retrieval pipeline:
1. Query Synthesis - LLM expands query for better retrieval
2. Candidate Retrieval - Vector search + graph traversal
3. Relevance Filtering - Filter candidates below threshold
4. Temporal Ranking - Time decay + priority scoring
5. Token-Budgeted Assembly - Select results within token budget

This module provides the core retrieval logic that powers the unified
memory_search tool.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.llm import OpenRouterClient
    from cems.models import SearchResult

logger = logging.getLogger(__name__)

# Default token encoding for context budgeting
DEFAULT_ENCODING = "cl100k_base"


def synthesize_query(query: str, client: "OpenRouterClient") -> list[str]:
    """Stage 1: Generate better search terms from user query.

    Uses LLM to expand the user's query into multiple search terms
    that might help find relevant memories.

    Args:
        query: Original user query
        client: OpenRouter client for LLM calls

    Returns:
        List of expanded search terms (original not included)
    """
    prompt = f"""Generate 2-3 search queries to find memories about the EXACT SAME TOPIC.

User query: {query}

CRITICAL RULES:
- Stay within the SAME specific domain/topic
- NO generalizing to broader categories
- Prefer specific technical terms over generic words
- Only add synonyms for the exact topic, not related areas

Return one search term per line. No bullets, no numbering."""

    try:
        result = client.complete(prompt, max_tokens=100, temperature=0.3)
        terms = [q.strip() for q in result.strip().split("\n") if q.strip()]
        # Filter out empty or very short terms, limit to 3 expansions
        return [t for t in terms if len(t) > 2][:3]
    except Exception as e:
        logger.warning(f"Query synthesis failed: {e}")
        return []


def calculate_relevance_score(
    base_score: float,
    days_since_access: int,
    priority: float = 1.0,
    pinned: bool = False,
) -> float:
    """Stage 3-4: Calculate final score with relevance + temporal ranking.

    Combines the raw similarity score with time decay and priority factors.

    Args:
        base_score: Raw similarity score from vector search (0-1)
        days_since_access: Days since the memory was last accessed
        priority: Memory priority weight (1.0 default, up to 2.0)
        pinned: Whether the memory is pinned (gets small boost)

    Returns:
        Final adjusted score
    """
    # Time decay: 1.0 / (1.0 + (age_days / 30))
    # - 0 days: 1.0
    # - 30 days: 0.5
    # - 60 days: 0.33
    # - 90 days: 0.25
    time_decay = 1.0 / (1.0 + (days_since_access / 30))

    # Apply priority boost (1.0 default, up to 2.0)
    score = base_score * time_decay * priority

    # Pinned memories get a small boost (10%)
    if pinned:
        score *= 1.1

    return score


def assemble_context(
    results: list["SearchResult"],
    max_tokens: int = 2000,
) -> tuple[list["SearchResult"], int]:
    """Stage 5: Select results within token budget.

    Greedily selects results in score order until the token budget
    is exhausted.

    Args:
        results: List of SearchResult objects, sorted by score
        max_tokens: Maximum tokens to include

    Returns:
        Tuple of (selected results, total tokens used)
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
    except ImportError:
        # Fallback: estimate 4 chars per token
        logger.warning("tiktoken not installed, using character-based estimation")

        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        selected = []
        token_count = 0

        for result in results:
            tokens = estimate_tokens(result.content)
            if token_count + tokens > max_tokens:
                break
            selected.append(result)
            token_count += tokens

        return selected, token_count

    selected = []
    token_count = 0

    for result in results:
        tokens = len(enc.encode(result.content))
        if token_count + tokens > max_tokens:
            break
        selected.append(result)
        token_count += tokens

    return selected, token_count


def format_memory_context(results: list["SearchResult"]) -> str:
    """Format memories for LLM context injection.

    Creates a structured text block that can be injected into an LLM
    prompt to provide relevant memory context.

    Args:
        results: List of SearchResult objects to format

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant memories found."

    lines = ["=== RELEVANT MEMORIES ===\n"]

    for r in results:
        # Format timestamp
        if r.metadata and r.metadata.last_accessed:
            timestamp = r.metadata.last_accessed.strftime("%Y-%m-%d")
        else:
            timestamp = "unknown"

        # Format confidence
        confidence = f"{r.score:.2f}"

        # Format category if available
        category = r.metadata.category if r.metadata else "general"

        lines.append(f"[{timestamp}] ({category}, confidence: {confidence})")
        lines.append(f"{r.content}\n")

    lines.append("=== END MEMORIES ===")
    return "\n".join(lines)


def deduplicate_results(results: list["SearchResult"]) -> list["SearchResult"]:
    """Remove duplicate results, keeping the highest-scored version.

    Args:
        results: List of SearchResult objects (may contain duplicates)

    Returns:
        Deduplicated list, preserving order of first occurrence
    """
    seen: dict[str, "SearchResult"] = {}

    for result in results:
        if result.memory_id not in seen:
            seen[result.memory_id] = result
        elif result.score > seen[result.memory_id].score:
            # Keep the higher-scored version
            seen[result.memory_id] = result

    # Return in original order (first occurrence)
    unique = []
    seen_ids: set[str] = set()
    for result in results:
        if result.memory_id not in seen_ids:
            unique.append(seen[result.memory_id])
            seen_ids.add(result.memory_id)

    return unique
