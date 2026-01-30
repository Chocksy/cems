"""Inference retrieval pipeline for CEMS.

Implements the enhanced retrieval pipeline:
1. Query Understanding - Extract intent, domains, entities (NEW)
2. Query Synthesis - LLM expands query for better retrieval
3. HyDE - Hypothetical Document Embeddings for better vector match (NEW)
4. Candidate Retrieval - Vector search + graph traversal
5. RRF Fusion - Combine results from multiple retrievers (NEW)
6. LLM Re-ranking - Use LLM to re-rank by actual relevance (NEW)
7. Relevance Filtering - Filter candidates below threshold
8. Temporal Ranking - Time decay + priority scoring
9. Token-Budgeted Assembly - Select results within token budget

This module provides the core retrieval logic that powers the unified
memory_search tool.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cems.config import CEMSConfig
    from cems.llm import OpenRouterClient
    from cems.models import MemoryMetadata, SearchResult

logger = logging.getLogger(__name__)

# Default token encoding for context budgeting
DEFAULT_ENCODING = "cl100k_base"


def _is_temporal_query(query: str) -> bool:
    """Detect if query is asking about temporal/chronological information."""
    temporal_patterns = [
        "first", "last", "before", "after", "when",
        "how many days", "how many weeks", "how many months",
        "how long", "earliest", "latest", "most recent",
        "happened first", "happened last", "which came first",
        "start", "began", "ended", "finished",
    ]
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in temporal_patterns)


def synthesize_query(query: str, client: "OpenRouterClient") -> list[str]:
    """Stage 1: Generate better search terms from user query.

    Uses LLM to expand the user's query into multiple search terms
    that might help find relevant memories. Has special handling for
    temporal queries to improve chronological reasoning.

    Args:
        query: Original user query
        client: OpenRouter client for LLM calls

    Returns:
        List of expanded search terms (original not included)
    """
    # Detect temporal queries for specialized expansion
    is_temporal = _is_temporal_query(query)

    if is_temporal:
        prompt = f"""Generate 3-4 search queries to find memories about temporal/chronological events.

User query: {query}

CRITICAL RULES for TEMPORAL queries:
- Focus on finding events, dates, sequences, and timelines
- Include date-related terms (when, date, time, started, ended)
- For "which first/last" questions, search for BOTH events mentioned
- Include terms that capture chronological ordering
- Stay within the SAME specific events/topics mentioned

Return one search term per line. No bullets, no numbering."""
    else:
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
        # Filter out empty or very short terms, limit to 4 expansions for temporal
        max_terms = 4 if is_temporal else 3
        return [t for t in terms if len(t) > 2][:max_terms]
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


# =============================================================================
# NEW: Unified Scoring Function (consolidates duplicate logic)
# =============================================================================


def apply_score_adjustments(
    result: "SearchResult",
    inferred_category: str | None = None,
    project: str | None = None,
) -> float:
    """Apply all score adjustments in a single, consolidated function.

    This is the SINGLE SOURCE OF TRUTH for scoring logic.
    Replaces duplicate scoring in search() and retrieve_for_inference().

    Args:
        result: SearchResult to score
        inferred_category: Optional category inferred from query
        project: Optional project ID for project-scoped scoring

    Returns:
        Adjusted score
    """
    score = result.score  # Start with base vector similarity

    if result.metadata:
        # Priority boost (1.0 default, up to 2.0 for hot memories)
        score *= result.metadata.priority

        # Time decay: 50% penalty per 2 months since last access (was 1 month)
        # Slower decay helps with temporal reasoning queries over longer periods
        now = datetime.now(UTC)
        days_since_access = (now - result.metadata.last_accessed).days
        time_decay = 1.0 / (1.0 + (days_since_access / 60))  # 60-day half-life
        score *= time_decay

        # Pinned boost (10%)
        if result.metadata.pinned:
            score *= 1.1

        # REMOVED: Cross-category penalty (was hurting recall)
        # REMOVED: Project penalty (was hurting eval recall)

    return score


# =============================================================================
# NEW: HyDE (Hypothetical Document Embeddings)
# =============================================================================


def generate_hypothetical_memory(query: str, client: "OpenRouterClient") -> str:
    """Generate a hypothetical memory that would answer this query.

    HyDE technique: Instead of searching with the query directly,
    generate what an ideal answer would look like, then search for
    documents similar to that answer.

    Has special handling for temporal queries to generate timeline-aware content.

    Args:
        query: User's search query
        client: OpenRouter client for LLM calls

    Returns:
        Hypothetical memory content (2-3 sentences)
    """
    is_temporal = _is_temporal_query(query)

    if is_temporal:
        prompt = f"""You are a memory retrieval system. Given this TEMPORAL query, generate a
hypothetical memory entry (2-3 sentences) that would perfectly answer it.

Query: {query}

IMPORTANT for temporal queries:
- Include specific dates or time references (e.g., "On March 15th...", "Last Tuesday...")
- Mention the sequence of events if comparing ("First X happened, then Y")
- Include duration or time differences if relevant ("3 days before...", "2 weeks after...")
- Be specific about WHEN things happened

Hypothetical memory:"""
    else:
        prompt = f"""You are a memory retrieval system. Given this query, generate a
hypothetical memory entry (2-3 sentences) that would perfectly answer it.

Query: {query}

Write the memory AS IF it was stored previously by a developer. Be specific and concrete.
Include relevant technical details, file paths, commands, or preferences that would help.

Hypothetical memory:"""

    try:
        result = client.complete(prompt, max_tokens=150, temperature=0.3)
        return result.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return ""


# =============================================================================
# NEW: Reciprocal Rank Fusion (RRF)
# =============================================================================


def reciprocal_rank_fusion(
    result_lists: list[list["SearchResult"]],
    k: int = 60,
    rrf_weight: float = 0.5,  # Increased from 0.3 for better multi-query fusion
) -> list["SearchResult"]:
    """Combine results from multiple retrievers using RRF.

    RRF Formula: score = sum(1 / (k + rank_i)) for each retriever i

    The RRF scores are normalized to 0-1 range before blending with
    original vector scores to preserve meaningful score magnitudes.

    Args:
        result_lists: List of result lists from different retrievers/queries
        k: Ranking constant (default 60, standard in literature)
        rrf_weight: Weight for normalized RRF score in final blend (default 0.5)
                    Final = rrf_weight * norm_rrf + (1-rrf_weight) * vector_score

    Returns:
        Fused and re-ranked list of SearchResults
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, "SearchResult"] = {}

    for results in result_lists:
        for rank, result in enumerate(results, 1):
            # Accumulate RRF score
            rrf_scores[result.memory_id] += 1.0 / (k + rank)
            # Keep the highest-scoring version of each result
            if result.memory_id not in result_map or result.score > result_map[result.memory_id].score:
                result_map[result.memory_id] = result

    if not rrf_scores:
        return []

    # Normalize RRF scores to 0-1 range
    min_rrf = min(rrf_scores.values())
    max_rrf = max(rrf_scores.values())
    rrf_range = max_rrf - min_rrf

    # Update scores with normalized RRF scores and sort
    fused = []
    for memory_id, rrf_score in rrf_scores.items():
        result = result_map[memory_id]
        
        # Normalize RRF score to 0-1 range
        if rrf_range > 0:
            norm_rrf = (rrf_score - min_rrf) / rrf_range
        else:
            norm_rrf = 1.0  # All same score = all top rank
        
        # Blend: 50% normalized RRF (for ranking fusion), 50% original vector score
        # Equal weighting gives multi-query fusion more influence on final ranking
        result.score = rrf_weight * norm_rrf + (1 - rrf_weight) * result.score
        fused.append(result)

    return sorted(fused, key=lambda x: x.score, reverse=True)


# =============================================================================
# NEW: LLM Re-ranking
# =============================================================================


def rerank_with_llm(
    query: str,
    candidates: list["SearchResult"],
    client: "OpenRouterClient",
    top_k: int = 10,
    config: "CEMSConfig | None" = None,
) -> list["SearchResult"]:
    """Use LLM to re-rank candidates by actual relevance.

    This is the key to smarter retrieval - the LLM evaluates whether
    each candidate ACTUALLY answers the query, not just similarity.

    Args:
        query: Original user query
        candidates: List of candidates to re-rank
        client: OpenRouter client
        top_k: Number of results to return
        config: Optional CEMS config for limits

    Returns:
        Re-ranked list of SearchResults
    """
    if not candidates:
        return []

    # Limit input for cost/speed (configurable)
    rerank_input_limit = config.rerank_input_limit if config else 40
    candidates_to_rank = candidates[:rerank_input_limit]

    # Format candidates for LLM
    candidate_text = "\n".join([
        f"{i+1}. [{c.metadata.category if c.metadata else 'general'}] {c.content[:200]}..."
        for i, c in enumerate(candidates_to_rank)
    ])

    prompt = f"""Given this search query, rank these memory candidates by ACTUAL RELEVANCE.

Query: {query}

Candidates:
{candidate_text}

IMPORTANT: Only include memories that are ACTUALLY relevant to the query.
A memory about SSH to Hetzner is NOT relevant to a query about Windows printers.
A memory about SEO scripts is NOT relevant to a query about fiscal printers.

Return a JSON array of indices (1-based) in relevance order.
Only include indices of memories that are TRULY relevant.
If nothing is relevant, return an empty array [].

Example: [3, 1, 7] means candidate 3 is most relevant, then 1, then 7.

JSON array:"""

    try:
        response = client.complete(prompt, max_tokens=100, temperature=0.1)
        
        # Parse JSON response
        response = response.strip()
        if response.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()
        
        indices = json.loads(response)
        
        if not isinstance(indices, list):
            logger.warning(f"LLM rerank returned non-list: {response}")
            return candidates[:top_k]

        # Reorder candidates based on LLM ranking
        reranked = []
        for rank, idx in enumerate(indices[:top_k]):
            if isinstance(idx, int) and 1 <= idx <= len(candidates_to_rank):
                result = candidates_to_rank[idx - 1]
                # Assign new score based on LLM rank
                llm_score = 1.0 / (1 + rank)
                # Blend: 70% LLM rank, 30% original score
                result.score = 0.7 * llm_score + 0.3 * result.score
                reranked.append(result)

        logger.debug(f"LLM reranking: {len(candidates_to_rank)} -> {len(reranked)} results")
        return reranked

    except Exception as e:
        logger.warning(f"LLM reranking failed: {e}")
        return candidates[:top_k]


# =============================================================================
# NEW: Query Understanding
# =============================================================================


def extract_query_intent(query: str, client: "OpenRouterClient") -> dict[str, Any]:
    """Extract semantic intent from query for smarter retrieval routing.

    Analyzes the query to determine:
    - Primary intent (troubleshooting, how-to, factual, etc.)
    - Complexity (simple/moderate/complex)
    - Domains (technical areas the query touches)
    - Key entities (specific things mentioned)

    Args:
        query: User's search query
        client: OpenRouter client

    Returns:
        Dict with intent, complexity, domains, entities, requires_reasoning
    """
    prompt = f"""Analyze this memory search query and extract its intent.

Query: {query}

Return JSON with:
{{
  "primary_intent": "<troubleshooting|how-to|factual|recall|preference>",
  "complexity": "<simple|moderate|complex>",
  "domains": ["<domain1>", "<domain2>"],
  "entities": ["<entity1>", "<entity2>"],
  "requires_reasoning": <true|false>
}}

Examples:
- "What's my Python version preference?" -> {{"primary_intent": "preference", "complexity": "simple", "domains": ["python"], "entities": ["python"], "requires_reasoning": false}}
- "How do I connect datecs printer to Windows remotely?" -> {{"primary_intent": "how-to", "complexity": "complex", "domains": ["printers", "windows", "remote-access"], "entities": ["datecs", "windows"], "requires_reasoning": true}}

JSON:"""

    try:
        response = client.complete(prompt, max_tokens=200, temperature=0.1)
        
        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()
        
        intent = json.loads(response)
        logger.debug(f"Query intent: {intent}")
        return intent

    except Exception as e:
        logger.warning(f"Query intent extraction failed: {e}")
        return {
            "primary_intent": "factual",
            "complexity": "moderate",
            "domains": [],
            "entities": [],
            "requires_reasoning": False,
        }


def route_to_strategy(intent: dict[str, Any]) -> str:
    """Select retrieval strategy based on query analysis.

    Args:
        intent: Query intent dict from extract_query_intent()

    Returns:
        Strategy name: "vector", "hybrid", or "tree"
    """
    complexity = intent.get("complexity", "moderate")
    requires_reasoning = intent.get("requires_reasoning", False)
    domains = intent.get("domains", [])

    # Simple queries with high confidence -> fast vector path
    if complexity == "simple" and not requires_reasoning:
        return "vector"

    # Complex queries or multi-domain -> full hybrid with reranking
    if complexity == "complex" or requires_reasoning or len(domains) > 2:
        return "hybrid"  # Uses HyDE + RRF + reranking

    # Default: hybrid without full reranking
    return "hybrid"
