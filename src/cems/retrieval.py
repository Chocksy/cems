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

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cems.lib.json_parsing import extract_json_from_response, parse_json_dict, parse_json_list

if TYPE_CHECKING:
    from cems.config import CEMSConfig
    from cems.llm import OpenRouterClient
    from cems.models import MemoryMetadata, SearchResult

logger = logging.getLogger(__name__)

# Default token encoding for context budgeting
DEFAULT_ENCODING = "cl100k_base"


def _run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version instead."
        )
    return asyncio.run(coro)


def normalize_lexical_score(score: float) -> float:
    """Normalize lexical scores to a 0-1 range.

    Uses a saturating transform so higher scores approach 1.0 without
    dominating other signals.
    """
    if score <= 0:
        return 0.0
    return score / (1.0 + score)


def is_strong_lexical_signal(
    top_score: float,
    second_score: float,
    threshold: float,
    gap: float,
) -> bool:
    """Decide whether lexical signal is strong enough to skip expansion."""
    top_norm = normalize_lexical_score(top_score)
    second_norm = normalize_lexical_score(second_score)
    return top_norm >= threshold and (top_norm - second_norm) >= gap


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


def _is_aggregation_query(query: str) -> bool:
    """Detect if query requires aggregating information from MULTIPLE sessions.

    These queries need special handling:
    - Larger candidate pool to find all relevant memories
    - Diversity-aware selection (not just top-scoring)
    - Query synthesis to generate variations that find different sessions

    Examples:
    - "How many different doctors did I visit?"
    - "What is the total amount I spent on luxury items?"
    - "How many camping trips did I take in total?"
    """
    aggregation_patterns = [
        # Counting patterns
        "how many", "how much",
        "number of", "count of",
        # Totaling patterns
        "total", "altogether", "in total", "combined",
        "sum of", "all the",
        # Frequency/repetition patterns
        "all the times", "every time", "each time",
        "how often", "how frequently",
        # Variety/diversity patterns
        "different", "various", "all the different",
        "how many different",
        # Aggregation across time
        "throughout", "across all", "over the past",
        "in the past", "in the last",
    ]
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in aggregation_patterns)


def _is_preference_query(query: str) -> bool:
    """Detect if query is asking for recommendations based on user preferences.

    Preference queries have a semantic gap problem: the question uses
    interrogative/generic phrasing while answers contain declarative/specific
    statements. E.g., "video editing resources?" vs "I use Adobe Premiere Pro".

    These queries benefit from query synthesis to bridge the gap.
    """
    preference_signals = [
        # Recommendation/suggestion phrases
        "recommend", "suggest", "suggestion", "advice",
        "any recommendations", "any suggestions",  # v3: explicit request forms
        "what would you", "what should", "what could",
        "would you", "could you", "can you recommend",
        # Resource/tool seeking
        "resources", "tools", "accessories", "equipment",
        "publications", "books", "articles", "tutorials",
        # Complement/match seeking
        "complement", "go with", "pair with", "match with",
        "compatible", "work with", "works well",
        # Interest/preference probing
        "might like", "might enjoy", "might find interesting",
        "based on my", "given my", "considering my",
        # Setup/workflow related
        "setup", "workflow", "routine", "practice",
        # Indirect preference patterns (v2 - catch more failing queries)
        "any tips", "tips for", "any ideas",  # advice-seeking
        "i've been feeling", "been feeling",  # emotional state → advice
        "i've been struggling", "been struggling",  # struggle → advice
        "i've been thinking about", "thinking about making",  # intent/planning
        "i'm thinking of", "thinking of inviting", "thinking of trying",  # v3: planning → recommendations
        "i was thinking of", "was thinking of trying",  # v3: past tense planning
        "what should i serve", "should i serve",  # meal planning
        "i'm getting excited about",  # anticipation → preferences
        "activities that i can do",  # activity seeking
        "show or movie", "movie for me",  # entertainment recommendations
        # v3: Planning patterns that expect preference-based answers
        "i'm planning", "i am planning", "planning a trip", "planning my",
        "any suggestions on what to",  # planning + suggestion
        "do you think", "what do you think",  # opinion/preference seeking
        # v3: Observation + question patterns (implicit preference queries)
        "i noticed", "i've noticed",  # observation → explanation based on preferences
        "i've got some free time", "got some free time",  # entertainment seeking
        "could there be a reason",  # explanation based on past context
    ]
    query_lower = query.lower()
    return any(signal in query_lower for signal in preference_signals)


def synthesize_query(
    query: str,
    client: "OpenRouterClient",
    is_preference: bool = False,
    profile_context: list[str] | None = None,
) -> list[str]:
    """Stage 1: Generate better search terms from user query.

    Uses LLM to expand the user's query into multiple search terms
    that might help find relevant memories. Has special handling for
    temporal queries and preference/recommendation queries.

    Args:
        query: Original user query
        client: OpenRouter client for LLM calls
        is_preference: Whether this is a preference/recommendation query
        profile_context: Dynamic examples from user's actual memories (for RAP)

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
    elif is_preference:
        # Preference queries need expansion to bridge semantic gap between
        # question phrasing ("recommend video editing resources") and
        # answer phrasing ("I use Adobe Premiere Pro")

        # RAP: Use dynamic examples from user's actual memories instead of hardcoded domains
        dynamic_examples = ""
        if profile_context:
            examples_list = "\n".join([f'- "{ctx}"' for ctx in profile_context[:5]])
            dynamic_examples = f"""
Here are ACTUAL things this user has mentioned in their memories:
{examples_list}

Use these as inspiration for the kinds of specific terms to search for.
"""

        prompt = f"""Generate 4-5 search queries to find memories about user preferences, tools, and interests.

User query: {query}
{dynamic_examples}
CRITICAL RULES for PREFERENCE/RECOMMENDATION queries:
- Generate queries that would match USER STATEMENTS about their tools/products/styles
- Include SPECIFIC product names, brands, software, styles the user might mention
- Include phrases like "I use", "I prefer", "my favorite", "I work with", "I'm into"
- Think about what DECLARATIVE statements would answer this question
- Bridge the gap: generic question → specific user statements

The goal is to find memories where the user talked about their preferences, not generic information.

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
        start = time.perf_counter()
        result = client.complete(prompt, temperature=0.3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[TIMING] synthesize_query LLM call: {elapsed_ms:.0f}ms")

        terms = [q.strip() for q in result.strip().split("\n") if q.strip()]
        # Filter out empty or very short terms
        # More expansions for preference queries (need to cover diverse domains)
        max_terms = 5 if is_preference else (4 if is_temporal else 3)
        return [t for t in terms if len(t) > 2][:max_terms]
    except Exception as e:
        logger.warning(f"[RETRIEVAL] Query synthesis failed: {e}")
        return []


def _word_set(text: str) -> set[str]:
    """Extract word set from text for similarity computation."""
    import re
    # Simple word extraction: lowercase alphanumeric sequences
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _max_similarity_to_selected(
    candidate_words: set[str],
    selected_word_sets: list[set[str]],
) -> float:
    """Compute max similarity between candidate and any selected document."""
    if not selected_word_sets:
        return 0.0
    return max(_jaccard_similarity(candidate_words, s) for s in selected_word_sets)


def assemble_context_diverse(
    results: list["SearchResult"],
    max_tokens: int = 2000,
    mmr_lambda: float = 0.6,
) -> tuple[list["SearchResult"], int]:
    """Session-aware assembly with MMR diversity for aggregation queries.

    Uses Maximal Marginal Relevance (MMR) to balance relevance and diversity:
    MMR = λ * relevance - (1-λ) * max_similarity_to_selected

    This helps multi-session queries by penalizing documents too similar
    to already-selected ones, spreading results across different sessions.

    Args:
        results: List of SearchResult objects, sorted by score
        max_tokens: Maximum tokens to include
        mmr_lambda: Relevance/diversity tradeoff (0.6 = 60% relevance, 40% diversity)

    Returns:
        Tuple of (selected results, total tokens used)
    """
    if not results:
        return [], 0

    try:
        import tiktoken
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)

        def count_tokens(text: str) -> int:
            return len(enc.encode(text))
    except ImportError:
        def count_tokens(text: str) -> int:
            return len(text) // 4

    # Group by source_ref (session identifier)
    from collections import defaultdict
    sessions: dict[str, list["SearchResult"]] = defaultdict(list)
    for result in results:
        source_ref = result.metadata.source_ref if result.metadata else "unknown"
        sessions[source_ref].append(result)

    # Defensive: ensure per-session ordering by score
    for session_results in sessions.values():
        session_results.sort(key=lambda x: x.score, reverse=True)

    # Cache token counts and word sets
    token_cache: dict[str, int] = {}
    word_cache: dict[str, set[str]] = {}

    def tokens_for(result: "SearchResult") -> int:
        cached = token_cache.get(result.memory_id)
        if cached is not None:
            return cached
        tokens = count_tokens(result.content)
        token_cache[result.memory_id] = tokens
        return tokens

    def words_for(result: "SearchResult") -> set[str]:
        cached = word_cache.get(result.memory_id)
        if cached is not None:
            return cached
        words = _word_set(result.content)
        word_cache[result.memory_id] = words
        return words

    selected: list["SearchResult"] = []
    selected_word_sets: list[set[str]] = []
    token_count = 0
    used_ids: set[str] = set()

    # Normalize scores to [0, 1] for MMR calculation
    max_score = max(r.score for r in results) if results else 1.0
    min_score = min(r.score for r in results) if results else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    def normalize_score(score: float) -> float:
        return (score - min_score) / score_range

    # Phase 1: Take top result from each session using MMR
    # Sort sessions by their top score for initial ordering
    session_order = []
    for source_ref, session_results in sessions.items():
        if not session_results:
            continue
        top_result = session_results[0]
        session_order.append((source_ref, top_result.score, tokens_for(top_result)))

    session_order.sort(key=lambda x: (-x[1], x[2]))

    for source_ref, _score, _tokens in session_order:
        session_results = sessions[source_ref]
        # Pick the best result from this session using MMR
        best_mmr = float('-inf')
        best_candidate = None

        for candidate in session_results:
            if candidate.memory_id in used_ids:
                continue
            tokens = tokens_for(candidate)
            if token_count + tokens > max_tokens:
                continue

            # Compute MMR score
            relevance = normalize_score(candidate.score)
            diversity_penalty = _max_similarity_to_selected(
                words_for(candidate), selected_word_sets
            )
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity_penalty

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            selected_word_sets.append(words_for(best_candidate))
            token_count += tokens_for(best_candidate)
            used_ids.add(best_candidate.memory_id)

    # Phase 2: Fill remaining budget using MMR selection
    remaining = [r for r in results if r.memory_id not in used_ids]

    while remaining and token_count < max_tokens:
        best_mmr = float('-inf')
        best_candidate = None
        best_idx = -1

        for idx, candidate in enumerate(remaining):
            tokens = tokens_for(candidate)
            if token_count + tokens > max_tokens:
                continue

            # Compute MMR score
            relevance = normalize_score(candidate.score)
            diversity_penalty = _max_similarity_to_selected(
                words_for(candidate), selected_word_sets
            )
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity_penalty

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = candidate
                best_idx = idx

        if best_candidate:
            selected.append(best_candidate)
            selected_word_sets.append(words_for(best_candidate))
            token_count += tokens_for(best_candidate)
            used_ids.add(best_candidate.memory_id)
            remaining.pop(best_idx)
        else:
            # No candidate fits the budget
            break

    # Re-sort by score for consistent output
    selected.sort(key=lambda x: x.score, reverse=True)

    logger.info(
        f"[RETRIEVAL] MMR assembly: {len(sessions)} sessions -> {len(selected)} results "
        f"(λ={mmr_lambda})"
    )
    return selected, token_count


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
                # BUG FIX: Don't break - try smaller subsequent results
                # This is critical for multi-session "all" recall where we need
                # to include results from different sessions, even if earlier
                # larger results don't fit.
                continue
            selected.append(result)
            token_count += tokens

        return selected, token_count

    selected = []
    token_count = 0

    for result in results:
        tokens = len(enc.encode(result.content))
        if token_count + tokens > max_tokens:
            # BUG FIX: Don't break - try smaller subsequent results
            # This allows fitting more diverse results from different sessions
            continue
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
    project: str | None = None,
    config: "CEMSConfig | None" = None,
    # Deprecated — kept for call-site compatibility, ignored
    inferred_category: str | None = None,
    skip_category_penalty: bool = False,
) -> float:
    """Apply all score adjustments in a single, consolidated function.

    This is the SINGLE SOURCE OF TRUTH for scoring logic.
    Replaces duplicate scoring in search() and retrieve_for_inference().

    Scoring factors applied (in order):
    1. Priority boost (1.0-2.0x from metadata)
    2. Time decay (60-day half-life)
    3. Pinned boost (1.1x)
    4. Project scoring (1.3x same-project, 0.8x different-project, 0.9x no-project-tag)

    Args:
        result: SearchResult to score
        project: Optional project ID for project-scoped scoring
        config: Optional config for penalty settings (defaults to penalties enabled)

    Returns:
        Adjusted score
    """
    score = result.score  # Start with base vector similarity

    if result.metadata:
        # Priority boost (1.0 default, up to 2.0 for hot memories)
        score *= result.metadata.priority

        # Time decay: 50% penalty per 2 months since last access
        # Slower decay helps with temporal reasoning queries over longer periods
        now = datetime.now(UTC)
        days_since_access = (now - result.metadata.last_accessed).days
        time_decay = 1.0 / (1.0 + (days_since_access / 60))  # 60-day half-life
        score *= time_decay

        # Pinned boost (10%)
        if result.metadata.pinned:
            score *= 1.1

        # Project-scoped scoring (boost same-project, penalize different-project)
        enable_project_penalty = config.enable_project_penalty if config else True
        if enable_project_penalty and project:
            source_ref = result.metadata.source_ref or ""
            if source_ref.startswith(f"project:{project}"):
                # Same project: boost
                boost = config.project_boost_factor if config else 1.3
                score *= boost
            elif source_ref.startswith("project:"):
                # Different project: penalty
                penalty = config.project_penalty_factor if config else 0.8
                score *= penalty
            else:
                # No project tag: mild penalty when project filter is active
                score *= 0.9

    # Floor at 0.0 (no upper clamp — scores >1.0 are valid for ranking)
    score = max(0.0, score)

    return score


# =============================================================================
# NEW: HyDE (Hypothetical Document Embeddings)
# =============================================================================


def generate_adaptive_probe(query: str, client: "OpenRouterClient") -> list[str]:
    """Generate preference probe phrases tailored to the query domain.

    Uses LLM to identify domain-specific terms that would match user
    preference statements. Only called when fixed probe underperforms.

    Args:
        query: User's search query
        client: OpenRouter client for LLM calls

    Returns:
        List of domain-specific search phrases (up to 5)
    """
    prompt = f"""Given the user question, produce 5-7 short search phrases to find their PREFERENCES in that domain.

Rules:
- Use only terms directly implied by the question's domain
- Include phrases that resemble memory statements: "I use X", "I prefer Y", "my favorite Z"
- Include domain-specific products, brands, tools, styles
- NO generic terms like "preferences" or "favorites" - be SPECIFIC to the domain

Question: {query}

Return JSON: {{"phrases":["phrase1","phrase2",...]}}"""

    try:
        start = time.perf_counter()
        result = client.complete(prompt, temperature=0.3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[TIMING] generate_adaptive_probe LLM call: {elapsed_ms:.0f}ms")

        # Parse JSON response
        parsed = parse_json_dict(result, fallback={"phrases": []})
        phrases = parsed.get("phrases", [])

        # Filter and limit
        valid_phrases = [p.strip() for p in phrases if isinstance(p, str) and len(p.strip()) > 2]
        logger.info(f"[RETRIEVAL] Adaptive probe generated {len(valid_phrases)} phrases: {valid_phrases[:3]}...")
        return valid_phrases[:5]
    except Exception as e:
        logger.warning(f"[RETRIEVAL] Adaptive probe failed: {e}")
        return []


def extract_profile_context(contents: list[str], max_phrases: int = 5) -> list[str]:
    """Extract key preference phrases from memory contents.

    Uses GENERALIZABLE patterns that work for ANY domain, not hardcoded
    domain-specific terms. Looks for first-person preference statements.

    Args:
        contents: List of memory content strings
        max_phrases: Maximum phrases to return

    Returns:
        List of extracted preference phrases
    """
    import re

    phrases = []
    # GENERALIZABLE patterns - work for any domain (code, cooking, music, etc.)
    # No hardcoded domain-specific terms (gin, mid-century, power bank, etc.)
    preference_patterns = [
        r"I (?:use|prefer|really like|love|enjoy|work with) ([^.!?\n]{5,60})",
        r"my favorite (?:is |are )?([^.!?\n]{5,60})",
        r"I (?:recently|just) (?:bought|purchased|got|started using) ([^.!?\n]{5,60})",
        r"I took a ([^.!?\n]{5,60}class[^.!?\n]{0,30})",
        r"I'm (?:really )?into ([^.!?\n]{5,60})",
        r"I've been (?:using|learning|practicing|experimenting with) ([^.!?\n]{5,60})",
        r"I (?:switched to|moved to|started with) ([^.!?\n]{5,60})",
        r"I'm a (?:big )?fan of ([^.!?\n]{5,60})",
    ]

    for content in contents:
        for pattern in preference_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                phrase = match.strip()
                if len(phrase) > 3 and phrase not in phrases:
                    phrases.append(phrase)
                    if len(phrases) >= max_phrases:
                        return phrases

    return phrases


def filter_preferences_by_relevance(
    preferences: list[str],
    query: str,
    client: "OpenRouterClient",
) -> list[str]:
    """Filter preferences to only those RELEVANT to the query domain.

    Uses LLM to judge relevance rather than simple pattern matching.
    This prevents irrelevant preferences (e.g., "Instagram sticker" for
    a cocktail query) from polluting the HyDE prompt.

    NOTE: Errs on the side of INCLUSION - if in doubt, keep the preference.
    Empty filtering is worse than keeping a slightly off-topic preference.

    Args:
        preferences: List of extracted preference phrases
        query: User's search query
        client: OpenRouter client for LLM calls

    Returns:
        List of preferences relevant to the query (may be empty)
    """
    if not preferences:
        return []

    # Format preferences for the prompt
    pref_list = "\n".join([f"- {p}" for p in preferences])

    prompt = f"""Given this user question and list of their stated preferences, select the preferences that are POTENTIALLY RELEVANT to answering the question.

Question: {query}

User's stated preferences:
{pref_list}

RULES:
- Include preferences that MIGHT help answer the question
- Be GENEROUS - if there's any connection, INCLUDE IT
- "recommend a cocktail" → keep drink/alcohol/mixology, AND related lifestyle preferences
- "video editing resources" → keep video/editing/software, AND creative tool preferences
- Only EXCLUDE preferences that are COMPLETELY UNRELATED (different domain entirely)
- When in doubt, INCLUDE the preference

Return JSON: {{"relevant": ["preference1", "preference2", ...]}}
If ALL preferences could be relevant, return ALL of them."""

    try:
        start = time.perf_counter()
        result = client.complete(prompt, temperature=0.1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[TIMING] filter_preferences_by_relevance LLM call: {elapsed_ms:.0f}ms")

        parsed = parse_json_dict(result, fallback={"relevant": []})
        relevant = parsed.get("relevant", [])

        # Validate that returned items are from the original list
        valid_relevant = [p for p in relevant if p in preferences]

        # LENIENT FALLBACK: If filter is too aggressive, keep original
        # This prevents over-filtering which hurt v1
        if len(valid_relevant) == 0 and len(preferences) > 0:
            logger.info(
                f"[RETRIEVAL] Relevance filter returned empty, keeping original {len(preferences)} preferences"
            )
            return preferences

        logger.info(
            f"[RETRIEVAL] Relevance filter: {len(preferences)} → {len(valid_relevant)} preferences"
        )
        return valid_relevant
    except Exception as e:
        logger.warning(f"[RETRIEVAL] Relevance filter failed: {e}, keeping all")
        return preferences  # Fallback: keep all on error


def generate_hypothetical_memory(
    query: str,
    client: "OpenRouterClient",
    is_preference: bool = False,
    profile_context: list[str] | None = None,
) -> str:
    """Generate a hypothetical memory that would answer this query.

    HyDE technique: Instead of searching with the query directly,
    generate what an ideal answer would look like, then search for
    documents similar to that answer.

    Has special handling for:
    - Temporal queries: generate timeline-aware content with dates
    - Preference queries: generate user statements about their tools/interests

    Args:
        query: User's search query
        client: OpenRouter client for LLM calls
        is_preference: Whether this is a preference/recommendation query

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
    elif is_preference:
        # Preference queries: generate USER STATEMENTS that would answer the question
        # This bridges the semantic gap between "recommend X" and "I use Y"

        # Include profile context if available from profile probe
        profile_section = ""
        if profile_context:
            profile_section = f"""
KNOWN USER PREFERENCES (from their memories):
{chr(10).join(f'- {p}' for p in profile_context)}

Use these known preferences to generate a MORE SPECIFIC hypothetical memory.
"""

        prompt = f"""You are a memory retrieval system. Given this PREFERENCE query, generate a
hypothetical memory entry (3-4 sentences) written FROM THE USER'S PERSPECTIVE.

Query: {query}
{profile_section}
CRITICAL for preference queries - write as if the USER said this previously:
- Use first-person: "I use...", "I prefer...", "I really enjoy...", "My favorite..."
- Mention SPECIFIC products, brands, tools, styles, or experiences by NAME
- Include context about WHY they like it, their skill level, or relevant constraints
- For recommendations: imagine what SPECIFIC thing they might already use/like
- Be concrete and domain-specific, not generic
- If user preferences are provided above, use them to guide your response

Hypothetical memory (from USER's perspective, be SPECIFIC about domain/style/preferences):"""
    else:
        prompt = f"""You are a memory retrieval system. Given this query, generate a
hypothetical memory entry (2-3 sentences) that would perfectly answer it.

Query: {query}

Write the memory AS IF it was stored previously by a developer. Be specific and concrete.
Include relevant technical details, file paths, commands, or preferences that would help.

Hypothetical memory:"""

    start = time.perf_counter()
    result = client.complete(prompt, temperature=0.3)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[TIMING] generate_hypothetical_memory (HyDE) LLM call: {elapsed_ms:.0f}ms")
    return result.strip()


# =============================================================================
# NEW: Reciprocal Rank Fusion (RRF)
# =============================================================================


def reciprocal_rank_fusion(
    result_lists: list[list["SearchResult"]],
    list_weights: list[float] | None = None,
    top_rank_bonus: tuple[float, float] = (0.05, 0.02),
    k: int = 60,
    rrf_weight: float = 0.5,
) -> list["SearchResult"]:
    """Combine results from multiple retrievers using RRF with QMD-style guardrails.

    RRF Formula: score = sum(weight_i / (k + rank_i)) for each retriever i

    QMD Enhancements:
    - List weights: Original query gets 2x weight vs expansions (protects precision)
    - Top-rank bonus: Rank 1 gets +0.05, ranks 2-3 get +0.02 (protects top hits)

    The RRF scores are normalized to 0-1 range before blending with
    original vector scores to preserve meaningful score magnitudes.

    Args:
        result_lists: List of result lists from different retrievers/queries
        list_weights: Weight for each result list (default: 1.0 for all)
        top_rank_bonus: (rank1_bonus, rank2_3_bonus) added per list
        k: Ranking constant (default 60, standard in literature)
        rrf_weight: Weight for normalized RRF score in final blend (default 0.5)
                    Final = rrf_weight * norm_rrf + (1-rrf_weight) * vector_score

    Returns:
        Fused and re-ranked list of SearchResults
    """
    if list_weights is None:
        list_weights = [1.0] * len(result_lists)

    bonus_r1, bonus_r23 = top_rank_bonus
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, "SearchResult"] = {}

    for weight, results in zip(list_weights, result_lists):
        for rank, result in enumerate(results, 1):
            # RRF score with list weight
            base = weight / (k + rank)
            # Top-rank bonus (per-list, stacks if item appears in multiple lists)
            bonus = bonus_r1 if rank == 1 else (bonus_r23 if rank <= 3 else 0.0)
            rrf_scores[result.memory_id] += base + bonus

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

    start = time.perf_counter()
    response = client.complete(prompt, temperature=0.1)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[TIMING] rerank_with_llm LLM call: {elapsed_ms:.0f}ms")

    # Parse JSON response using shared utility
    indices = parse_json_list(response, fallback=None)

    if indices is None:
        logger.warning(f"LLM rerank JSON parse failed: {response[:200]}")
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

    # Fallback if LLM returned empty or invalid indices
    if not reranked:
        logger.warning(f"LLM rerank produced no valid results, falling back to original order")
        return candidates[:top_k]

    logger.info(f"[TIMING] rerank_with_llm total: {len(candidates_to_rank)} -> {len(reranked)} results")
    return reranked


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

    start = time.perf_counter()
    response = client.complete(prompt, temperature=0.1)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[TIMING] extract_query_intent LLM call: {elapsed_ms:.0f}ms")

    # Parse JSON using shared utility
    default_intent = {
        "primary_intent": "factual",
        "complexity": "moderate",
        "domains": [],
        "entities": [],
        "requires_reasoning": False,
    }
    intent = parse_json_dict(response, fallback=default_intent)
    logger.info(f"[TIMING] extract_query_intent result: {intent}")
    return intent


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
