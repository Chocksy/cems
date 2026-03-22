"""Agentic search: LLM agents replace embeddings for memory retrieval.

Inspired by Supermemory's ASMR (Agentic Search and Memory Retrieval).
Three parallel search agents (Direct Seeker, Inference Engine, Temporal Navigator)
reason over stored memories to find relevant results.

Usage in the API:
    POST /api/memory/search {"mode": "agentic", "query": "..."}
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from cems.lib.json_parsing import parse_json_list
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # 1M context, 3x cheaper than 2.5, faster
RRF_K = 60

# ---------------------------------------------------------------------------
# Search Agent Prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPTS = {
    "direct_seeker": (
        "You are a Direct Seeker — a memory retrieval specialist focused on EXACT MATCHES.\n\n"
        "Your strategy:\n"
        "- Look for memories that EXPLICITLY mention the specific entities, names, values, "
        "or facts asked about in the question.\n"
        "- Match concrete details: names, numbers, dates, places, products, tools, versions.\n"
        "- Prefer memories with LITERAL matches over memories with vague topical overlap.\n"
        "- Ignore memories that are merely in the same topic area but lack the specific answer.\n\n"
        "You are precision-focused. Return fewer highly-relevant memories rather than many "
        "loosely-related ones."
    ),
    "inference_engine": (
        "You are an Inference Engine — a memory retrieval specialist focused on IMPLICIT CONNECTIONS.\n\n"
        "Your strategy:\n"
        "- Look for memories where the answer can be INFERRED even if not explicitly stated.\n"
        "- Connect dots across related memories.\n"
        "- Consider social and contextual cues: relationships, roles, preferences expressed indirectly.\n"
        "- Look for memories that provide CONTEXT needed to answer the question.\n"
        "- For preference questions, find memories where the user demonstrated a preference "
        "through actions or choices, not just explicit statements.\n\n"
        "You are recall-focused. Cast a wider net to catch memories that a literal matcher would miss."
    ),
    "temporal_navigator": (
        "You are a Temporal Navigator — a memory retrieval specialist focused on TIME and SEQUENCE.\n\n"
        "Your strategy:\n"
        "- Pay close attention to dates and chronological ordering.\n"
        "- For \"most recent\" or \"latest\" questions, find the LAST memory that mentions the topic.\n"
        "- For knowledge-update questions (where info changed), find BOTH old and new, "
        "ranking the most recent first.\n"
        "- Track state changes: if the user switched from X to Y, the memory mentioning Y "
        "is more relevant.\n\n"
        "You are temporally-focused. Your unique value is understanding WHEN things happened."
    ),
}

SEARCH_USER_PROMPT = """I need to find which stored memories are relevant to answering this question:

QUESTION: {question}
{project_context}
Below are {n} stored memories. Each is labeled with its ID (a short hash).

{formatted_memories}

Return a JSON array of memory IDs ranked by relevance to the question, most relevant first.
Return ONLY IDs that are relevant. If none seem relevant, return an empty array.
Return at most 10 IDs.
IMPORTANT: Strongly prefer memories from the current project over memories from other projects.

Example output: ["abc12345", "def67890"]"""


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def _format_memories_for_agents(memories: list[dict]) -> str:
    """Format memory documents for search agent context.

    Args:
        memories: List of memory dicts with keys: id, content, category, source_ref, created_at

    Returns:
        Formatted text block
    """
    parts = []
    for mem in memories:
        mid = mem.get("id", "")[:8]
        content = mem.get("content", "")
        category = mem.get("category", "")
        source_ref = mem.get("source_ref", "")
        created = mem.get("created_at", "")
        if created and hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d")

        parts.append(
            f"--- Memory {mid} (category: {category}, date: {created}, source: {source_ref}) ---\n"
            f"{content}"
        )
    return "\n\n".join(parts)


def _parse_agent_response(response: str, valid_ids: set[str]) -> list[str]:
    """Parse search agent response into a list of memory IDs.

    Tries JSON parsing first, falls back to regex extraction.
    """
    parsed = parse_json_list(response)
    if parsed:
        result = [str(s).strip() for s in parsed if str(s).strip() in valid_ids]
        if result:
            return result[:10]

    # Fallback: regex for hex-like IDs (memory IDs are UUIDs, we use first 8 chars)
    found = re.findall(r'["\']?([a-f0-9]{8})["\']?', response)
    result = [s for s in found if s in valid_ids]
    seen: set[str] = set()
    deduped = []
    for s in result:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:10]


def _run_single_agent(
    role: str,
    question: str,
    memories_text: str,
    n_memories: int,
    model: str,
    project: str | None = None,
) -> tuple[str, str | list[str]]:
    """Run a single search agent. Thread-safe."""
    system = AGENT_SYSTEM_PROMPTS[role]
    project_context = f"\nCURRENT PROJECT: {project}\n" if project else ""
    user_prompt = SEARCH_USER_PROMPT.format(
        question=question,
        n=n_memories,
        formatted_memories=memories_text,
        project_context=project_context,
    )

    client = get_client()

    try:
        response = client.complete(
            prompt=user_prompt,
            system=system,
            model=model,
            temperature=0.1,
            max_tokens=1000,
            fast_route=False,
        )
    except Exception as e:
        logger.warning(f"Agentic search agent {role} failed: {e}")
        return role, []

    if not response:
        return role, []

    return role, response


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> list[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_idx, item_id in enumerate(ranking):
            if item_id not in scores:
                scores[item_id] = 0.0
            scores[item_id] += 1.0 / (k + rank_idx + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# Categories that are always relevant regardless of project
PROFILE_CATEGORIES = {"preferences", "guidelines", "gate-rules", "category-summary"}

# How far back "recent" memories go
RECENT_DAYS = 14

# Max chars to send to agents (leaves headroom in Gemini's 1M context)
MAX_CONTEXT_CHARS = 700_000


def _relevance_score(doc: dict) -> float:
    """Compute a relevance priority score from feedback signals.

    Higher = more relevant. Used to sort memories before filling context budget.
    - relevant_count boosts score
    - noise_count penalizes
    - shown_count is a tiebreaker (more exposure = more signal)
    """
    relevant = doc.get("relevant_count", 0) or 0
    noise = doc.get("noise_count", 0) or 0
    shown = doc.get("shown_count", 0) or 0
    # Net relevance signal, with shown as minor tiebreaker
    return (relevant - noise) + (shown * 0.01)


async def _load_context_memories(
    document_store,
    user_id: str,
    project: str | None = None,
    scope: str | None = None,
) -> list[dict]:
    """Load memories using 3-bucket smart context loading with budget control.

    Bucket 1: PROJECT — all memories with source_ref matching current project
    Bucket 2: PROFILE — preferences, guidelines, gate-rules, category-summary
    Bucket 3: RECENT — last 14 days across all projects, sorted by relevance feedback

    Deduplicates by document ID. Respects MAX_CONTEXT_CHARS budget.

    Args:
        document_store: DocumentStore instance
        user_id: User UUID
        project: Project ID (e.g., "chocksy/cems") for project-scoped filtering
        scope: Optional scope filter

    Returns:
        Deduplicated list of memory dicts, within context budget
    """
    from datetime import UTC, datetime, timedelta

    seen_ids: set[str] = set()
    all_memories: list[dict] = []
    total_chars = 0
    scope_filter = scope if scope != "both" else None

    def _add_unique(docs: list[dict]) -> int:
        nonlocal total_chars
        added = 0
        for doc in docs:
            doc_id = str(doc.get("id", ""))
            if doc_id and doc_id not in seen_ids:
                content_len = len(doc.get("content", ""))
                if total_chars + content_len > MAX_CONTEXT_CHARS:
                    logger.debug(f"Context budget reached at {total_chars} chars, stopping")
                    break
                seen_ids.add(doc_id)
                all_memories.append(doc)
                total_chars += content_len
                added += 1
        return added

    b1 = 0
    # Bucket 1: PROJECT memories — DB-filtered by source_ref, all time
    if project:
        # source_ref is stored as "project:chocksy/cems" format
        source_prefix = f"project:{project}"
        project_docs = await document_store.get_all_documents(
            user_id=user_id,
            scope=scope_filter,
            source_ref_prefix=source_prefix,
            limit=1000,
            order="desc",
        )
        project_docs.sort(key=_relevance_score, reverse=True)
        b1 = _add_unique(project_docs)
        logger.debug(f"Agentic context bucket 1 (project={project}): {b1} memories")

    # Bucket 2: PROFILE memories — no project, always relevant
    for cat in PROFILE_CATEGORIES:
        cat_docs = await document_store.get_all_documents(
            user_id=user_id,
            scope=scope_filter,
            category=cat,
            limit=200,
        )
        _add_unique(cat_docs)
    b2_total = len(all_memories) - b1
    logger.debug(f"Agentic context bucket 2 (profile): {b2_total} memories")

    # Bucket 3: RECENT general memories — same project + no-project only
    # Excludes other-project memories to prevent noise
    recent_docs = await document_store.get_all_documents(
        user_id=user_id,
        scope=scope_filter,
        limit=500,
        order="desc",
    )
    cutoff = datetime.now(UTC) - timedelta(days=RECENT_DAYS)
    recent_filtered = []
    for d in recent_docs:
        created = d.get("created_at")
        if not (created and hasattr(created, "timestamp") and created >= cutoff):
            continue
        # Only include: same project, no project, or profile categories
        src = d.get("source_ref") or ""
        cat = d.get("category") or ""
        if project and src and f"project:{project}" not in src.lower():
            # Different project — skip unless it's a profile category
            if cat not in PROFILE_CATEGORIES:
                continue
        recent_filtered.append(d)

    recent_filtered.sort(key=_relevance_score, reverse=True)
    b3 = _add_unique(recent_filtered)
    logger.debug(f"Agentic context bucket 3 (recent {RECENT_DAYS}d, same project): {b3} memories")

    logger.info(
        f"Agentic context loaded: {len(all_memories)} memories, {total_chars/1000:.0f}K chars "
        f"(project={b1}, profile={b2_total}, recent={b3})"
    )
    return all_memories


async def agentic_search_async(
    document_store,
    user_id: str,
    query: str,
    scope: str = "both",
    limit: int = 10,
    max_tokens: int = 4000,
    model: str = DEFAULT_MODEL,
    project: str | None = None,
) -> dict:
    """Run agentic search using smart 3-bucket context loading.

    Loads memories from 3 sources:
    1. Current project memories (all time)
    2. Profile memories (preferences, guidelines, etc.)
    3. Recent memories (last 14 days, any project)

    Then runs 3 parallel search agents and merges via RRF.

    Args:
        document_store: DocumentStore instance
        user_id: User UUID
        query: Search query
        scope: Memory scope filter
        limit: Max results to return
        max_tokens: Token budget (for compatibility)
        model: LLM model for search agents
        project: Project ID for project-scoped filtering

    Returns:
        Dict matching the standard search response format
    """
    # Smart context loading: 3 buckets
    memories = await _load_context_memories(
        document_store, user_id, project=project, scope=scope,
    )

    if not memories:
        return {
            "results": [],
            "count": 0,
            "mode": "agentic",
            "tokens_used": 0,
            "queries_used": 0,
            "total_candidates": 0,
            "filtered_count": 0,
        }

    # Format all memories for agents
    memories_text = _format_memories_for_agents(memories)
    n_memories = len(memories)

    # Build valid ID set (short IDs for response parsing)
    id_to_full: dict[str, str] = {}
    for mem in memories:
        short_id = str(mem.get("id", ""))[:8]
        id_to_full[short_id] = str(mem.get("id", ""))
    valid_ids = set(id_to_full.keys())

    # Run 3 search agents in parallel using ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    rankings: list[list[str]] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for role in AGENT_SYSTEM_PROMPTS:
            future = loop.run_in_executor(
                pool,
                _run_single_agent,
                role, query, memories_text, n_memories, model, project,
            )
            futures.append(future)

        results = await asyncio.gather(*futures, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Agentic search agent failed: {result}")
                continue
            role, raw_response = result
            if isinstance(raw_response, str):
                parsed = _parse_agent_response(raw_response, valid_ids)
            else:
                parsed = raw_response
            if parsed:
                rankings.append(parsed)

    if not rankings:
        return {
            "results": [],
            "count": 0,
            "mode": "agentic",
            "tokens_used": 0,
            "queries_used": 3,
            "total_candidates": n_memories,
            "filtered_count": 0,
        }

    # RRF merge
    merged_short_ids = reciprocal_rank_fusion(rankings)[:limit]

    # Build result list matching the standard format
    mem_by_id = {}
    for mem in memories:
        full_id = str(mem.get("id", ""))
        short_id = full_id[:8]
        mem_by_id[short_id] = mem

    result_list = []
    for i, short_id in enumerate(merged_short_ids):
        mem = mem_by_id.get(short_id, {})
        # Compute a synthetic score based on RRF position (1.0 → 0.5 range)
        score = 1.0 - (i * 0.5 / max(len(merged_short_ids), 1))

        result_list.append({
            "memory_id": str(mem.get("id", "")),
            "content": mem.get("content", ""),
            "category": mem.get("category", ""),
            "scope": mem.get("scope", "personal"),
            "source_ref": mem.get("source_ref", ""),
            "tags": mem.get("tags", []),
            "score": round(score, 3),
            "created_at": str(mem.get("created_at", "")),
        })

    return {
        "results": result_list,
        "count": len(result_list),
        "mode": "agentic",
        "tokens_used": 0,
        "queries_used": 3,
        "total_candidates": n_memories,
        "filtered_count": len(result_list),
    }
