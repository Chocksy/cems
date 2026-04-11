"""Agentic search: LLM agents replace embeddings for memory retrieval.

Inspired by Supermemory's ASMR (Agentic Search and Memory Retrieval).
Four parallel search agents:
  - Entity Picker: searches entity page summaries (~100 docs, ~50K chars)
  - Direct Seeker, Inference Engine, Temporal Navigator: search raw memories

Returns structured response: {entities: [...], memories: [...], results: [...]}

Usage in the API:
    POST /api/memory/search {"mode": "agentic", "query": "..."}
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from cems.lib.json_parsing import parse_json_list
from cems.llm.client import get_client
from cems.memory.retrieval import _make_snippet

logger = logging.getLogger(__name__)

from cems.agentic.rrf import RRF_K, reciprocal_rank_fusion

DEFAULT_MODEL = os.environ.get(
    "CEMS_AGENTIC_MODEL", "google/gemini-2.5-flash-lite"
)  # 1M context recommended — agents receive full memory dump

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

ENTITY_PICKER_SYSTEM = (
    "You are a Topic Matcher — a knowledge retrieval specialist.\n\n"
    "You receive a list of knowledge topic pages (entity pages). Each has a title "
    "and a brief summary synthesized from multiple source memories.\n\n"
    "Your task: given a user question, identify which topic pages are most likely "
    "to contain relevant knowledge. Return the IDs of the most relevant topics.\n\n"
    "Prefer topics that DIRECTLY address the question over loosely related ones.\n"
    "Return at most 5 IDs, ranked by relevance. If none are relevant, return an empty array."
)

ENTITY_PICKER_USER_PROMPT = """I need to find which knowledge topics are relevant to this question:

QUESTION: {question}
{project_context}
Below are {n} knowledge topic pages. Each has a title and summary.

{formatted_entities}

Return a JSON array of topic IDs ranked by relevance, most relevant first.
Return ONLY IDs that are relevant. If none seem relevant, return an empty array.
Return at most 5 IDs.

Example output: ["abc12345", "def67890"]"""


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


async def _load_entity_summaries(
    document_store,
    user_id: str,
    project: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Load entity page summaries for entity picker agent.

    Returns compact list: id, title, summary, source count.
    Same query pattern as api_wiki_index.
    """
    from uuid import UUID

    pool = await document_store._get_pool()
    user_uuid = UUID(user_id)

    async with pool.acquire() as conn:
        if project:
            rows = await conn.fetch(
                """
                SELECT id, title, content, source_ref, tags
                FROM memory_documents
                WHERE user_id = $1 AND category = 'entity-page'
                  AND deleted_at IS NULL AND source_ref LIKE $2
                ORDER BY COALESCE(shown_count, 0) DESC
                LIMIT $3
                """,
                user_uuid,
                f"project:{project}%",
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, title, content, source_ref, tags
                FROM memory_documents
                WHERE user_id = $1 AND category = 'entity-page' AND deleted_at IS NULL
                ORDER BY COALESCE(shown_count, 0) DESC
                LIMIT $2
                """,
                user_uuid,
                limit,
            )

    entries = []
    for row in rows:
        title = row["title"] or ""
        content = row["content"] or ""
        # Extract first 2-3 sentences as summary (skip title line)
        sentences = content.replace("\n", " ").split(". ")
        summary = ". ".join(sentences[1:4]).strip()
        if len(summary) > 200:
            summary = summary[:200] + "..."

        cluster_size = ""
        for tag in (row["tags"] or []):
            if tag.startswith("cluster-size:"):
                cluster_size = tag.split(":")[1]

        entries.append({
            "id": str(row["id"]),
            "title": title,
            "summary": summary,
            "sources": cluster_size,
        })
    return entries


def _format_entities_for_picker(entities: list[dict]) -> str:
    """Format entity summaries for entity picker agent context."""
    parts = []
    for ent in entities:
        eid = ent.get("id", "")[:8]
        title = ent.get("title", "")
        summary = ent.get("summary", "")
        sources = ent.get("sources", "")
        source_info = f", {sources} sources" if sources else ""
        parts.append(
            f"--- Topic {eid} (title: {title}{source_info}) ---\n"
            f"{summary}"
        )
    return "\n\n".join(parts)


def _run_entity_picker(
    question: str,
    entities_text: str,
    n_entities: int,
    model: str,
    project: str | None = None,
) -> tuple[str, str | list[str]]:
    """Run entity picker agent. Thread-safe. Returns (role, raw_response)."""
    project_context = f"\nCURRENT PROJECT: {project}\n" if project else ""
    user_prompt = ENTITY_PICKER_USER_PROMPT.format(
        question=question,
        n=n_entities,
        formatted_entities=entities_text,
        project_context=project_context,
    )

    client = get_client()
    try:
        response = client.complete(
            prompt=user_prompt,
            system=ENTITY_PICKER_SYSTEM,
            model=model,
            temperature=0.1,
            max_tokens=500,
            fast_route=False,
        )
    except Exception as e:
        logger.warning(f"Entity picker agent failed: {e}")
        return "entity_picker", []

    if not response:
        return "entity_picker", []

    return "entity_picker", response


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


# Categories that are always relevant regardless of project
PROFILE_CATEGORIES = {"preferences", "guidelines", "gate-rules", "category-summary"}

# Categories handled separately (entity picker) — exclude from memory buckets
EXCLUDED_MEMORY_CATEGORIES = {"entity-page"}

# How far back "recent" memories go
RECENT_DAYS = 14

# Max chars to send to agents (leaves headroom in model context)
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
        project_docs = [d for d in project_docs if d.get("category") not in EXCLUDED_MEMORY_CATEGORIES]
        project_docs.sort(key=_relevance_score, reverse=True)
        b1 = _add_unique(project_docs)
        logger.debug(f"Agentic context bucket 1 (project={project}): {b1} memories")

    # Bucket 2: PROFILE memories — no project, always relevant (parallel)
    import asyncio as _aio
    profile_results = await _aio.gather(*[
        document_store.get_all_documents(
            user_id=user_id,
            scope=scope_filter,
            category=cat,
            limit=200,
        )
        for cat in PROFILE_CATEGORIES
    ])
    for cat_docs in profile_results:
        _add_unique(cat_docs)
    b2_total = len(all_memories) - b1
    logger.debug(f"Agentic context bucket 2 (profile): {b2_total} memories")

    # Bucket 3: RECENT general memories — same project + no-project only
    # Excludes other-project memories to prevent noise
    cutoff = datetime.now(UTC) - timedelta(days=RECENT_DAYS)
    recent_docs = await document_store.get_all_documents(
        user_id=user_id,
        scope=scope_filter,
        limit=500,
        order="desc",
        created_after=cutoff,
    )
    recent_filtered = []
    for d in recent_docs:
        src = d.get("source_ref") or ""
        cat = d.get("category") or ""
        if cat in EXCLUDED_MEMORY_CATEGORIES:
            continue
        if project and src and f"project:{project}".lower() not in src.lower():
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
    """Run agentic search with entity-first architecture.

    Runs 4 parallel agents:
      1. Entity Picker: searches entity page summaries (~100 docs, ~50K chars)
      2. Direct Seeker: searches raw memories for exact matches
      3. Inference Engine: searches raw memories for implicit connections
      4. Temporal Navigator: searches raw memories for temporal relevance

    Returns structured response with separate entities and memories arrays.

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
        Dict with entities, memories, and results (backward compat) arrays
    """
    # Load entity summaries and raw memories in parallel
    entity_summaries, memories = await asyncio.gather(
        _load_entity_summaries(document_store, user_id, project=project),
        _load_context_memories(document_store, user_id, project=project, scope=scope),
    )

    empty_response = {
        "entities": [],
        "memories": [],
        "results": [],
        "count": 0,
        "mode": "agentic",
        "tokens_used": 0,
        "queries_used": 0,
        "total_candidates": 0,
        "entity_candidates": len(entity_summaries),
        "filtered_count": 0,
    }

    if not memories and not entity_summaries:
        return empty_response

    # Format memories for memory agents
    memories_text = _format_memories_for_agents(memories) if memories else ""
    n_memories = len(memories)

    # Build valid memory ID set
    mem_id_to_full: dict[str, str] = {}
    for mem in memories:
        short_id = str(mem.get("id", ""))[:8]
        mem_id_to_full[short_id] = str(mem.get("id", ""))
    valid_mem_ids = set(mem_id_to_full.keys())

    # Build valid entity ID set
    ent_id_to_full: dict[str, dict] = {}
    for ent in entity_summaries:
        short_id = ent.get("id", "")[:8]
        ent_id_to_full[short_id] = ent
    valid_ent_ids = set(ent_id_to_full.keys())

    # Run 4 agents in parallel
    loop = asyncio.get_running_loop()
    memory_rankings: list[list[str]] = []
    entity_ranking: list[str] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []

        # Entity picker agent (if entity pages exist)
        if entity_summaries:
            entities_text = _format_entities_for_picker(entity_summaries)
            futures.append(
                loop.run_in_executor(
                    pool, _run_entity_picker,
                    query, entities_text, len(entity_summaries), model, project,
                )
            )

        # 3 memory search agents (if memories exist)
        if memories:
            for role in AGENT_SYSTEM_PROMPTS:
                futures.append(
                    loop.run_in_executor(
                        pool, _run_single_agent,
                        role, query, memories_text, n_memories, model, project,
                    )
                )

        if not futures:
            return empty_response

        # Server-side timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Agentic search timed out after 10s")
            results = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Agentic search agent failed: {result}")
                continue
            role, raw_response = result

            if role == "entity_picker":
                # Parse entity picker response against entity IDs
                if isinstance(raw_response, str):
                    entity_ranking = _parse_agent_response(raw_response, valid_ent_ids)
                elif isinstance(raw_response, list):
                    entity_ranking = raw_response
            else:
                # Parse memory agent response
                if isinstance(raw_response, str):
                    parsed = _parse_agent_response(raw_response, valid_mem_ids)
                else:
                    parsed = raw_response
                if parsed:
                    memory_rankings.append(parsed)

    # Build entity results (top 3)
    top_entities = []
    for short_id in entity_ranking[:3]:
        ent = ent_id_to_full.get(short_id, {})
        if ent:
            top_entities.append(ent)

    # RRF merge for memories (top 3)
    top_memories = []
    if memory_rankings:
        merged_short_ids = reciprocal_rank_fusion(memory_rankings)[:3]
        mem_by_id = {}
        for mem in memories:
            full_id = str(mem.get("id", ""))
            short_id = full_id[:8]
            mem_by_id[short_id] = mem

        for i, short_id in enumerate(merged_short_ids):
            mem = mem_by_id.get(short_id, {})
            score = 1.0 - (i * 0.5 / max(len(merged_short_ids), 1))
            raw_content = mem.get("content", "")
            snippet, truncated = _make_snippet(raw_content)
            entry = {
                "memory_id": str(mem.get("id", "")),
                "content": snippet,
                "category": mem.get("category", ""),
                "scope": mem.get("scope", "personal"),
                "source_ref": mem.get("source_ref", ""),
                "tags": mem.get("tags", []),
                "score": round(score, 3),
                "created_at": str(mem.get("created_at", "")),
            }
            if truncated:
                entry["truncated"] = True
                entry["full_length"] = len(raw_content)
            elif mem.get("content_detailed"):
                entry["has_detailed"] = True
                entry["full_length"] = len(mem["content_detailed"])
            top_memories.append(entry)

    queries_used = (1 if entity_summaries else 0) + (3 if memories else 0)

    logger.info(
        f"Agentic search complete: {len(top_entities)} entities, "
        f"{len(top_memories)} memories from {n_memories} candidates "
        f"+ {len(entity_summaries)} entity pages"
    )

    return {
        "entities": top_entities,
        "memories": top_memories,
        "results": top_memories,  # backward compat
        "count": len(top_memories) + len(top_entities),
        "mode": "agentic",
        "tokens_used": 0,
        "queries_used": queries_used,
        "total_candidates": n_memories,
        "entity_candidates": len(entity_summaries),
        "filtered_count": len(top_memories) + len(top_entities),
    }
