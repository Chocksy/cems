# Imperative Entity Recall + Session-Enriched Search

**Date:** 2026-04-09
**Status:** Ready for planning

## What We're Building

Two improvements to memory recall quality:

### Thing 1: Imperative Entity Fetch in Hook Prompt

**Problem:** Today's data shows **0% follow-through** on entity topic suggestions. Across 55 entity ref suggestions in 22 user turns, the LLM never once called `/recall` to fetch the full entity page. The current passive language ("Use /recall <id> to read full topic pages for deeper context") is completely ignored.

**Solution:** Change the hook's agentic response format to use **imperative language** that forces the LLM to fetch ALL listed entity pages before proceeding. Based on research of mem0, Letta, Zep, and Khoj, the most effective pattern is explicit conditions ("you MUST do X when Y").

**Key decisions:**
- Fetch ALL entity topics found (not just top 1)
- Use imperative "You MUST call `/recall <id>`" for each entity
- Add a clear "REQUIRED ACTIONS" section above the entity list
- Frame entities as documents the LLM hasn't read yet (not optional references)

**Where to change:** `_format_agentic_response()` in `hooks/cems_user_prompts_submit.py` (lines 89-147). Change the footer from passive to imperative. Add a REQUIRED ACTIONS block listing each entity ID.

### Thing 2: Session-Enriched Search

**Problem:** Short or confirmatory prompts ("yes, let's do this") have zero context for the search query. Even with the assistant message fallback for confirmatory prompts, the search often returns irrelevant results because it lacks conversation context.

**Solution:** Pass `session_id` in the search request. The **server** looks up the latest observer session summary for that session and uses it to enrich/contextualize the search query. This works for ALL clients:

- **Hooks** — already have `session_id` from input_data
- **MCP tools** — `memory_search` can accept an optional `session_id` param
- **CLI** — `cems search` can accept `--session-id`
- **Codex /recall skill** — instructs the LLM to pass session context (or the MCP/CLI handles it transparently)

**Key decisions:**
- Server-side enrichment (not client-side) — so it works everywhere
- Use only the latest epoch's session summary (not all epochs)
- The session summary is prepended to the search query for better semantic matching
- For Codex (no hooks), the `/recall` skill should instruct the LLM to include session context in the search query as a fallback

**Where to change:**
1. Hook: add `session_id` to search payload in `search_cems()`
2. Server: `/api/memory/search` handler looks up session summary when `session_id` is provided
3. MCP: `memory_search` tool adds optional `session_id` parameter
4. Skills: update recall skill to mention session context

## Why This Approach

- **Imperative prompts work:** mem0's recall-protocol uses explicit "Use memory_search WHEN..." conditions and is the most sophisticated prompt in the space. The 0% rate proves passive suggestions fail.
- **Server-side enrichment is universal:** putting the session lookup in the server means hooks, MCP, CLI, and Codex skills all benefit without duplicating logic.
- **Entity pages are pre-curated:** we already spent LLM tokens on the server to identify relevant entities. Forcing the LLM to read them is not wasteful — it's completing the retrieval pipeline.

## Open Questions

1. Should we add a token budget check? If 3 entities x ~2000 chars each = 6000 chars, that's significant context. Maybe cap at 2 entities?
2. For session-enriched search, how long should the session summary be? Currently capped at 10K chars — should we use a shorter excerpt for query enrichment?
3. Should the imperative instruction only apply when the user's prompt is substantive (not "yes" or short confirmations)?
