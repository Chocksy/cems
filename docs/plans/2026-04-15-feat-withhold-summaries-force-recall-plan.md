---
title: "feat: Withhold summaries from KNOWLEDGE TOPICS to force LLM recall"
type: feat
date: 2026-04-15
---

# feat: Withhold summaries from KNOWLEDGE TOPICS to force LLM recall

## Overview

The UserPromptSubmit hook injects KNOWLEDGE TOPICS with 200-char summaries. LLMs read these summaries, judge them "good enough," and skip the `/recall` commands — never fetching full entity page content. This defeats the purpose of curated knowledge pages.

**Fix**: Remove summaries from the hook output. Inject only titles + IDs + source counts, framed as "unresolved references" that require fetching. The LLM must call `memory_get` or `/recall` to see any content.

## Problem Statement

Observed in production: when the hook injects KNOWLEDGE TOPICS like:

```
KNOWLEDGE TOPICS matching your query:

1. CEMS Session Summarization and Authentication Troubleshooting (3 sources)
   The core challenge revolved around resolving a 404 error initially assumed to indicate a missing endpoint...
   → /recall 76a73854 for full details
```

The LLM reads the summary, makes a relevance judgment from it, and never calls `/recall`. The "REQUIRED: Fetch these knowledge pages before responding" instruction is treated as advisory. Research confirms Claude treats CLAUDE.md and system-reminder instructions as advisory (anthropics/claude-code#7777).

**Root cause**: When summaries are present, the LLM has enough information to *feel* informed without actually being informed. Karpathy's LLM Knowledge Base pattern validates this — his `index.md` contains titles + links only, forcing the LLM to open pages.

## Proposed Solution

Change the hook output format from summary-included to stub-only. Two changes:

### Change 1: Hook formatter — `_format_agentic_response()` in `hooks/cems_user_prompts_submit.py`

**Before** (current, lines 103-122):
```
KNOWLEDGE TOPICS matching your query:

1. CEMS Session Summarization and Auth Troubleshooting (3 sources)
   The core challenge revolved around resolving a 404 error...
   → /recall 76a73854 for full details

REQUIRED: Fetch these knowledge pages before responding:
  /recall 76a73854
  /recall a52d3e75
These are curated documents relevant to your task. Read them first.
```

**After** (proposed):
```
KNOWLEDGE TOPICS (content not loaded — you must fetch before responding):

1. CEMS Session Summarization and Auth Troubleshooting (3 sources)
   ⚠ PENDING — call /recall 76a73854 to load content
2. CEMS Credentials Resolution and Configuration Management (4 sources)
   ⚠ PENDING — call /recall a52d3e75 to load content

DO NOT reference these topics without loading them first.
Summaries are intentionally withheld — you must read the full document.
```

Key design decisions:
- **Title stays** — it provides enough signal for the LLM to judge *whether* to fetch, but not enough to fake having read it
- **Source count stays** — signals importance (more sources = more curated)
- **Summary removed** — this is the key change. No content excerpt whatsoever
- **"PENDING" framing** — makes the absence structural, not optional. Like an unresolved import
- **Warning emoji** — visual break that signals "action needed"
- **"DO NOT reference without loading"** — direct prohibition rather than "please fetch"

### Change 2: MCP tool formatter — `_format_agentic_for_mcp()` in `src/cems/mcp_stdio.py`

**Decision: Keep summaries in MCP tool output.**

Rationale: When the LLM explicitly calls `memory_search`, it's already doing the work of retrieval. Summaries help it decide which results to dig into. The problem is only with *auto-injected* context (the hook), where the LLM didn't ask for the information and treats it as disposable background.

However, update the wording to be consistent:
- Change `→ /recall {id} for full details` to `→ call /recall {id} to read full document`

### What does NOT change

| Component | Change? | Reason |
|---|---|---|
| `/api/memory/search` response | No | API returns full data; formatting layers decide presentation |
| `_load_entity_summaries()` in agentic search | No | Entity Picker LLM agent needs summaries to match entities to queries |
| `/recall` skill (all platforms) | No | Already fetches full content via `memory_get` |
| Codex/Cursor hooks | No | They don't inject per-prompt KNOWLEDGE TOPICS |
| RELEVANT MEMORIES section in hook | No | These are already content snippets, not summaries. Truncation hints already say "use /recall to read full document" |
| Classic vector mode formatter | No | No KNOWLEDGE TOPICS section exists |

## Acceptance Criteria

- [x] KNOWLEDGE TOPICS in hook output contain title + source count only — no summary text
- [x] Each topic line includes `⚠ PENDING — call /recall {id} to load content`
- [x] Footer says "DO NOT reference these topics without loading them first"
- [x] The "REQUIRED: Fetch these knowledge pages" block is replaced with the new format
- [x] MCP `memory_search` tool output still includes summaries (unchanged behavior)
- [x] `/recall` skill still works as before (calls `memory_get` for full content)
- [x] Bundled hooks in `src/cems/data/claude/hooks/` are updated to match
- [x] Hook tests pass with updated expected output format
- [ ] Installed hooks (via `cems setup`) pick up the new format

## Technical Approach

### Files to modify

1. **`hooks/cems_user_prompts_submit.py`** — `_format_agentic_response()` function
   - Remove summary inclusion (lines ~108-109 where `e.get("summary")` is used)
   - Reformat KNOWLEDGE TOPICS section header and per-entity lines
   - Replace the "REQUIRED: Fetch these knowledge pages" block with new footer
   - Keep the entity ID short format (first 8 chars of UUID)

2. **`src/cems/data/claude/hooks/cems_user_prompts_submit.py`** — bundled copy
   - Must be kept in sync. Either:
     - (a) Edit both files identically, or
     - (b) Copy `hooks/` → `src/cems/data/claude/hooks/` after editing

3. **`src/cems/mcp_stdio.py`** — `_format_agentic_for_mcp()` (minor wording tweak only)
   - Update recall hint wording for consistency

### Implementation steps

1. Edit `hooks/cems_user_prompts_submit.py` — modify `_format_agentic_response()`
2. Copy updated hook to `src/cems/data/claude/hooks/cems_user_prompts_submit.py`
3. Update MCP formatter wording in `src/cems/mcp_stdio.py`
4. Run hook tests: `.venv/bin/python3 -m pytest tests/test_hooks.py -x -q`
5. Manual verification: run hook against real API and inspect output format
6. Install updated hooks: `cems setup --claude`

### Testing

- **Unit tests**: Update expected output in `tests/test_hooks.py` for new KNOWLEDGE TOPICS format
- **Manual test**: Run the hook script directly with a test prompt and verify output
- **Integration**: Start a Claude Code session in a project with entity pages and verify the LLM calls `/recall` before referencing topics

## Context: Why this matters for Codex

Codex doesn't have a UserPromptSubmit hook that injects KNOWLEDGE TOPICS. Memory in Codex flows through:
1. `/recall` skill (user-invoked) → calls `memory_search` MCP → displays results with summaries → fetches full docs for truncated ones
2. Direct `memory_search` MCP tool call by the agent

Since Codex already requires explicit action to get memory (no auto-injection), the "lazy reading" problem is less acute there. The MCP tool keeping summaries is fine — the agent chose to search.

If we later want parity, the path would be adding a Codex hook that auto-injects topics — and that hook would use the same stub format.

## References

- Research conversation (2026-04-14): Option C analysis, ClawMem/claude-recall/Karpathy patterns
- `hooks/cems_user_prompts_submit.py` — primary hook, lines 89-157 for agentic formatter
- `src/cems/agentic/search.py` — entity summary generation, lines 145-210
- `src/cems/mcp_stdio.py` — MCP tool formatter, lines 175-213
- Karpathy's LLM Knowledge Base — title+links only in index.md
- anthropics/claude-code#7777 — Claude treats CLAUDE.md as advisory
