# Task Plan: Option D — Hybrid Observer for CEMS

## Goal
Replace CEMS's verbose implementation-detail learnings with **high-level observations** inspired by Mastra's Observational Memory. Memories should be context ("User is deploying CEMS to production") not documentation ("docker compose build rebuilds containers").

**Detailed plan**: `research/option-d-observer-plan.md`

## Phases

### Phase 1: Observer Prompt + API Endpoint + Extraction
**Status:** complete

Create the observation extraction pipeline server-side.

**Files to create:**
- `src/cems/llm/observation_extraction.py` — Observer prompt + extraction logic (Gemini 2.5 Flash via OpenRouter)
- `src/cems/api/handlers/observation.py` — `POST /api/session/observe` endpoint
- `tests/test_observation_extraction.py` — unit tests for extraction

**Files to modify:**
- `src/cems/api/handlers/__init__.py` — export new handler
- `src/cems/server.py` — register `/api/session/observe` route

**Key decisions:**
- Model: Gemini 2.5 Flash via OpenRouter (same as Mastra uses)
- Observer prompt adapted from Mastra: assertion vs question, temporal anchoring, observation-level not implementation-level
- Storage: `memory_documents` with `category="observation"`, `tags=["observation"]`
- Max 3 observations per extraction call, each under 200 chars

### Phase 2: stop.py Quick Win
**Status:** complete

Add observation extraction to the existing stop hook for immediate value.

**Files to modify:**
- `hooks/stop.py` — after existing `analyze_session()`, also call `/api/session/observe`

**Key behavior:**
- At session end, compress transcript to text, send to new endpoint
- Fire-and-forget, same pattern as existing `analyze_session()`
- Gets us observations on every session end with zero new infrastructure

### Phase 3: Observer Daemon
**Status:** complete

Standalone Python daemon for mid-session observations.

**Files to create:**
- `src/cems/observer/__init__.py`
- `src/cems/observer/daemon.py` — main polling loop (30s interval)
- `src/cems/observer/session.py` — session discovery from `~/.claude/projects/*/`
- `src/cems/observer/state.py` — per-session state tracking (`~/.claude/observer/`)
- Entry point: `python -m cems.observer`

**Key behavior:**
- Scans `~/.claude/projects/*/` for JSONL files modified in last 2 hours
- Reads `cwd` + `gitBranch` from JSONL entries for project identification
- Triggers observation when 50KB of new content accumulates
- Tracks byte offset per session to only process new content

### Phase 4: Observation Surfacing in Hooks
**Status:** complete

Surface recent observations in Claude's context.

**Files to modify:**
- `hooks/user_prompts_submit.py` — fetch recent observations for current project
- Possibly `hooks/cems_session_start.py` — inject project observations at session start

**Key behavior:**
- After existing search, also fetch recent observations filtered by `source_ref`
- Inject as "Recent Observations" section in context

### Phase 5: Rich Transcript Extraction
**Status:** complete

Improve the quality of data sent to the observer by extracting richer signals from JSONL session files.

**Problem discovered:**
- Sessions have very few `type: "text"` user messages (3 out of 211 entries in a typical session)
- Most user entries are `tool_result` (191/211) — currently ignored
- Assistant text blocks average 276 chars (short narration between tool calls)
- The real substance is in tool_use inputs (what files were read/edited, commands run)
- Result: observer gets thin content → weak observations for tool-heavy sessions

**Research completed:**
- Claude Code JSONL has 6 entry types: `user`, `assistant`, `progress`, `system`, `file-history-snapshot`, `queue-operation`
- `assistant` content blocks: `text`, `thinking`, `tool_use` (with name + input)
- `user` content blocks: `text`, `tool_result` (with content, stdout, stderr)
- `isMeta` flag distinguishes system-injected messages from real user input
- Entire.io extracts: user prompts, last assistant text, modified files from Write/Edit/NotebookEdit
- Simon Willison's parser: most mature, handles edge cases

**Files to modify:**
- `src/cems/observer/session.py` — `read_content_delta()` to include tool_use summaries
- `hooks/stop.py` — `_extract_message_text()` + `observe_session()` to match

**Strategy: Add tool action summaries**
For `assistant` entries with `tool_use` blocks, extract a one-line summary:
- `[TOOL] Read: src/cems/server.py` (file path from input)
- `[TOOL] Edit: src/cems/config.py` (file path from input)
- `[TOOL] Bash: docker compose build` (command, truncated)
- `[TOOL] Write: tests/test_new.py` (new file created)
- `[TOOL] Grep: "pattern" in src/` (search pattern + path)

For `user` entries that are raw strings (not tool_result), ensure they're captured as `[USER]`.

Skip: `progress` entries, `file-history-snapshot`, `tool_result` content (too verbose), `thinking` blocks.

**Acceptance criteria:**
- Same session that previously produced 0 user messages now produces rich content
- Observer daemon and stop.py both use the same extraction logic (shared function)
- Existing tests still pass, new tests cover tool_use extraction
- Docker integration test confirms observations are stored

### Phase 6: Transcript Compaction
**Status:** complete

Compact raw tool action lines into higher-level activity summaries before sending to the observer. Currently the transcript sends 184 individual `[TOOL] Read: /path` lines — the observer sees noise and sometimes leaks file paths into observations.

**Problem:**
- Phase 5 added tool_use summaries, but they're 1:1 (one line per tool call)
- A session with 50 file reads in `src/cems/` produces 50 `[TOOL] Read:` lines
- The observer LLM sees all these paths and sometimes includes them in observations despite exclusion rules
- WebFetch/WebSearch actions get lost in the noise of file operations

**Strategy: Aggregate tool actions into activity summaries**

Transform individual tool lines into compacted activity descriptions:
- `[TOOL] Read: src/cems/server.py` x12 → `[ACTIVITY] Assistant explored src/cems/ (12 files read)`
- `[TOOL] Edit: src/cems/config.py` + `[TOOL] Edit: src/cems/server.py` → `[ACTIVITY] Assistant modified 2 files in src/cems/`
- `[TOOL] Bash: docker compose build` + `[TOOL] Bash: docker compose up` → `[ACTIVITY] Assistant ran Docker commands (build, deploy)`
- `[TOOL] Grep: "pattern" in src/` x5 → `[ACTIVITY] Assistant searched codebase for patterns`
- WebFetch/WebSearch → `[ACTIVITY] Assistant visited mastra.ai docs` (preserve URLs — these matter for context)

**Files to modify:**
- `src/cems/observer/transcript.py` — add `compact_tool_lines()` post-processing step
- `hooks/stop.py` — mirror the same compaction in inlined extraction

**Acceptance criteria:**
- Transcript sent to observer is ~70% shorter (fewer individual tool lines)
- Tool actions grouped by type + directory
- Web visits preserved with domain names
- Observer produces observations without file path leakage
- Existing tests still pass, new tests for compaction logic

### Phase 7: Reflector / Consolidation (Optional)
**Status:** pending

Merge old overlapping observations, soft-delete superseded ones.

**Files to modify:**
- `src/cems/maintenance/consolidation.py` or new maintenance job

---

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Observer Prompt + API + Extraction | `complete` | 18 tests, Mastra-style prompt |
| Phase 2: stop.py Quick Win | `complete` | Fire-and-forget observe at session end |
| Phase 3: Observer Daemon | `complete` | 14 tests, `python -m cems.observer` |
| Phase 4: Observation Surfacing | `complete` | Fetch + inject in UserPromptSubmit |
| Phase 5: Rich Transcript Extraction | `complete` | 28 new tests, shared extraction, 5/5 user-focused obs |
| Phase 6: Transcript Compaction | `complete` | 23 new tests, 42% tool line reduction, 5/5 observations clean |
| Phase 7: Reflector/Consolidation | `pending` | Optional |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `_parse_observations` empty for valid JSON | 1 | Used `parse_json_list()` not `extract_json_from_response()` |
| Test AttributeError on lazy import | 1 | Moved to top-level import for test patching |
| Test expects 1 search req, now 2 | 1 | Updated assertion to `== 2` (memory + observation) |
| FAST_PROVIDERS latency for Gemini | codex | Added `fast_route=False` parameter to `complete()` |
| Markdown-fenced JSON without closing backticks | 1 | Added fallback regex in `extract_json_from_response()` |
| 0 user messages in 211-entry session | research | JSONL stores user text as raw_string, not always in content blocks |
