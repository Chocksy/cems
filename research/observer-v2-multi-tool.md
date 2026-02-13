# Observer V2: Multi-Tool Signal-Based Architecture

## Date: 2026-02-13
## Status: Research Complete, Planning Phase

---

## 1. Architecture Vision

Transform the CEMS observer from a Claude Code-only daemon into a **multi-tool session learning engine** with:
- **Adapters** for Claude Code, Codex CLI, and Cursor IDE
- **Signal-based lifecycle** (hooks write tiny signals, daemon does all processing)
- **Staleness detection** as fallback for crashed/killed sessions
- **Epoch model** for long sessions with compact/reset points

### Core Principle
> The observer already watches session files — it has all the data. Hooks should be cheap signals, not data processors.

---

## 2. Tool-Specific Research

### 2.1 Claude Code (Current — Fully Supported)

**Session Files:**
- Location: `~/.claude/projects/{project-dir}/{session-uuid}.jsonl`
- Format: JSONL (one JSON object per line)
- Growth: Incremental (new lines appended as conversation progresses)
- Example entry types: `user`, `assistant` (with `text`, `tool_use`, `tool_result` blocks)

**Verified on machine:**
```
~/.claude/projects/-Users-razvan-Development-cems/
  ├── b40eb706-03db-420a-a853-4d75d19aa964.jsonl  (current session)
  └── ... (many more sessions)
```

**Hooks System:**
- SessionStart: Injects user profile via `hookSpecificOutput.additionalContext`
- UserPromptSubmit: Memory search, observations, gate rules, ultrathink
- PreToolUse: Gate rules (block=exit 2 stderr, warn=additionalContext)
- PostToolUse: Tool learning to `/api/tool/learning`
- Stop: Session analysis + summary
- PreCompact: Session analysis before context compaction

**Key Capability:** Hooks can inject context into Claude's conversation (UserPromptSubmit, SessionStart).

**Transcript Parser:** `src/cems/observer/transcript.py` — extracts `[USER]`, `[ASSISTANT]`, `[TOOL]` lines, compacts to `[ACTIVITY]` summaries.

### 2.2 Codex CLI (Ready for Adapter)

**Session Files:**
- Location: `~/.codex/sessions/YYYY/MM/DD/rollout-{timestamp}-{uuid}.jsonl`
- Format: JSONL (same concept as Claude Code but different schema)
- Growth: Incremental (appended as events occur)
- Schema: `session_meta`, `response_item`, `event_msg`, `turn_context` event types

**Verified on machine:**
```
~/.codex/sessions/2026/02/05/rollout-2026-02-05T13-29-42-019c2d90-ad5b-77c2-b283-f0ebc1de51fe.jsonl
~/.codex/sessions/2026/02/04/ (3 sessions)
~/.codex/sessions/2026/02/03/ (2 sessions)
~/.codex/sessions/2025/ (many sessions back to Sep 2025)
```

**JSONL Schema (from actual file):**
```jsonl
{"timestamp":"...","type":"session_meta","payload":{"id":"...","cwd":"...","model_provider":"openai","git":{...}}}
{"timestamp":"...","type":"response_item","payload":{"type":"message","role":"developer","content":[...]}}
{"timestamp":"...","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"..."}]}}
{"timestamp":"...","type":"event_msg","payload":{"type":"user_message","message":"...","images":[]}}
{"timestamp":"...","type":"event_msg","payload":{"type":"agent_reasoning","text":"..."}}
{"timestamp":"...","type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"..."}}
{"timestamp":"...","type":"response_item","payload":{"type":"function_call_output","call_id":"...","output":"..."}}
{"timestamp":"...","type":"turn_context","payload":{"model":"gpt-5.2-codex","effort":"high",...}}
{"timestamp":"...","type":"event_msg","payload":{"type":"token_count","info":{...}}}
```

**Hooks System:** None yet (actively being designed, Feb 2026). Signal → staleness detection only.

**Key Fields for Extraction:**
- `session_meta.payload.cwd` → project directory
- `session_meta.payload.git.repository_url` → project ID
- `session_meta.payload.git.branch` → git branch
- `response_item` with `role:"user"` → user messages
- `response_item` with `role:"assistant"` or reasoning → assistant output
- `function_call` / `function_call_output` → tool usage

### 2.3 Cursor IDE (Watchable via Transcripts)

**Session Files:**
- Location: `~/.cursor/projects/{project-dir}/agent-transcripts/{uuid}.txt`
- Format: Plain text (human-readable)
- Growth: Grows during agent sessions
- Additional: SQLite `state.vscdb` (harder to watch real-time)

**Verified on machine:**
```
~/.cursor/projects/Users-razvan-Development-cems/agent-transcripts/
  ├── ce7b1dfa-...txt  (4.7KB)
  ├── 137e2b99-...txt  (74KB)
  ├── 5d798100-...txt  (572KB)
  ├── 70892aa7-...txt  (636KB)  ← largest
  └── ... (14 transcripts total for CEMS project)
```

**Transcript Format (from actual file):**
```
user:
<cursor_commands>
--- Cursor Command: research ---
... (command metadata) ...
--- End Command ---
its there... FML dumb... :(
can we use the x-ai/grok-4.1-fast for those too?

assistant:
(response text)
```

**Hooks System (verified `~/.cursor/hooks.json`):**
```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [{"command": "./hooks/cems_session_start.py"}],
    "afterAgentResponse": [{"command": "./hooks/cems_agent_response.py"}],
    "stop": [{"command": "./hooks/cems_stop.py"}]
  }
}
```

**6 Cursor hook events:**
| Hook | Can Block? | Can Inject Context? |
|------|-----------|-------------------|
| beforeSubmitPrompt | No (informational) | NO — read-only |
| beforeShellExecution | Yes | N/A |
| beforeMCPExecution | Yes | N/A |
| beforeReadFile | Yes | N/A |
| afterFileEdit | No | N/A |
| stop | No | N/A |

**Key limitation:** `beforeSubmitPrompt` is informational only. Cannot inject memory context like Claude Code's `UserPromptSubmit`. Memory recall for Cursor must go through MCP tools.

---

## 3. Architecture Design: Option 3 (Signal + Staleness)

### 3.1 Signal File Model

Hooks write tiny signal files instead of processing transcripts:

```
~/.claude/observer/signals/{session_id}.json
```

Signal format:
```json
{"type": "compact", "ts": 1707840000.0, "tool": "claude"}
{"type": "stop", "ts": 1707841000.0, "tool": "claude"}
```

Signal types:
| Signal | Source | Daemon Action |
|--------|--------|--------------|
| `compact` | Pre-compact hook | Finalize current epoch, bump epoch counter |
| `stop` | Stop hook | Finalize, mark session done, stop watching |
| `clear` | (future) | Finalize, start new epoch |

### 3.2 Epoch Model

Long sessions with multiple compacts produce multiple documents:

```
Epoch 0: session start → compact → finalize doc (tag: session:{id[:8]})
Epoch 1: after compact → next compact → finalize doc (tag: session:{id[:8]}:e1)
Epoch 2: after compact → stop → finalize doc (tag: session:{id[:8]}:e2)
```

State addition to `ObservationState`:
```python
@dataclass
class ObservationState:
    session_id: str
    epoch: int = 0                    # NEW: incremented on compact
    # ... existing fields ...
```

### 3.3 Staleness Detection (Fallback)

For sessions without signal hooks (Codex, crashed sessions):
- No file growth for 5 minutes → auto-finalize
- Handles: killed processes, laptop close, SSH disconnect
- Conservative: only triggers after definitive inactivity

### 3.4 Multi-Tool Adapter Pattern

```python
class SessionAdapter(Protocol):
    """Interface each tool adapter implements."""
    def discover_sessions(self, max_age_hours: int) -> list[SessionInfo]: ...
    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None: ...
    def parse_session_id(self, path: Path) -> str: ...
    def enrich_metadata(self, session: SessionInfo) -> SessionInfo: ...
```

Three adapters:
1. `ClaudeAdapter` — watches `~/.claude/projects/*/*.jsonl`
2. `CodexAdapter` — watches `~/.codex/sessions/**/*.jsonl`
3. `CursorAdapter` — watches `~/.cursor/projects/*/agent-transcripts/*.txt`

### 3.5 Daemon Loop (Updated)

```
each cycle (30s):
  for each adapter:
    sessions = adapter.discover_sessions()
    for each session:
      1. check signals → handle stop/compact
      2. check file growth → incremental if threshold met
      3. check staleness → auto-finalize if no growth for 5 min
```

---

## 4. Hook Transformation

### What stays in hooks (no change):
- **SessionStart** → profile injection (observer can't do this)
- **UserPromptSubmit** → memory search, observations, gate rules
- **PreToolUse** → gate rules

### What becomes thin signals:
- **Stop hook** → writes signal file, keeps session logging (450 lines → ~30 lines)
- **PreCompact hook** → writes signal file (160 lines → ~15 lines)
- **PostToolUse** → could move to observer later

### Hook stays for Cursor too:
- Cursor's `stop` hook can write the same signal file format
- Cursor's `beforeSubmitPrompt` is useless for memory injection (use MCP instead)

---

## 5. Capability Matrix

| Capability | Claude Code | Codex CLI | Cursor IDE |
|-----------|------------|-----------|------------|
| Observer file watching | JSONL ✅ | JSONL ✅ | .txt transcripts ✅ |
| Signal hooks | stop + pre-compact ✅ | none (staleness only) | stop hook ✅ |
| Memory injection | UserPromptSubmit hook ✅ | MCP tools only | MCP tools only |
| Guard rules | PreToolUse hook ✅ | MCP tools | beforeShellExecution ✅ |
| Profile injection | SessionStart hook ✅ | MCP tools | MCP tools |
| Epoch detection | Signal + hook ✅ | Staleness heuristic | Signal + hook ✅ |

---

## 6. Current Codebase Inventory

### Observer Package (`src/cems/observer/`)
| File | Purpose | Lines | Changes Needed |
|------|---------|-------|----------------|
| `daemon.py` | Poll loop, process sessions | 280 | Add signal handling, staleness, adapter dispatch |
| `session.py` | Discover Claude sessions, read deltas | 164 | Extract to ClaudeAdapter |
| `transcript.py` | Parse Claude JSONL to text | 376 | Keep as Claude-specific parser |
| `state.py` | Per-session state tracking | 97 | Add epoch field, signal file support |
| `__main__.py` | CLI entry point | ~20 | No change |

### Hooks (`hooks/`)
| File | Current Size | After Refactor |
|------|-------------|----------------|
| `cems_stop.py` | 457 lines | ~30 lines (signal + session logging) |
| `cems_pre_compact.py` | 157 lines | ~15 lines (signal only) |
| `cems_session_start.py` | ~100 lines | No change (profile injection) |
| `cems_user_prompts_submit.py` | ~200 lines | No change (memory search) |
| `cems_pre_tool_use.py` | ~100 lines | No change (gate rules) |

### API Handlers (`src/cems/api/handlers/`)
| File | Purpose | Changes Needed |
|------|---------|----------------|
| `session.py` | `/api/session/summarize` upsert | Add epoch-aware tagging |

---

## 7. Implementation Priority

1. **Signal infrastructure** — signal files, daemon signal reader
2. **Epoch model** — state tracking, epoch-aware tags
3. **Incremental append** — don't replace, accumulate summaries
4. **Staleness detection** — auto-finalize idle sessions
5. **Codex adapter** — new transcript parser, session discovery
6. **Cursor adapter** — new transcript parser, session discovery
7. **Hook simplification** — strip stop/pre-compact to signals
8. **Tests** — unit tests for adapters, signals, staleness
