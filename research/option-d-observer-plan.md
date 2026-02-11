# Option D: Hybrid Observer — Implementation Plan

## Vision

Replace CEMS's verbose implementation-detail learnings with **high-level observations** inspired by Mastra's Observational Memory. The core insight: memories like "User is deploying CEMS to production on Hetzner" are far more useful than "docker compose build rebuilds the container".

Two systems working together:
1. **Observer Service** — standalone daemon that watches session transcripts and produces observations mid-session
2. **Existing Hooks** — continue surfacing memories, gate rules, ultrathink — but now also surface recent observations

The Observer doesn't replace hooks. It runs alongside them, producing a new class of memory (`category="observation"`) that gets stored in the same `memory_documents` table and surfaced via the same search pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code Session                 │
│                                                     │
│  User ↔ Claude  (writes session JSONL continuously) │
└──────────────┬──────────────────────────────────────┘
               │ JSONL file grows on disk
               ▼
┌──────────────────────────┐    ┌─────────────────────┐
│   Observer Daemon         │    │  Hooks (existing)   │
│                          │    │                     │
│  - Polls active sessions │    │  SessionStart       │
│  - Detects new content   │    │  UserPromptSubmit   │
│  - Sends to CEMS API     │    │  PreToolUse         │
│    /api/session/observe   │    │  PostToolUse        │
│  - Tracks observation    │    │  Stop               │
│    state per session     │    │                     │
└──────────┬───────────────┘    └──────────┬──────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────┐
│                    CEMS Server                        │
│                                                      │
│  POST /api/session/observe   (NEW)                   │
│  POST /api/session/analyze   (existing — learnings)  │
│  POST /api/memory/search     (existing — retrieval)  │
│                                                      │
│  memory_documents table:                             │
│    category="observation" → Observer output           │
│    category="*"           → Existing learnings        │
│                                                      │
│  Retrieval pipeline returns both types                │
└──────────────────────────────────────────────────────┘
```

---

## Component 1: Observer Daemon

### What It Does

A lightweight Python daemon that:
1. Discovers active Claude Code sessions by scanning `~/.claude/projects/*/`
2. Polls session JSONL files for new content (every 30s)
3. When enough new content accumulates (50KB delta), sends it to CEMS for observation extraction
4. Tracks per-session state (last observed byte offset, session metadata)

### Why a Daemon (Not a Hook)

- **Hooks must be fast** — Claude Code blocks on hook execution. Observation extraction takes 3-10s (LLM call). Can't block the user.
- **Hooks see one prompt at a time** — Observer needs to see the full session arc to produce good observations.
- **Fire-and-forget from hooks would work** but loses the ability to track state cleanly. A daemon owns its state.
- **Session file access** — hooks get `transcript_path` but only for the current session. The daemon can watch ALL active sessions across ALL projects.

### Session Discovery

```python
# Session files live at:
# ~/.claude/projects/{encoded-project-path}/{session-uuid}.jsonl
#
# The project path encodes the working directory:
#   /Users/razvan/Development/cems → -Users-razvan-Development-cems
#
# Each JSONL entry has metadata:
#   {"sessionId": "uuid", "cwd": "/Users/razvan/Development/cems",
#    "gitBranch": "main", "version": "2.1.39", ...}

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

def discover_active_sessions(max_age_hours: int = 2) -> list[SessionInfo]:
    """Find sessions modified within the last N hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    sessions = []

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            if jsonl_file.stat().st_mtime > cutoff:
                sessions.append(SessionInfo(
                    path=jsonl_file,
                    project_dir=project_dir.name,
                    session_id=jsonl_file.stem,
                ))
    return sessions
```

### Project & Repository Identification

Critical requirement from user: "the observer needs to understand its project and repository."

The JSONL entries contain `cwd` and `gitBranch`. From `cwd` we can derive the git remote (same logic as `stop.py:get_project_id()`):

```python
def get_session_metadata(session_path: Path) -> dict:
    """Read first entry to get session metadata."""
    with open(session_path) as f:
        first_line = f.readline()
        entry = json.loads(first_line)
        cwd = entry.get("cwd", "")
        git_branch = entry.get("gitBranch", "")

    # Derive project ID from git remote
    project_id = get_project_id(cwd)  # reuse stop.py logic

    return {
        "cwd": cwd,
        "git_branch": git_branch,
        "project_id": project_id,        # e.g., "chocksy/cems"
        "source_ref": f"project:{project_id}" if project_id else None,
    }
```

### Observation State Tracking

Per-session state file at `~/.claude/observer/{session-uuid}.json`:

```json
{
  "session_id": "a968ab3a-fce7-4fcf-b17b-cc361af57bb0",
  "project_id": "chocksy/cems",
  "source_ref": "project:chocksy/cems",
  "last_observed_bytes": 1048576,
  "last_observed_at": "2026-02-11T10:30:00Z",
  "observation_count": 3,
  "session_started": "2026-02-11T09:00:00Z"
}
```

### Content Delta Reading

Only read new content since last observation:

```python
OBSERVATION_THRESHOLD = 50_000  # ~12-15k tokens of new content

def get_new_content(session: SessionInfo, state: ObservationState) -> str | None:
    """Read transcript content added since last observation."""
    file_size = session.path.stat().st_size
    delta = file_size - state.last_observed_bytes

    if delta < OBSERVATION_THRESHOLD:
        return None  # Not enough new content

    with open(session.path, "rb") as f:
        f.seek(state.last_observed_bytes)
        new_bytes = f.read()

    # Parse JSONL entries, extract user/assistant messages only
    messages = []
    for line in new_bytes.decode("utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
            if entry.get("type") in ("user", "assistant"):
                msg = entry.get("message", {})
                role = msg.get("role", entry.get("type"))
                content = extract_text_content(msg.get("content", ""))
                if content:
                    messages.append(f"[{role}]: {content[:2000]}")
        except json.JSONDecodeError:
            continue

    return "\n\n".join(messages) if messages else None
```

### Daemon Main Loop

```python
async def observer_loop():
    """Main observer loop — runs every 30 seconds."""
    while True:
        sessions = discover_active_sessions(max_age_hours=2)

        for session in sessions:
            state = load_state(session.session_id)
            metadata = get_session_metadata(session.path)

            new_content = get_new_content(session, state)
            if new_content is None:
                continue

            # Fire observation request to CEMS
            await send_observation(
                content=new_content,
                session_id=session.session_id,
                source_ref=metadata["source_ref"],
                project_context=f"{metadata['project_id']} ({metadata['git_branch']})",
                previous_observations=state.observation_count,
            )

            # Update state
            state.last_observed_bytes = session.path.stat().st_size
            state.last_observed_at = datetime.utcnow().isoformat()
            state.observation_count += 1
            save_state(state)

        await asyncio.sleep(30)
```

### Running the Daemon

```bash
# Start as background process
python -m cems.observer.daemon &

# Or via systemd/launchctl for persistence
# Or integrated into the CEMS Docker service
```

The daemon is lightweight — it only reads files and makes HTTP calls. No LLM calls happen locally; all extraction is done server-side.

---

## Component 2: Observer API Endpoint

### New Endpoint: `POST /api/session/observe`

```python
# src/cems/api/handlers/observation.py

async def api_session_observe(request: Request):
    """Extract high-level observations from session content.

    POST /api/session/observe
    Body: {
        "content": "...",           # New transcript content (text)
        "session_id": "...",        # Session identifier
        "source_ref": "project:org/repo",  # Project context
        "project_context": "chocksy/cems (main)",  # Human-readable
        "previous_observations": 3,  # How many observations already made
    }

    Response: {
        "success": true,
        "observations_stored": 2,
        "observations": [...]
    }
    """
```

### Key Difference from `/api/session/analyze`

| | `/api/session/analyze` (existing) | `/api/session/observe` (new) |
|---|---|---|
| **When called** | Session end (stop hook) | Mid-session (observer daemon) |
| **Input** | Full transcript as message array | Recent content delta as text |
| **Extracts** | Implementation learnings (code patterns, error fixes) | High-level observations (what user is doing) |
| **LLM model** | Grok 4.1 Fast | Gemini 2.5 Flash (via OpenRouter) |
| **Storage** | `tags=["session-learning"]` | `tags=["observation"]`, `category="observation"` |
| **Volume** | 1-5 learnings per session | 1-3 observations per 50KB chunk |

Both write to `memory_documents`. Both are searchable. Different extraction prompts produce different kinds of memories.

---

## Component 3: Observer Extraction Prompt

This is the highest-value piece — adapted from Mastra's Observer prompt for CEMS's cross-session use case.

```python
OBSERVER_SYSTEM_PROMPT = """You are the memory consciousness of a coding assistant.
Your observations will be stored as long-term memories and recalled across future sessions.

You are observing a coding session for project: {project_context}

Extract 1-3 high-level observations about what the user is doing, deciding, or learning.

## Rules

### Assertion vs Question
- User TELLS something → HIGH priority observation
  "I want to use Tailwind" → "User decided to use Tailwind CSS for styling"
- User ASKS something → MEDIUM priority, only store if significant
  "How do I deploy?" → only note if it reveals a knowledge gap pattern

### What to Observe
- Project goals and context ("User is building a memory system for AI agents")
- Decisions made ("User chose PostgreSQL over SQLite for production")
- Preferences expressed ("User prefers explicit over implicit error handling")
- State changes ("User switched from REST to GraphQL for the API")
- Key facts: names, dates, deadlines, project names, people mentioned
- Workflow patterns ("User always runs tests before committing")

### What NOT to Observe
- Implementation details (file paths, function signatures, exact commands)
- Transient debugging steps (unless they reveal a recurring pattern)
- Tool output (error messages, build logs)
- Routine operations (reading files, running tests)

### Format
Return JSON array:
[
  {
    "content": "User is building a production deployment pipeline for CEMS on Hetzner",
    "priority": "high",
    "category": "observation",
    "tags": ["observation", "deployment"]
  }
]

### Language
- Use present tense for ongoing activities: "User is building..."
- Use past tense for completed decisions: "User decided to..."
- Be specific: "CEMS memory system" not "the project"
- Include proper nouns: project names, service names, people
- Keep each observation under 200 characters
"""
```

### Why This Prompt Works Better Than Current Extraction

Current `extract_session_learnings` produces:
> `[WORKING_SOLUTION] Use docker compose build cems-server to rebuild after code changes (session: 5af1f05)`

Observer would produce:
> `User is overhauling CEMS memory quality — cleaning 2,499 memories down to 584, normalizing categories, raising relevance threshold`

The first is implementation documentation. The second is **context** — it tells future sessions what the user cares about right now.

---

## Component 4: Surfacing Observations

### In `UserPromptSubmit` Hook

The existing hook already searches CEMS and injects results as context. Observations will naturally appear in search results because they're stored in the same `memory_documents` table.

However, we should also fetch the **most recent observations** for the current project, regardless of search relevance:

```python
# In user_prompts_submit.py — after existing search

def fetch_recent_observations(project_id: str, limit: int = 5) -> list[str]:
    """Fetch recent observations for the current project."""
    response = httpx.post(
        f"{CEMS_API_URL}/api/memory/search",
        headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
        json={
            "query": f"project observations for {project_id}",
            "filters": {
                "category": "observation",
                "source_ref": f"project:{project_id}",
            },
            "limit": limit,
        },
        timeout=3.0,
    )
    if response.status_code == 200:
        results = response.json().get("results", [])
        return [r["memory"] for r in results]
    return []
```

Output format in the context injection:

```
## Recent Observations (project: chocksy/cems)
- User is overhauling CEMS memory quality — cleaning 2,499 memories down to 584
- User adopted Mastra's observational memory approach for better extraction
- User prefers Option D: hybrid hooks + standalone observer daemon
```

This gives Claude **project awareness** at the start of every prompt, even without search hits.

---

## Component 5: Server-Side Observation Extraction

### New file: `src/cems/llm/observation_extraction.py`

```python
def extract_observations(
    content: str,
    project_context: str,
    previous_count: int = 0,
) -> list[dict]:
    """Extract high-level observations from session content.

    Uses Gemini 2.5 Flash via OpenRouter for fast, cheap extraction.
    Returns list of observations with content, priority, category, tags.
    """
    client = get_client(model="google/gemini-2.5-flash")

    prompt = OBSERVER_SYSTEM_PROMPT.format(project_context=project_context)

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Session content to observe:\n\n{content}"},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    return parse_observations(response.choices[0].message.content)
```

### Model Choice

Mastra uses Gemini 2.5 Flash for both Observer and Reflector. We should too:
- **Cheap**: ~$0.001 per observation call
- **Fast**: 1-3 seconds
- **Good at extraction**: Flash excels at structured extraction from large context
- **Available via OpenRouter**: Already configured in CEMS

CEMS currently uses Grok 4.1 Fast for learning extraction. The Observer uses a different model for a different task — that's fine.

---

## Implementation Order

### Phase 1: Observer Prompt + API Endpoint (1 day)

Files to create/modify:
1. `src/cems/llm/observation_extraction.py` — new file, Observer prompt + extraction logic
2. `src/cems/api/handlers/observation.py` — new file, `/api/session/observe` endpoint
3. `src/cems/api/handlers/__init__.py` — export new handler
4. `src/cems/server.py` — register new route
5. Tests for extraction and endpoint

**Deliverable**: Can POST transcript content to `/api/session/observe` and get observations stored in `memory_documents`.

### Phase 2: Observer Daemon (1-2 days)

Files to create:
1. `src/cems/observer/__init__.py`
2. `src/cems/observer/daemon.py` — main polling loop
3. `src/cems/observer/session.py` — session discovery + metadata
4. `src/cems/observer/state.py` — per-session state tracking
5. `observer.py` — entry point script (or `python -m cems.observer`)

**Deliverable**: Background daemon that watches sessions and auto-triggers observations.

### Phase 3: Observation Surfacing (0.5 day)

Files to modify:
1. `hooks/user_prompts_submit.py` — add recent observation fetching
2. Possibly `hooks/cems_session_start.py` — inject project observations at session start

**Deliverable**: Observations appear in Claude's context on every prompt.

### Phase 4: Reflector / Consolidation (1 day, optional)

Files to modify:
1. `src/cems/maintenance/consolidation.py` — add observation consolidation logic
2. Merge old observations that overlap, soft-delete superseded ones

**Deliverable**: Observations stay manageable over time.

---

## Open Questions

1. **Should the daemon run on Mac or in Docker?**
   - Mac: Direct filesystem access to `~/.claude/projects/`. Simple.
   - Docker: Would need volume mount for `~/.claude/`. More isolated.
   - **Recommendation**: Mac daemon for now. It only reads files + makes HTTP calls.

2. **Should observations replace session learnings?**
   - No. They're complementary. Observations = context. Learnings = implementation details.
   - Over time, if observations prove more useful, we can reduce learning extraction.

3. **How to handle multi-project sessions?**
   - Some sessions span multiple repos (user switches `cwd`).
   - Observer reads `cwd` from each JSONL entry. If it changes mid-session, tag observations with the current project at that point.

4. **Token budget for observation extraction?**
   - 50KB of transcript ≈ 12-15k tokens input.
   - Max 2k tokens output (3 observations × ~200 chars each).
   - Cost: ~$0.001 per observation call via Gemini Flash.

5. **Should we use the hook-based approach (Option A) as a stepping stone?**
   - Could add observation extraction to `stop.py` immediately as a quick win.
   - Then build the daemon for mid-session observations.
   - **Recommendation**: Yes — start with stop.py, iterate to daemon.

---

## Quick Win: Observer in stop.py (Phase 0)

Before building the daemon, we can add observation extraction to the existing `stop.py` hook immediately:

```python
# In stop.py, after the existing analyze_session() call:

def observe_session(transcript, session_id, project, cwd):
    """Extract observations (not learnings) from the session."""
    # Compress transcript to text
    text_content = "\n".join(
        f"[{msg.get('type', 'unknown')}]: {extract_text(msg)}"
        for msg in transcript
        if msg.get("type") in ("user", "assistant")
    )[:100_000]  # Cap at ~25k tokens

    payload = {
        "content": text_content,
        "session_id": session_id,
        "source_ref": f"project:{project}" if project else None,
        "project_context": f"{project} ({cwd})",
    }

    # Fire-and-forget to CEMS
    req = urllib.request.Request(
        f"{CEMS_API_URL}/api/session/observe",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {CEMS_API_KEY}", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30)
```

This gives us observations on every session end with zero new infrastructure. The daemon adds mid-session observation later.

---

## Summary

| Component | Effort | Value | Dependency |
|-----------|--------|-------|------------|
| Observer prompt design | 0.5 day | **Critical** — this IS the system | None |
| `/api/session/observe` endpoint | 0.5 day | High | Prompt |
| Phase 0: stop.py integration | 0.5 day | High (quick win) | Endpoint |
| Observer daemon | 1-2 days | High (mid-session) | Endpoint |
| Observation surfacing in hooks | 0.5 day | High | Endpoint |
| Reflector/consolidation | 1 day | Medium | Observations exist |

**Total: ~4-5 days for full implementation, or 1.5 days for Phase 0 + 1 quick win.**
