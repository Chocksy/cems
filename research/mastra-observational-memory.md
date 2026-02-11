# Observational Memory: A Human-Inspired Memory System for AI Agents

**Source**: https://x.com/mastra/status/2021280193273336131
**Author**: Mastra (@mastra) — Tyler Barnes
**Date**: Feb 10, 2026
**Links**:
- Docs: https://mastra.ai/docs/memory/observational-memory
- Research: https://mastra.ai/research/observational-memory
- Source Code: `packages/memory/src/processors/observational-memory/` in mastra-ai/mastra
- Benchmark: https://github.com/mastra-ai/mastra/tree/main/explorations/longmemeval

---

## Summary

Mastra shipped a new type of memory for agentic systems: **observational memory**.

- Text-based (no vector/graph DB needed)
- SoTA on benchmarks like LongMemEval (94.87% with gpt-5-mini)
- Compatible with Anthropic/OpenAI/etc prompt caching
- Open-source TypeScript implementation (~7,000 lines)

---

## Part 1: How It Works

### Compressing Context to Observations

Your brain processes millions of pixels but distills down to one or two observations. Observational memory works the same way. A coding agent session compresses down to:

```
Date: 2026-01-15
🔴 12:10 User is building a Next.js app with Supabase auth, due in 1 week (meaning January 22nd 2026)
🔴 12:10 App uses server components with client-side hydration
🟡 12:12 User asked about middleware configuration for protected routes
🔴 12:15 User stated the app name is "Acme Dashboard"
```

### Three-Tier Memory Architecture

1. **Raw messages** — exact conversation history, keeps appending
2. **Observations** — when raw messages hit 30k tokens, Observer agent compresses to dated log
3. **Reflections** — when observations hit 40k tokens, Reflector agent garbage-collects

Context window has two blocks:
- Block 1: Observations (compressed, append-only log)
- Block 2: Raw messages (recent uncompressed)

### Core Design Principles

- **Formatted text, not structured objects.** No knowledge graphs, no vectors. Text is the universal interface.
- **Three-date model** for temporal reasoning: observation date, message timestamp, referenced date.
- **Emoji-based prioritization** (log levels): 🔴 important, 🟡 maybe important, 🟢 info only.
- **Prompt caching**: Append-only structure means stable prefixes = full cache hits on every turn.

---

## Part 2: Source Code Deep Dive

### File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `observational-memory.ts` | 5,628 | Main orchestration class |
| `observer-agent.ts` | 958 | Observer prompts, message formatting, output parsing |
| `reflector-agent.ts` | ~250 | Reflector prompts, compression logic |
| `token-counter.ts` | ~120 | Token counting (tiktoken o200k_base) |
| `types.ts` | ~200 | TypeScript type definitions |

### The Observer Agent

**System prompt core**: `"You are the memory consciousness of an AI assistant. Your observations will be the ONLY information the assistant has about past interactions with this user."`

**Three prompt variants** (A/B testing via env var):
- `CURRENT_OBSERVER_EXTRACTION_INSTRUCTIONS` (~200 lines, default) — comprehensive rules
- `CONDENSED_OBSERVER_EXTRACTION_INSTRUCTIONS` (~45 lines) — principle-based
- `LEGACY_OBSERVER_EXTRACTION_INSTRUCTIONS` (~60 lines) — Jan 2026 version

**Key extraction rules:**

1. **Assertion vs Question distinction** (critical rule):
   - User TELLS something → `🔴 (14:30) User stated has two kids`
   - User ASKS something → `🟡 (15:00) User asked help with X`
   - "USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own life."

2. **Temporal anchoring** — Two timestamps per observation:
   - Beginning: message time (ALWAYS)
   - End: referenced time if different ("last week" → estimated date)

3. **Detail preservation** (very specific):
   - Names, handles, @usernames
   - Numbers, quantities, measurements, prices
   - Exact phrasing when unusual ("movement session" not "exercise")
   - Sequences and orderings
   - User's specific role (presenter, not just "attended")
   - Verbatim text being collaborated on

4. **Precise action verbs**: Replace vague "getting"/"got" with "purchased"/"received"/"subscribed to"

5. **State change tracking**: "User will use the new method (replacing the old approach)"

**Output format** — XML-tagged:

```xml
<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User stated they have 3 kids: Emma (12), Jake (9), and Lily (5)
* 🟡 (14:33) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass
</observations>

<current-task>
Primary: Implementing OAuth2 flow for the auth refactor
Secondary: Waiting for user to confirm database schema changes
</current-task>

<suggested-response>
The OAuth2 implementation is ready for testing. Would you like me to walk through the flow?
</suggested-response>
```

**Guidelines**: "Add 1 to 5 observations per exchange. Use terse language to save tokens."

### The Reflector Agent

**System prompt core**: `"You are the memory consciousness of an AI assistant. Your memory observation reflections will be the ONLY information the assistant has about past interactions with this user."`

Receives the Observer's extraction instructions to understand observation format.

**Three-level compression guidance:**
- Level 0: No guidance (first attempt)
- Level 1: "Aim for 8/10 detail level" — gentle compression
- Level 2: "Aim for 6/10 detail level" — aggressive compression

**Retry logic**: If reflected tokens still exceed threshold after Level 0, retry Level 1, then Level 2. `validateCompression()` checks output < target.

**Key behaviors:**
- Re-organize and streamline, draw connections and conclusions
- Condense older observations more aggressively, retain more detail for recent ones
- "CRITICAL: your reflections are THE ENTIRETY of the assistant's memory. Any information you do not add to your reflections will be immediately forgotten."
- Maintains thread attribution via `<thread id="...">` sections

### Configuration Defaults

```typescript
OBSERVATIONAL_MEMORY_DEFAULTS = {
  observation: {
    model: 'google/gemini-2.5-flash',    // NOT Claude 4.5
    messageTokens: 30_000,                // Trigger threshold
    modelSettings: {
      temperature: 0.3,
      maxOutputTokens: 100_000,
    },
    providerOptions: {
      google: { thinkingConfig: { thinkingBudget: 215 } },
    },
    maxTokensPerBatch: 10_000,
    bufferTokens: 0.2,                    // Buffer every 20% (6k tokens)
    bufferActivation: 0.8,                // Activate at 80% of threshold
  },
  reflection: {
    model: 'google/gemini-2.5-flash',
    observationTokens: 40_000,
    modelSettings: {
      temperature: 0,                     // Zero for consistency
      maxOutputTokens: 100_000,
    },
    providerOptions: {
      google: { thinkingConfig: { thinkingBudget: 1024 } },
    },
    bufferActivation: 0.5,                // Start at 50% (20k tokens)
  },
}
```

### Context Optimization for the Actor

Before presenting to the agent, `optimizeObservationsForContext()`:
- Strips 🟡 and 🟢 emojis (keeps only 🔴)
- Removes semantic tags and arrow indicators
- Full format preserved in storage

At injection time, `addRelativeTimeToObservations()` adds relative time to date headers:
`"Date: May 15, 2023 (5 days ago)"`

### Async Buffering System

- **Sync mode**: Blocks conversation at threshold
- **Async buffering** (default): Background observation at intervals
  - `bufferTokens: 0.2` = every 20% of threshold (every 6k tokens)
  - Chunks accumulated in `bufferedObservationChunks`
  - Activated at `bufferActivation: 0.8` = 80% of threshold
  - Reflection also has async buffering at 50% of its threshold
- In-memory locks serialize operations per thread/resource

---

## Part 3: Benchmark Results

### LongMemEval Scores

**GPT-5-Mini (94.87% — SoTA):**

| Category | Score |
|----------|-------|
| Knowledge-update | 96.2% |
| Multi-session | 87.2% |
| Single-session-assistant | 94.6% |
| Single-session-preference | 100% |
| Single-session-user | 95.7% |
| Temporal-reasoning | 95.5% |

**GPT-4o (84.23%):**

| Category | Score |
|----------|-------|
| Knowledge-update | 85.9% |
| Multi-session | 79.7% |
| Single-session-assistant | 82.1% |
| Single-session-preference | 73.3% |
| Single-session-user | 98.6% |
| Temporal-reasoning | 85.7% |

### Comparative Scores (all gpt-4o)

| System | Score |
|--------|-------|
| Mastra OM | 84.23% |
| Oracle (evidence-only sessions) | 82.4% |
| Supermemory | 81.60% |
| Mastra RAG (semantic recall) | 80.05% |
| Zep | 71.20% |
| Full context baseline | 60.20% |

### Scaling Advantage

Supermemory improved 3.6 points gpt-4o → gemini-3-pro. Mastra OM gained **9 points** — better leverage of model improvements.

### Known Limitations

1. **Multi-session ceiling at 87.2%** — both Mastra OM and Hindsight hit this wall
2. **Single-session-preference volatility** — only 30 questions (each = 3.3% swing)
3. **Claude 4.5 NOT recommended** as Observer/Reflector (no explanation given)
4. **No knowledge discovery** — only recalls what agent has seen, not open-ended search
5. **Resource scope slowness** — processing unobserved messages across many threads

---

## Part 4: CEMS Adaptation Analysis

### Current CEMS Architecture vs Mastra OM

| Aspect | CEMS Current | Mastra OM |
|--------|-------------|-----------|
| **When observations happen** | Session end OR auto-compact OR per-tool-use | Token threshold (30k raw messages) |
| **What's extracted** | Granular learnings (code patterns, error fixes) | High-level observations (what user is doing) |
| **Storage** | Embeddings in `memory_documents` + vector search | Plain text in context window |
| **Retrieval** | Vector/hybrid search (~5 results per prompt) | Full text in context (no search needed) |
| **Context management** | Search results injected per-prompt | Observations + raw messages blocks |
| **LLM for extraction** | Grok 4.1 Fast | Gemini 2.5 Flash |

### The Fundamental Architectural Mismatch

Mastra OM is an **in-process memory layer** managing blocks INSIDE the agent's context window. CEMS is an **external memory server** accessed via REST API hooks.

CEMS hooks can:
- Inject context (SessionStart, UserPromptSubmit output)
- Read transcripts (all hooks receive `transcript_path`)
- Send data to server (fire-and-forget API calls)

CEMS **cannot**:
- Manage Claude's context window blocks
- Replace raw messages with observations
- Control when compaction happens
- Restructure conversation history

### What CEMS Should Adopt

#### 1. Observer-Style Periodic Observations (HIGH VALUE)

Instead of only extracting learnings at session end, run observation mid-session.

**Hook point**: `UserPromptSubmit` — track transcript size, trigger when it grows past ~50KB.

**New endpoint**: `POST /api/session/observe`
- Receives recent transcript portion + previous observations
- Observer extracts high-level observations (not granular learnings)
- Stored as `memory_documents` with `category="observation"`
- Surfaced alongside search results on subsequent prompts

**Key difference from current stop.py**: Current session analysis extracts implementation-level learnings ("use docker compose build to rebuild"). Observer would extract context-level observations ("User is doing a production deployment of CEMS memory quality overhaul").

#### 2. Observer Prompt Adaptation (HIGH VALUE)

Adopt Mastra's Observer prompt philosophy with CEMS-specific modifications:
- Assertion vs Question distinction
- Temporal anchoring with three-date model
- Detail preservation rules (names, numbers, exact phrasing)
- Priority levels mapping to confidence scores
- State change tracking

#### 3. Three-Date Temporal Model (MEDIUM VALUE)

Add `referenced_date` column to `memory_documents`. Observer extracts temporal references ("user said deadline is Jan 22") and stores them for temporal-aware retrieval.

#### 4. Reflector-Style Garbage Collection (LOW-MEDIUM VALUE)

New maintenance job that periodically:
- Finds observations from same project that are semantically similar
- Merges/consolidates them
- Marks superseded ones as soft-deleted

Already partially exists via time decay + shown_count feedback.

### What CEMS Should NOT Adopt

1. **Text-only storage** — CEMS's vector search is fundamental to cross-session retrieval. Keep embeddings.
2. **In-context observation blocks** — Can't manage Claude's context window from hooks.
3. **Synchronous observation** — Hooks must be fast. Observation should be async/fire-and-forget.
4. **Dropping semantic search** — Mastra OM addresses within-session memory. CEMS addresses cross-session memory. Different problems.

### Concrete Implementation Plan

**Phase 1: Observer Hook (2-3 days)**

Files to change:
- `hooks/user_prompts_submit.py` — transcript size tracking + observation trigger
- `src/cems/api/handlers/session.py` — new `api_session_observe` endpoint
- `src/cems/llm/learning_extraction.py` — new `extract_observations()` function
- `src/cems/server.py` — register new route

Key code pattern:
```python
# In user_prompts_submit.py — track transcript growth
def should_observe(transcript_path: str, session_id: str) -> bool:
    state = get_observation_state(session_id)
    current_size = Path(transcript_path).stat().st_size
    delta = current_size - state["last_observed_bytes"]
    return delta > 50_000  # ~12-15k tokens of new content
```

**Phase 2: Observation Surfacing (1 day)**

Modify `user_prompts_submit.py` to fetch recent observations alongside search results.

**Phase 3: Reflector Maintenance (2 days)**

New weekly maintenance job that consolidates old observations.

### Session Transcript Access from Hooks

All Claude Code hooks receive the transcript via `input_data["transcript_path"]`. This is a `.jsonl` file where each line is a message with role, content, timestamp, tool calls, etc.

The key insight: hooks CAN read the full session context. Currently:
- `stop.py` reads full transcript at session end
- `pre_compact.py` reads full transcript before auto-compaction
- `user_prompts_submit.py` only reads last few messages for context

To implement Observer-style observation, `user_prompts_submit.py` would need to:
1. Read the full transcript (or recent portion since last observation)
2. Check if enough new content has accumulated (50KB threshold)
3. Fire-and-forget to `/api/session/observe` with the new content
4. Track observation state in a per-session file

### Key Insight

**The prompts ARE the system.** 90% of Mastra OM's value is in the Observer/Reflector prompt engineering — the assertion vs question distinction, temporal anchoring rules, detail preservation, verb precision. The infrastructure (token counting, async buffering, context management) is plumbing around very carefully crafted prompts.

For CEMS, this means: the biggest win is **changing what we ask the LLM to extract**, not necessarily changing the storage or retrieval architecture. Our current `learning_extraction.py` prompts extract granular implementation details. Adding an Observer-style prompt that extracts high-level observations would complement the existing system without replacing it.

---

## Part 5: CEMS vs Mastra — Two Different Problems

| | CEMS | Mastra OM |
|---|---|---|
| **Primary use case** | Cross-session memory for coding agent | Within-session context management |
| **Architecture** | External server + hooks | In-process memory layer |
| **Strength** | Unlimited history, semantic search | Bounded context, prompt caching |
| **LongMemEval** | 98% Recall@5 (50 questions) | 94.87% (500 questions, gpt-5-mini) |
| **Best for** | "What did I do last week?" | "What are we working on right now?" |

The two systems are **complementary**, not competitive. CEMS excels at long-term cross-session knowledge. Mastra OM excels at within-session context compression. The ideal system would use both: observations for current session context + vector search for historical knowledge.
