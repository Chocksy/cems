# CEMS Knowledge Engine

**Date**: 2026-04-06  
**Status**: Brainstorm  
**Inspired by**: [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), [Graphify](https://github.com/safishamsi/graphify)

## What We're Building

Expand CEMS from a flat memory store into a compounding knowledge engine by adding three layers:

1. **Auto-Relations** — automatically link related memories on every add
2. **Lint & Contradictions** — detect conflicts, stale claims, orphans, and knowledge gaps
3. **Entity Pages** — LLM-compiled synthesis documents that combine knowledge from memory clusters

Plus a **wiki-style dashboard** for browsing, exploring, and resolving knowledge health issues.

## Why This Approach

Karpathy's insight: knowledge should be synthesized once and maintained, not re-derived per query. Graphify proves the graph-based navigation works (71.5x token savings), but uses in-memory NetworkX — not suitable for multi-user teams.

CEMS already has the infrastructure (PostgreSQL + pgvector, chunking pipeline, maintenance jobs, `memory_relations` table, multi-user support). The gap is: nothing populates relations, nothing compiles knowledge into higher-order pages, and nothing detects contradictions.

**Build order: bottom-up.** Relations → Lint → Entity Pages → Dashboard. Each layer ships independent value.

## Key Decisions

### 1. Relations are populated on every `add_async()`
After chunking + embedding, do a similarity search against existing memories. Top matches above cosine > 0.75 get `memory_relations` rows. Types: `similar`, `extends`, `contradicts`.

**Cost**: One extra pgvector search per add (fast, already have the embedding).

### 2. Entity pages are regular `memory_documents`
Category: `entity-page`. Source: `compiler`. Linked to source memories via `memory_relations` with `relation_type='compiled_from'`. They participate in search naturally.

**Evolution**: When new memories join a cluster, entity page is flagged stale → recompiled on next maintenance run.

### 3. Contradictions are flagged, not auto-resolved
Detected inline (on add, when similar memory found) and periodically (lint job). Surfaced in dashboard with resolution actions: keep newer, merge, dismiss. Human decides.

### 4. Three-tier recall
Entity page summary → entity full page → individual source memories. LLM gets entity summary first (high-signal, pre-synthesized), drills into sources only if needed.

### 5. Lint is a new maintenance job
Checks for: contradictions, stale claims, orphan memories, knowledge gaps (mentioned but unsynthesized concepts), missing cross-references. Outputs actionable report.

## Architecture

```
Memory Added (hooks / API / MCP / indexer)
    │
    ▼
Chunk + Embed + Store (existing)
    │
    ▼
Auto-Link Relations (NEW)
  - similarity search against existing memories
  - populate memory_relations
    │
    ▼
Contradiction Detection (NEW, inline)
  - quick LLM check on highly similar pairs
  - flag relation_type='contradicts'
    │
    ▼ (async, batched)
Entity Page Compilation (NEW maintenance job)
  - cluster related memories (connected components or Leiden)
  - LLM synthesizes cluster → entity page
  - stored as memory_documents with category='entity-page'
```

### Dashboard Additions
- **Browse view**: Entity pages as wiki articles, navigable by concept links
- **Graph view**: Visual knowledge graph, hubs prominent, orphans visible
- **Lint panel**: Actionable health report with resolution buttons
- **Search**: Entity pages surface as first-class results

## What CEMS Already Has vs. What's New

| Capability | Status |
|-----------|--------|
| Document storage + chunking | Existing |
| Vector embedding + search | Existing |
| `memory_relations` table | Existing (empty) |
| Consolidation/summarization | Existing |
| Multi-user / multi-project | Existing |
| **Auto-relation population** | **NEW** |
| **Contradiction detection** | **NEW** |
| **Entity page compilation** | **NEW** |
| **Lint job** | **NEW** |
| **Wiki dashboard** | **NEW** |

## How It Handles Scale

- **Small** (<500 memories): Entity pages + index sufficient
- **Medium** (500-5000): Existing pgvector search + entity page summaries as first-tier recall
- **Large** (5000+): Summarization/distillation prune low-value. Entity pages *replace* the need to surface many individual memories. Three-tier recall keeps LLM context small.
- Entity pages are always derived artifacts — recompilable from sources, safe to regenerate.

## Comparison to Alternatives

| | CEMS (proposed) | Graphify | Karpathy Wiki |
|---|---|---|---|
| Storage | PostgreSQL + pgvector | NetworkX JSON | Markdown files |
| Multi-user | Yes | No | No (git collab) |
| Contradictions | Detect + surface | AMBIGUOUS tag only | Flag in lint |
| Scale strategy | DB + tiered recall | In-memory (limited) | Index + qmd |
| Code awareness | AST extractors | tree-sitter AST | None |
| Maintenance | 5+ automated jobs | SHA256 cache | Manual lint |

## Deep Dive: How Each Piece Actually Works

### How Observations & Session Summaries Fit In

The observer daemon (`src/cems/observer/daemon.py`) already watches Claude Code and Cursor sessions,
extracts observations, and stores them as `category='session-summary'` documents. The
`ObservationReflector` maintenance job consolidates overlapping session summaries per project.

**In the knowledge engine, session summaries are first-class source material:**
- They get auto-linked to related memories on store (like any other document)
- They feed into entity pages — a session that touches "Stripe integration" links to the
  Stripe entity page and enriches it on next compilation
- They're high-value signals for contradiction detection — "in this session we decided X"
  vs an older memory saying "we always do Y"

**No changes needed to the observer.** It already produces documents that flow through
`add_async()`. The new layers (relations, lint, compilation) apply to ALL documents
regardless of source. Observer memories just become richer because they're now connected.

### Auto-Link Relations: Exact Mechanics

**NOT just a DB call — it's a vector search + DB insert.**

Current `add_async()` flow (`src/cems/memory/write.py:86-198`):
1. Chunk content → 2. Embed chunks → 3. Store document + chunks → 4. Return

**New step 3.5 — find and link neighbors:**
```python
# After storing doc, use the first chunk's embedding to find neighbors
neighbors = await doc_store.search_chunks(
    embedding=embeddings[0],  # Already computed in step 2!
    user_id=user_id,
    limit=10,
    score_threshold=0.75,
)
# Insert relations for top matches (excluding self)
for neighbor in neighbors:
    if neighbor["document_id"] != doc_id:
        await doc_store.add_relation(
            source_id=doc_id,
            target_id=neighbor["document_id"],
            relation_type="similar",
            similarity=neighbor["score"],
        )
```

**Key insight: we already have the embedding from step 2.** No extra embedding call.
The only new cost is one pgvector search (~5ms) and a few INSERT statements.

**Currently missing:** `add_relation()` method in `DocumentStore`. The table exists,
`get_related_documents()` exists (`document_store.py:1366`), but there's no write method.
We need to add it.

### Contradiction Detection: How It's Determined

**Two-stage approach:**

**Stage 1 — Cheap filter (on every add, ~5ms)**:
When auto-linking finds a neighbor with cosine > 0.85 (very similar), check if the
documents have opposing signals. Simple heuristic:
- Same category + same source_ref + high similarity = probably related, not contradictory
- Different source/time + high similarity = contradiction candidate
- Flag as `relation_type='potential_contradiction'` for Stage 2

**Stage 2 — LLM verification (batched, in lint job)**:
For all `potential_contradiction` relations, send both documents to an LLM:
```
"Do these two memories contradict each other? 
Memory A: {content_a}
Memory B: {content_b}
Answer: CONTRADICTS / EXTENDS / AGREES / UNRELATED"
```
If CONTRADICTS → update to `relation_type='contradicts'`, surface in dashboard.

**Why not LLM on every add?** Cost and latency. An LLM call per add would add 1-3 seconds
and $0.001-0.01 per memory. Batching in the lint job means we can use a cheap model
(Gemini Flash) and process 50 candidates in one batch call.

### Handling Massive Data (Scale Validation Plan)

**Current prod stats** (from MEMORY.md): ~2,580 memories after maintenance sweeps.
That's modest. But for enterprise use, think 50K-500K memories.

**Why pgvector handles this:**
- pgvector with IVFFlat index: ~10ms search at 100K vectors, ~50ms at 1M
- Auto-relations: one search per add = negligible overhead
- Relations table: simple foreign key joins, indexed
- Entity pages: one per cluster, not per memory — maybe 100-500 entity pages for 50K memories

**What we need to validate (test plan):**
1. **Dump prod DB** → load into local Docker
2. **Measure baseline**: search latency, add latency, maintenance job duration
3. **Add auto-relations to all existing memories** (backfill job)
4. **Measure with relations**: does search slow down? How many relations per memory?
5. **Run compilation on clusters**: how long? How many entity pages generated?
6. **Scale test**: synthetically duplicate memories to 50K, re-run measurements

**Test environment:** Local Docker (`docker compose up -d`), dump with:
```bash
pg_dump -h <prod-host> -U cems cems_db > prod_dump.sql
# Load into local Docker postgres
docker exec -i cems-postgres psql -U cems cems_db < prod_dump.sql
```

### Memory Protection: The "Brain" Model

**The problem:** If maintenance jobs keep running, won't entity pages and important
memories get pruned/summarized away?

**Current protections (verified in code):**
- `PROTECTED_CATEGORIES` in `maintenance/__init__.py`: `gate-rules`, `guidelines`,
  `preferences`, `category-summary`, `session-summary` are exempt from pruning
- `shown_count` / `relevant_count` / `noise_count` track usage feedback
- Summarization only compresses memories 14+ days old

**What we need for the knowledge engine:**

**Entity pages are self-healing.** Even if an entity page gets deleted, it can be
recompiled from source memories. They're derived, not primary. Mark them with
`category='entity-page'` and add to `PROTECTED_CATEGORIES`.

**For source memories — the "importance signal" is already there:**
- `shown_count` = how often this memory is surfaced to users
- `relevant_count` = how often users marked it as relevant
- Memories with high shown_count + high relevant_count are "hot" memories

**The brain analogy is exactly right.** Here's how to implement it:

```
┌─────────────────────────────────────────────┐
│              HOT MEMORIES                     │
│  High shown_count + relevant_count           │
│  → NEVER summarized/pruned                   │
│  → Fast access (always in search results)    │
│  → These are your "working memory"           │
├─────────────────────────────────────────────┤
│              WARM MEMORIES                    │
│  Moderate usage, <14 days old                │
│  → Normal search, normal ranking             │
│  → These are your "short-term memory"        │
├─────────────────────────────────────────────┤
│              COOL MEMORIES                    │
│  Low usage, 14-60 days old                   │
│  → Compressed into category summaries        │
│  → Originals soft-deleted (recoverable)      │
│  → These are your "long-term memory"         │
├─────────────────────────────────────────────┤
│              COLD MEMORIES (soft-deleted)     │
│  Zero usage, 60+ days old                    │
│  → Not in search results                     │
│  → Still in DB (recoverable)                 │
│  → Entity pages preserve their knowledge     │
│  → Searchable via "deep recall" mode         │
└─────────────────────────────────────────────┘
```

**The key addition**: a `heat_score` computed from shown_count, relevant_count, and recency:
```python
heat_score = (shown_count * 2 + relevant_count * 5) / max(age_days, 1)
```
Memories above a heat threshold are NEVER compressed. This means:
- Memories that keep getting surfaced and used stay hot indefinitely
- Memories nobody ever looks at gradually cool and compress
- Entity pages preserve the knowledge even from cold memories
- "Deep recall" mode searches soft-deleted memories too (longer, but finds everything)

### How This Differs from Karpathy's System

Karpathy's wiki is **human-curated** — the human decides what to file back, what to lint.
His wiki is a **single-user tool** with Obsidian as frontend.

CEMS Knowledge Engine is **automatically curated** — the system detects what's important
based on usage patterns, auto-builds entity pages, auto-detects contradictions. It's
**multi-user** with a web dashboard. And it evolves continuously through the observer
daemon — every coding session automatically enriches the knowledge graph.

The evolution model:
1. **Developer works** → observer captures session summary → stored as memory
2. **Memory auto-links** to related memories → graph grows
3. **Clusters form** → compilation job generates entity pages
4. **Lint detects issues** → contradictions surfaced in dashboard
5. **Team browses wiki** → high-traffic pages stay hot, low-traffic pages compress
6. **New developer joins** → entity pages give them instant context

## Deep Dive: Heat Score & Memory Tiers

### What Already Exists (retrieval.py:693-744)

The scoring system already has most of the signals:
- **Time decay**: `1.0 / (1.0 + (days_since_access / half_life))`
- **Category-aware half-lives**: 21d (session/tool), 60d (general), 120d (core)
- **Relevance feedback**: `0.50 + 0.70 * relevance_ratio` applied to time_decay
- **Noise penalty**: snippet-level noise caps at 20% reduction
- **Project boost/penalty**: 1.3x same-project, 0.8x different

The docstring at line 706 says "memories shown 10+ times get min 0.95 decay" but this
**is NOT implemented** — there's no `if shown_count >= 10` guard in the actual code.
This is the gap we need to fill.

### The Heat Score Formula

Heat score is a single number that determines a memory's protection tier:

```python
def compute_heat_score(doc: dict) -> float:
    """Compute memory heat score from usage signals.
    
    Range: 0.0 (ice cold, never used) to ~100+ (red hot, constantly used).
    """
    shown = doc.get("shown_count", 0)
    relevant = doc.get("relevant_count", 0)
    noise = doc.get("noise_count", 0)
    age_days = max((now - doc["created_at"]).days, 1)
    
    # Usage intensity (shown_count is the base signal)
    usage = shown * 1.0
    
    # Quality multiplier (relevant memories are worth more)
    if shown > 0:
        quality = 1.0 + (relevant - noise) / shown  # Range: 0.0 to 2.0
    else:
        quality = 0.5  # Unknown quality
    
    # Recency boost (recently active memories are hotter)
    last_shown_days = max((now - doc.get("last_shown_at", doc["created_at"])).days, 1)
    recency = 30.0 / (30.0 + last_shown_days)  # 1.0 if shown today, 0.5 at 30d
    
    # Relation count boost (well-connected memories are more valuable)
    relation_count = doc.get("relation_count", 0)  # NEW field
    connectivity = 1.0 + min(relation_count * 0.1, 1.0)  # Up to 2x boost
    
    return usage * quality * recency * connectivity
```

### Heat Tiers and Protection Rules

| Tier | Heat Score | Protection | Maintenance Behavior |
|------|-----------|------------|---------------------|
| **HOT** | > 10.0 | **Immune** | Never summarized, never compressed. Time decay floored at 0.95. |
| **WARM** | 2.0 - 10.0 | **Resistant** | Time decay floored at 0.80. Can be consolidated but not deleted. |
| **COOL** | 0.5 - 2.0 | **Normal** | Current behavior. Summarized at 14d, soft-deleted when summary exists. |
| **COLD** | < 0.5 | **Expendable** | Aggressively summarized at 7d. Soft-deleted. Knowledge preserved in entity pages. |

### How This Integrates with Existing Scoring

In `apply_score_adjustments()` at `retrieval.py:728`:

```python
time_decay = 1.0 / (1.0 + (days_since_access / half_life))

# NEW: Heat-based floor (implement the documented but missing adaptive ceiling)
heat = compute_heat_score(result)
if heat > 10.0:
    time_decay = max(time_decay, 0.95)  # Hot: almost no decay
elif heat > 2.0:
    time_decay = max(time_decay, 0.80)  # Warm: slow decay
# Cool and Cold: no floor, current behavior
```

### How This Integrates with Maintenance

In `SummarizationJob._compress_by_category()`:

```python
# Before compressing, check heat
hot_docs = [d for d in docs if compute_heat_score(d) > 2.0]
cold_docs = [d for d in docs if compute_heat_score(d) <= 2.0]
# Only compress cold docs. Hot/warm docs stay as individual memories.
```

### The Brain Analogy Made Precise

Think about how your brain works:
- You **instantly recall** things you use every day (hot). No searching needed.
- You **quickly find** recent things or things you care about (warm). Small effort.
- You **can remember** older things if prompted (cool). Takes a moment, might be fuzzy.
- You **vaguely know** you knew something once (cold). "I read about this but can't remember details." The entity page is that vague shape.
- You've **forgotten** entirely (deleted). But someone else on the team might still have it.

**Deep recall mode**: When normal search returns < 3 results, automatically expand to
include soft-deleted memories. Or the user can toggle "deep recall" in the dashboard.
This makes cold memories accessible without polluting normal search.

### Entity Pages as "Compiled Knowledge"

Entity pages are the key to making cold storage work. When 20 individual memories about
"Stripe integration" cool down and compress, the entity page STILL has their knowledge:

```
Entity Page: Stripe Integration
Compiled from 20 memories (8 hot, 5 warm, 7 cold)

Summary: [high-quality synthesis of all 20 memories]
Key facts: [extracted from all sources]
Contradictions: [any flagged conflicts]
Sources: [links to all 20 memories, indicating tier]

Last compiled: 2026-04-05
Heat: 45.0 (HOT - frequently accessed)
```

Even if the 7 cold source memories are soft-deleted, the entity page preserves their
knowledge. And the entity page itself stays hot because it keeps getting surfaced in search.

### What About Manually Pinned Memories?

We already have pinning (indexer sets priority 1.5-2.0 for pinned docs). Pinned memories
should bypass the heat score entirely — they're HOT by declaration, not by usage.

```python
if doc.get("source") == "indexer" or doc_is_pinned(doc):
    return float('inf')  # Always hot
return compute_heat_score(doc)
```

## Deep Dive: Dashboard Wiki Vision

### Current Dashboard (static/dashboard/)

Vanilla JS, single-page app. Features:
- Login with API key
- Memory list (paginated, searchable)
- Category filters (pills)
- Scope toggle (All / Personal / Team)
- View toggle (List / Projects chart)
- Edit modal (content, category, tags, source_ref)
- Chart.js for project distribution

**It's a flat list viewer.** No relationships, no navigation between memories, no health info.

### Wiki Dashboard: The New Views

#### 1. Entity Page View (the "Wikipedia" experience)

```
┌─────────────────────────────────────────────────────┐
│  CEMS Knowledge Engine           [Search...] [User] │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│ Topics   │  # Stripe Integration                    │
│          │  Heat: ████████░░ HOT                    │
│ ● Stripe │  Updated: 2 hours ago | 20 sources      │
│ ● Auth   │                                          │
│ ● Deploy │  ## Overview                             │
│ ● Docker │  The Stripe integration uses webhooks    │
│ ● Rails  │  for payment confirmation. The account   │
│ ● API    │  is very large — full scans can run for  │
│          │  hours...                                │
│ ──────── │                                          │
│ Recently │  ## Key Decisions                        │
│ accessed │  - Keep API requests sampled by default  │
│          │  - Webhook endpoint at /api/webhooks     │
│ ○ OAuth  │                                          │
│ ○ CI/CD  │  ## Related Topics                       │
│ ○ Redis  │  → Payment Processing                    │
│          │  → Webhook Handlers                      │
│          │  → API Rate Limiting                     │
│          │                                          │
│          │  ## Source Memories                       │
│          │  🔥 "Keep Stripe requests sampled..."     │
│          │  🔥 "Webhook endpoint confirmed at..."    │
│          │  ❄️ "Initial Stripe setup notes from..."  │
│          │                                          │
│          │  ⚠️ 1 Contradiction                       │
│          │  "Docker images built on Mac" vs          │
│          │  "NEVER build Docker on Mac"              │
│          │  [Keep newer] [Merge] [Dismiss]          │
│          │                                          │
└──────────┴──────────────────────────────────────────┘
```

**Left sidebar**: Entity pages sorted by heat score. Hot topics at top. Expandable tree
by project or category. Recently accessed section for quick navigation.

**Main area**: Wiki article. Rendered markdown with:
- Heat indicator (visual bar)
- Overview (LLM-synthesized summary)
- Key decisions extracted from source memories
- Related topics (clickable links to other entity pages)
- Source memories with heat indicators (🔥 hot, 🌡️ warm, ❄️ cold)
- Contradiction alerts with inline resolution actions

#### 2. Graph View (the "Obsidian" experience)

```
┌─────────────────────────────────────────────────────┐
│  Graph View                    [Filter] [Zoom] [3D] │
├─────────────────────────────────────────────────────┤
│                                                     │
│        ┌──────┐                                     │
│        │Stripe│────────┐                            │
│        └───┬──┘        │                            │
│            │       ┌───┴────┐                       │
│        ┌───┴──┐    │Payments│                       │
│        │Webhks│    └───┬────┘                       │
│        └──────┘        │                            │
│                    ┌───┴──┐     ┌──────┐            │
│                    │ API  │─────│Docker│            │
│                    └───┬──┘     └──┬───┘            │
│                        │          │                 │
│                    ┌───┴──┐   ┌───┴──┐              │
│                    │ Auth │   │Deploy│              │
│                    └──────┘   └──────┘              │
│                                                     │
│  ○ 152 nodes  ─ 347 edges  ⚠ 3 contradictions      │
│  Legend: ● HOT  ◉ WARM  ○ COOL  ◌ COLD             │
└─────────────────────────────────────────────────────┘
```

- **Node size** = heat score (bigger = hotter)
- **Node color** = tier (red=hot, orange=warm, blue=cool, gray=cold)
- **Edge thickness** = similarity score
- **Red edges** = contradictions
- **Click a node** → opens entity page in split view
- **Hover** → shows memory snippet and connection count
- **Filter by**: project, category, heat tier, date range
- **Clustering**: toggle community detection to see topic clusters

**Library choice**: D3.js force-directed graph (we already have Chart.js loaded, D3 is
the natural step up). Or sigma.js for WebGL rendering at 10K+ nodes.

#### 3. Lint Panel (the "Health Check" experience)

```
┌─────────────────────────────────────────────────────┐
│  Knowledge Health              Score: 87/100 ████▓░ │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ⚠ CONTRADICTIONS (3)                    [Fix all]  │
│  ┌─────────────────────────────────────────────┐    │
│  │ "Docker images built on Mac"                │    │
│  │ vs "NEVER build Docker on Mac"              │    │
│  │ [Keep newer] [Merge] [Dismiss]              │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  📭 ORPHAN MEMORIES (12)              [Review all]  │
│  │ 12 memories with zero connections               │
│  │ [Auto-archive] [Connect manually]               │
│                                                     │
│  🔍 KNOWLEDGE GAPS (5)               [Generate all] │
│  │ "OAuth flow" mentioned 5x but no entity page    │
│  │ "Redis caching" mentioned 3x but no entity page │
│  │ [Generate entity page]                          │
│                                                     │
│  📊 STALE ENTITY PAGES (2)           [Recompile]   │
│  │ "Auth Middleware" — 15 new memories since last   │
│  │ "Testing Patterns" — 8 new memories since last  │
│  │ [Recompile now]                                 │
│                                                     │
│  ✅ HEALTHY                                         │
│  │ 245 memories well-connected                     │
│  │ 38 entity pages up-to-date                      │
│  │ 98% of knowledge covered by entity pages        │
└─────────────────────────────────────────────────────┘
```

**Health score formula**:
```
100 - (contradictions * 5) - (orphans * 0.5) - (gaps * 2) - (stale_pages * 1)
```

Each issue has **inline resolution actions** — one click to fix. The lint panel is the
"control tower" for knowledge quality.

#### 4. Memory Timeline (the "Git Log" experience)

```
┌─────────────────────────────────────────────────────┐
│  Timeline: Stripe Integration                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Apr 5 ─── 🔥 "Webhook endpoint confirmed at /api" │
│         │     session: claude-abc123                 │
│  Apr 3 ─── 🔥 "Keep Stripe requests sampled"       │
│         │     source: user-preference                │
│  Mar 28 ── 🌡️ "Stripe test mode key rotation"       │
│         │     session: cursor-def456                 │
│  Mar 15 ── ❄️ "Initial Stripe setup notes"           │
│         │     source: repo-indexer                   │
│  Feb 20 ── ❄️ "Payment flow discussion"              │
│              session: claude-ghi789                  │
│                                                     │
│  ◆ Entity page compiled (20 sources)                │
│  ◆ 1 contradiction detected                        │
│  ◆ 3 memories consolidated                         │
└─────────────────────────────────────────────────────┘
```

Shows the evolution of a topic over time. Each memory shows its source (observer session,
user-added, indexer). Entity page compilation events and lint events appear on the timeline too.

### Dashboard Architecture

```
static/dashboard/         (existing: list view)
static/wiki/              (NEW: entity page + graph + lint + timeline views)
  index.html
  app.js
  components/
    entity-page.js        (wiki article renderer)
    graph-view.js         (D3 force-directed graph)
    lint-panel.js         (health check UI)
    timeline.js           (memory evolution timeline)
    sidebar.js            (topic navigation)
  lib/
    d3.min.js             (graph rendering)
```

**API endpoints needed:**
- `GET /api/wiki/entities` — list all entity pages with heat scores
- `GET /api/wiki/entity/:id` — entity page with source memories and relations
- `GET /api/wiki/graph` — nodes + edges for graph visualization
- `GET /api/wiki/lint` — lint report with actionable issues
- `POST /api/wiki/lint/resolve` — resolve a lint issue (dismiss, merge, keep)
- `GET /api/wiki/timeline/:entity_id` — memory timeline for an entity

These are thin wrappers over existing DocumentStore queries + the new relations data.

## Open Questions

1. **Relation threshold**: What cosine similarity threshold for auto-linking? Too low = noise, too high = sparse graph. Need experimentation (start at 0.75, test with prod data).
2. **Clustering algorithm**: Simple connected components, or Leiden for better community detection? Connected components is simpler but might create mega-clusters. Start simple, upgrade if needed.
3. **Entity page regeneration frequency**: Batched in maintenance (not on every add). Probably daily or on-demand.
4. **LLM cost of compilation**: Each entity page = one LLM call reading N memories. Use Gemini Flash for compilation (~$0.001 per page). At 200 entity pages = $0.20 per full rebuild.
5. **Dashboard tech**: Extend existing `static/dashboard/` (vanilla JS) or new frontend? Graph visualization: D3 for custom, vis.js for quick wins.
6. **Heat score formula**: What thresholds for hot/warm/cool? Need to calibrate against prod usage data.
7. **Deep recall UX**: How does a user trigger "search old memories too"? Checkbox? Automatic fallback when normal search returns few results?
8. **Backfill strategy**: How to populate relations for existing 2,580 memories? Batch job that does pairwise similarity? Or only for new memories going forward?

## Testing & Validation Plan

1. **Dump prod data** into local Docker instance
2. **Measure baselines**: search latency, add latency, maintenance duration
3. **Implement auto-relations** → backfill all existing memories → measure relation count and search impact
4. **Implement lint** → run on prod data → verify contradiction detection quality
5. **Implement compilation** → generate entity pages → verify quality and cluster coherence
6. **Scale test**: duplicate to 50K memories → verify performance holds
7. **Dashboard**: browse entity pages, resolve contradictions, navigate graph
8. All testing against local Docker (`docker compose up -d`) with prod-like data

## Next Steps

Run `/workflows:plan` to create implementation plan for Layer 1 (Auto-Relations).
