---
title: "Knowledge Engine: Next Steps"
type: feat
date: 2026-04-06
previous: docs/plans/2026-04-06-feat-auto-relations-knowledge-engine-plan.md
brainstorm: docs/brainstorms/2026-04-06-knowledge-engine-brainstorm.md
---

# Knowledge Engine: Next Steps

## What's Built (this session)

- Auto-relations: every `add_async()` links related memories via pgvector similarity
- RelationBuilderJob: backfill existing memories (3,900+ relations, 2,124 connected)
- Heat score: shown_count-based decay floor (hot memories resist compression)
- LintJob: contradiction detection, orphan counting, knowledge gap detection
- CompilationJob: cluster discovery + LLM synthesis of entity pages (126 generated)
- Wiki dashboard: D3 graph, stats, entity browser (Wikipedia layout), health panel, timeline
- Observer integration: session summaries auto-link to knowledge graph
- Entity dedup: cosine similarity check prevents near-duplicate entity pages

## What's Left

### Priority 1: Entity Page Lifecycle (the core remaining problem)

**Problem**: Entity pages are created once and never updated. Duplicates accumulate.
The system doesn't behave like Karpathy's "living wiki" yet.

**Key architecture fact**: Entity pages ARE regular `memory_documents` rows
with `category='entity-page'`. Same table, same IDs, same APIs. `/recall <id>`
and `mcp__cems__memory_get` already work for fetching full entity pages.

**Decisions from brainstorming (2026-04-07)**:

1. **Scheduled recompilation every 10 minutes**
   - Don't try N-memory triggers — too complex with observer bursts
   - Simple timer in scheduler: run CompilationJob every 10 min
   - Each run: find clusters, dedup against existing, create/update pages
   - Also schedule RelationBuilderJob (process new memories) every 10 min
   - LintJob: run daily (less urgent)
   - **Implementation**: Update `scheduler.py` with 3 new jobs

2. **Entity page dedup/merge**
   - Before creating: cosine similarity check against existing entity pages (>0.85 = skip)
   - Already partially built in CompilationJob
   - Need: merge logic when two entity pages converge to similar content
   - **Implementation**: Enhance CompilationJob dedup + add merge step

3. **Project-filtered entity index (MUST)**
   - Entity pages are project-scoped via `source_ref`
   - Index injected into hook MUST filter to current project
   - Cross-project entities are noise — filter them out
   - **Implementation**: Pass project to index query

4. **Orphan memories (live with for now)**
   - ~1,600 memories don't belong to any cluster
   - In Karpathy's model they shouldn't exist — wiki should cover everything
   - Future: generate "catch-all" entity pages per category for unclustered memories
   - For now: orphans are still found by vector search, just not in entity pages

### Priority 2: Entity Index in Hook (the killer feature)

**Decision**: Zero LLM cost on our side. Claude/Codex decides when to drill deeper.

**How it works**:
1. Hook already injects `<memory-recall>` with 5 memory snippets
2. NEW: Also fetch entity page titles + summaries (one DB query, no LLM)
3. Inject entity index into same `<memory-recall>` block
4. Claude sees: "Entity: Stripe Integration (15 sources) — covers webhooks, rate limiting..."
5. IF Claude needs more → uses existing `/recall <entity-id>` or `mcp__cems__memory_get`
6. Our server: serves a simple GET. Zero LLM calls.

**Cost to us**: One SQL query for entity index. That's it.
**Cost to Claude**: Only pays tokens when it decides to drill deeper.
**No agentic search changes needed** — this works with the existing hook.

**Implementation**:
- Modify `cems_user_prompts_submit.py` to include entity summaries in `<memory-recall>`
- Add API endpoint: `GET /api/wiki/index` returns titles + first 2-3 sentences per entity
- Format: "Entity: [title] (sources: N) — [summary]. Use `/recall <id>` to read full page."
- Entity pages already work with `mcp__cems__memory_get` — no tool changes needed

**Key insight**: We don't do the extraction. Claude does — when it wants to.
Like Karpathy's system where the LLM navigates the wiki on its own.

### Priority 3: Production Deployment

1. Fix prod FK: `memory_relations` FK points to `memories` (should be `memory_documents`)
2. Run `migrate_relations_fk.sql` on prod via SSH
3. Tag release → CI builds amd64 image
4. Run backfill on prod: `POST /api/memory/maintenance {job_type: relations}`
5. Run compilation on prod: `POST /api/memory/maintenance {job_type: compilation}`
6. Verify dashboard at prod URL

### Priority 4: Dashboard Polish

- Remove manual action buttons (Compile, Run Lint) — done
- Health tab auto-loads stats — done
- Fix graph to show only connected nodes — done
- Proper markdown rendering with marked.js — done
- Custom scrollbars — done
- **Remaining**: Mobile responsive, keyboard navigation, search within entity pages

### Lower Priority (Future)

- Deep recall mode (search soft-deleted memories)
- Relation type taxonomy (extends, contradicts, compiled_from)
- Community detection (Leiden algorithm instead of connected components)
- Query → file-back loop (save valuable answers as wiki pages)

## Architecture: How Entity Pages Integrate with Existing Systems

```
                    ┌─────────────────────────────┐
                    │     User types a prompt       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  UserPromptSubmit Hook        │
                    │  1. Search memories (vector)  │
                    │  2. Include entity index      │
                    │     (titles + summaries)      │
                    │  3. Inject <memory-recall>    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Claude sees:                 │
                    │  - 5 relevant memory snippets │
                    │  - Entity page index          │
                    │    "Use /recall ID to read"   │
                    └──────────────┬──────────────┘
                                   │
                          Claude decides:
                    ┌──────────────┼──────────────┐
                    │              │              │
              Has enough    Needs more      Needs deep
              context       about topic     historical
                    │              │              │
              Answers      Uses /recall     Agentic mode
              directly     to read full     searches all
                           entity page      memories
```

## Key Decisions from Brainstorming

1. **No manual buttons in dashboard** — entity compilation and lint are maintenance operations
2. **Entity pages are offered as index to LLM** — like Karpathy's index.md pattern
3. **LLM drills deeper via existing tools** — `mcp__cems__memory_get` reads full entity page
4. **Agentic search should use entity summaries** — 7x token reduction potential
5. **Entity lifecycle must be automatic** — scheduled compilation, staleness detection, dedup
6. **Don't send entity pages in bulk to LLM** — send index only, let LLM pull what it needs
