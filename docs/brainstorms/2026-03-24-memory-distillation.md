---
date: 2026-03-24
topic: memory-distillation
---

# Memory Distillation — Two-Column Progressive Condensation

## What We're Building

Every memory gets two versions: a condensed summary (~500 chars, stored in `content`) and full detailed content (stored in a new `content_detailed` column). Search, hooks, and agentic search always return the condensed version. LLMs can fetch full details on-demand via `/recall <id>`.

Nightly maintenance progressively distills both columns — `content` gets tighter each pass, `content_detailed` also gets condensed over time to prevent infinite growth.

## Why This Approach

- **Token cost**: Agentic search loads 700K chars of context. Condensed summaries cut this dramatically.
- **Relevance**: Terse, observer-quality summaries surface better in search than bloated multi-KB dumps.
- **No data loss**: Full content preserved in `content_detailed` — consolidate, never delete philosophy upheld.
- **Infrastructure exists**: Search already returns 500-char snippets, `memory_get` already serves full content, hooks already hint "use /recall for full doc".

## Key Decisions

- **Two columns**: `content` (condensed, ~500 chars) + `content_detailed` (full original, also condensable over time).
- **No distillation on ingest**: Normal `memory_add` already does LLM summarization. Nightly maintenance handles progressive distillation.
- **Big document ingest**: Index scripts and `/store` skill detect large content, store full doc in `content_detailed`, LLM-distilled summary in `content` at write time.
- **Agentic search uses condensed**: Agents get summaries by default. They call `/recall <id>` if they need more detail.
- **Progressive nightly distillation**: Each night, consolidation tightens `content` further. `content_detailed` also gets condensed periodically to prevent unbounded growth.
- **Access via /recall**: Extend existing `/recall <id>` to return `content_detailed` when available, fall back to `content`.
- **Observer prompt as template**: Distillation prompt follows the observer daemon's style — terse, fact-preserving, temporal anchoring, exact names/numbers.

## Scope

### In scope
- `content_detailed TEXT` column on `memory_documents`
- Migration script
- Distillation job in maintenance (new or extension of summarization)
- Update `/recall` to serve `content_detailed`
- Update agentic search to use condensed content
- Update `/store` skill to support full-doc ingestion
- Distillation prompt (observer-style)

### Out of scope (for now)
- Re-distilling all existing memories (can be done as a one-time migration after feature ships)
- Index script for parsing repositories (separate feature)
- Changes to vector search / chunk embeddings (chunks already embed independently of content)

## Open Questions

1. **Chunk embeddings**: Chunks are embedded from the full content. After distillation, should chunks be re-embedded from the condensed content, or keep the original chunk embeddings? (Original is probably better for search recall.)
2. **content_detailed growth cap**: What's the max size before we also condense content_detailed? 10K chars? 50K?
3. **Existing consolidated memories**: The 35 consolidated docs from today (3K-8K chars each) — do we retroactively distill them as part of the first nightly run?

## Next Steps
→ Implement via `/workflows:plan`
