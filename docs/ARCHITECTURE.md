# CEMS Architecture

## Overview

<p align="center">
  <img src="../assets/architecture.png" alt="CEMS Architecture" width="800">
</p>

## Storage

Everything lives in **PostgreSQL with pgvector** — no Redis, Qdrant, or external vector DB needed.

| Table | Purpose |
|-------|---------|
| `memory_documents` | Documents with content, user/team scoping, categories, tags, soft-delete |
| `memory_chunks` | Chunked content with 1536-dim vector embeddings (HNSW index) + full-text search (tsvector) |
| `users` / `teams` | Authentication via bcrypt-hashed API keys |

## Embeddings

`text-embedding-3-small` via OpenRouter (1536 dimensions). Batch support for bulk operations.

## Search Pipeline

Multi-stage retrieval pipeline:

```
Query → Understanding → Synthesis → HyDE → Retrieval → RRF Fusion → Filtering → Scoring → Assembly → Results
```

| Stage | What it does |
|-------|-------------|
| 1. Query Understanding | LLM routes to vector or hybrid strategy |
| 2. Query Synthesis | LLM expands query into 2-5 search terms |
| 3. HyDE | Generates hypothetical ideal answer for better matching |
| 4. Candidate Retrieval | pgvector HNSW (vector) + tsvector (BM25 full-text) |
| 5. RRF Fusion | Reciprocal Rank Fusion combines result lists |
| 6. Relevance Filtering | Removes results below threshold |
| 7. Scoring Adjustments | Time decay, priority boost, project-scoped boost |
| 8. Token-Budgeted Assembly | Greedy selection within token budget (default: 2000) |

Search modes: `vector` (fast, 0 LLM calls), `hybrid` (thorough, 3-4 LLM calls), `auto` (smart routing).

## Maintenance

Scheduled via APScheduler — runs in-process, no separate worker needed.

| Job | Schedule | Purpose |
|-----|----------|---------|
| Consolidation | Nightly 3 AM | Merge semantic duplicates (cosine >= 0.92) |
| Observation Reflection | Nightly 3:30 AM | Condense observations per project |
| Summarization | Weekly Sun 4 AM | Compress old memories, prune stale |
| Re-indexing | Monthly 1st 5 AM | Rebuild embeddings, archive dead memories |

## Observer Daemon

The observer (`cems-observer`) runs as a background process on the client machine:

- Polls `~/.claude/projects/*/` JSONL transcript files every 30 seconds
- When 50KB of new content accumulates, sends it to the server
- Server extracts high-level observations via Gemini 2.5 Flash
- Observations like "User deploys via Coolify" or "Project uses PostgreSQL" are stored as memories

## MCP Integration

The MCP wrapper (port 8766) exposes CEMS as an MCP server with 6 tools:

| Tool | Description |
|------|-------------|
| `memory_add` | Store a memory |
| `memory_search` | Search with the full retrieval pipeline |
| `memory_get` | Retrieve full document by ID |
| `memory_forget` | Delete or archive a memory |
| `memory_update` | Update memory content |
| `memory_maintenance` | Trigger maintenance jobs |
