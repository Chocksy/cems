# CEMS Architecture — Ground Truth (from code investigation)

## What We Actually Use

### Services (Docker Compose)
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | PostgreSQL 16 + pgvector — unified database |
| `cems-server` | Built from `./Dockerfile` | 8765 | Python REST API (Starlette + uvicorn) |
| `cems-mcp` | Built from `./mcp-wrapper/Dockerfile` | 8766 | Express.js MCP wrapper (Streamable HTTP) |
| `llama-embed` | llama.cpp server | 8081 | Local embedding (DISABLED — uses OpenRouter) |
| `llama-rerank` | llama.cpp server | 8082 | Local reranker (DISABLED — hurts performance) |

### Database (PostgreSQL + pgvector ONLY)
- **memory_documents** — Main doc store (content, user_id, scope, category, tags, soft-delete)
- **memory_chunks** — Chunked content with `vector(1536)` embeddings, HNSW index, tsvector FTS
- **users / teams / team_members** — Auth and team management
- NO Qdrant, NO Kuzu, NO Mem0

### Embeddings
- **Model**: `text-embedding-3-small` via OpenRouter
- **Dimension**: 1536
- **Endpoint**: `https://openrouter.ai/api/v1/embeddings`

### Search Pipeline (9 stages)
1. Query Understanding → routes to vector or hybrid
2. Profile Probe (for preferences)
3. Query Synthesis (LLM expands to 2-5 terms)
4. HyDE (hypothetical document generation)
5. Candidate Retrieval (pgvector HNSW + BM25 tsvector)
6. RRF Fusion
7. Re-ranking (DISABLED — hurts perf)
8. Scoring Adjustments (time decay, priority, project boost)
9. Token-Budgeted Assembly

### Maintenance (4 jobs on APScheduler)
| Job | Schedule | Purpose |
|-----|----------|---------|
| Consolidation | Nightly 3 AM | Merge semantic duplicates |
| Observation Reflection | Nightly 3:30 AM | Condense observations |
| Summarization | Weekly Sun 4 AM | Compress old, prune stale |
| Re-indexing | Monthly 1st 5 AM | Rebuild embeddings |

### Observer Daemon
- Polls `~/.claude/projects/*/` JSONL files every 30s
- Extracts observations via Gemini 2.5 Flash (OpenRouter)
- 50KB threshold, state in `~/.claude/observer/`

### CLI Entry Points
| Command | Purpose |
|---------|---------|
| `cems` | CLI (status, health, add, search, list, delete, update, setup, etc.) |
| `cems-server` | HTTP REST API server |
| `cems-observer` | Observer daemon |

### MCP Wrapper (6 tools, 3 resources)
- Tools: memory_add, memory_search, memory_forget, memory_update, memory_maintenance, session_analyze
- Resources: memory://status, memory://personal/summary, memory://shared/summary

## README Corrections Needed
- ~~"built on Mem0"~~ → custom DocumentStore on PostgreSQL
- ~~"Kuzu-based relationship tracking"~~ → removed
- ~~"Qdrant vector database on port 6333"~~ → pgvector in PostgreSQL
- ~~"cems-qdrant service"~~ → doesn't exist
- ~~"Mem0 Backend"~~ → custom engine
- Missing: uninstall command (doesn't exist at all)

## Missing Features
- **No `cems uninstall`** — setup has no reverse
