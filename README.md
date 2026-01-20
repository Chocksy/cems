# CEMS - Continuous Evolving Memory System

A dual-layer memory system (personal + shared) with scheduled maintenance and knowledge graph, built on top of [Mem0](https://github.com/mem0ai/mem0) and exposed as an MCP server for Claude Code integration.

## Features

- **Dual-Layer Memory**: Personal (per-user) and Shared (team) memory namespaces
- **Mem0 Backend**: Production-ready memory engine with automatic fact extraction
- **Knowledge Graph**: Kuzu-based graph for relationship tracking between memories
- **Scheduled Maintenance**: Automatic memory decay, consolidation, and optimization
  - Nightly: Merge duplicates, promote frequently-accessed memories
  - Weekly: Compress old memories, prune stale ones
  - Monthly: Rebuild embeddings, archive dead memories
- **MCP Server**: Integrates with Claude Code, Cursor, VS Code, and other MCP clients
- **Extended Metadata**: Access tracking, categories, tags, priorities
- **5-Stage Retrieval Pipeline**: Query synthesis, vector search, graph traversal, relevance filtering, token-budgeted assembly

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-memory.git
cd llm-memory

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Environment Setup

CEMS requires only **one API key** - everything goes through OpenRouter:

```bash
# Required: OpenRouter key for all operations (LLM + embeddings)
export OPENROUTER_API_KEY="sk-or-your-openrouter-key"

# Optional: User identification
export CEMS_USER_ID="your-username"
export CEMS_TEAM_ID="your-team"  # Enables shared memory
```

**Why OpenRouter?**
- **Single key** for all LLM and embedding operations
- **Model flexibility** - switch between OpenAI, Anthropic, Google models without code changes
- **Cost tracking** - unified billing across all providers
- OpenRouter provides both [chat completions](https://openrouter.ai/docs/api/reference/completions) and [embeddings](https://openrouter.ai/docs/api/reference/embeddings) APIs

### Option 1: Docker Deployment (Recommended)

The easiest way to run CEMS is with Docker Compose:

```bash
cd deploy

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services (Qdrant, PostgreSQL, CEMS server)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cems-server
```

Services:
- **cems-server**: MCP server on port 8765
- **cems-qdrant**: Vector database on port 6333
- **cems-postgres**: Metadata storage on port 5432

### Option 2: Local Development

For local development without Docker:

```bash
# Start the MCP server in stdio mode (for local Claude Code)
python -m cems.server

# Or run in HTTP mode
CEMS_MODE=http python -m cems.server
```

### Configure Claude Code

#### For HTTP Mode (Docker/Server)

First, get a user API key from your CEMS admin:

```bash
# Admin creates a user and gets an API key
curl -X POST http://localhost:8765/admin/users \
  -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username"}'
# Returns: {"api_key": "cems_usr_abc123..."}
```

Then add to `~/.claude.json` in the `mcpServers` section:

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {
        "Authorization": "Bearer cems_usr_your_api_key",
        "X-Team-ID": "your-team"
      }
    }
  }
}
```

#### For stdio Mode (Local)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "cems": {
      "command": "python",
      "args": ["-m", "cems.server"],
      "env": {
        "CEMS_USER_ID": "your-username",
        "OPENROUTER_API_KEY": "sk-or-your-openrouter-key"
      }
    }
  }
}
```

### Usage in Claude Code

After configuration and restarting Claude Code, you can use the MCP tools directly:

```
# Add a memory (via MCP tool)
Use memory_add to store: "I prefer Python for backend development"

# Search memories
Use memory_search to find: "What are my coding preferences?"

# The server provides 5 tools:
# - memory_add: Store memories
# - memory_search: Search with 5-stage pipeline
# - memory_forget: Delete or archive
# - memory_update: Update content
# - memory_maintenance: Run maintenance jobs
```

### CLI Usage

```bash
# Check status and configuration
cems status

# Add a memory
cems add "I prefer dark mode" --category preferences

# Search memories
cems search "coding preferences"

# List all memories
cems list

# Run maintenance manually
cems maintenance run consolidation
cems maintenance run all

# Show maintenance schedule
cems maintenance schedule

# Index a repository
cems index repo /path/to/repo --scope shared
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CEMS (What We Built)                          │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │ MCP Server  │  │  Scheduler  │  │  CLI                     │ │
│  │ (FastMCP)   │  │ (APScheduler│  │  cems status/add/search  │ │
│  │ 5 tools     │  │  3 jobs)    │  │                          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────────────┘ │
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Memory Wrapper (namespace isolation)            ││
│  │         personal:{user_id}  |  shared:{team_id}             ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│  ┌───────────────────┬──────┴──────┬───────────────────────────┐│
│  │   Metadata Store  │ Graph Store │  Vector Store (Mem0)      ││
│  │   (SQLite)        │ (Kuzu)      │  (Qdrant)                 ││
│  └───────────────────┴─────────────┴───────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Mem0 (Dependency)                             │
│  • Fact extraction prompts (via OpenRouter)                     │
│  • ADD/UPDATE/DELETE/NONE logic                                 │
│  • Qdrant vector store                                          │
│  • Embeddings (via OpenRouter)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

CEMS is configured via environment variables (all prefixed with `CEMS_`):

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_USER_ID` | `default` | Your user identifier (stdio mode) |
| `CEMS_TEAM_ID` | (none) | Team ID for shared memory |
| `CEMS_STORAGE_DIR` | `~/.cems` | Storage directory |
| `CEMS_MODE` | `stdio` | Server mode: `stdio` or `http` |
| `CEMS_DATABASE_URL` | (none) | PostgreSQL URL (required for HTTP mode) |
| `CEMS_ADMIN_KEY` | (none) | Admin API key for user management |

### LLM Configuration (Single-Key via OpenRouter)

All LLM and embedding operations use OpenRouter:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | - | **Required** - Single key for all operations |
| `CEMS_MEM0_MODEL` | `openai/gpt-4o-mini` | Model for Mem0 fact extraction |
| `CEMS_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `CEMS_LLM_MODEL` | `anthropic/claude-3-haiku` | Model for maintenance ops |

Model names use OpenRouter format: `provider/model` (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`)

### Vector Store Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_VECTOR_STORE` | `qdrant` | Backend: `qdrant`, `chroma`, `lancedb` |
| `CEMS_QDRANT_URL` | (none) | Qdrant server URL (e.g., `http://localhost:6333`) |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_ENABLE_QUERY_SYNTHESIS` | `true` | Enable LLM query expansion |
| `CEMS_RELEVANCE_THRESHOLD` | `0.3` | Minimum score to include in results |
| `CEMS_DEFAULT_MAX_TOKENS` | `2000` | Token budget for retrieval results |

### Graph Store Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_ENABLE_GRAPH` | `true` | Enable knowledge graph |
| `CEMS_GRAPH_STORE` | `kuzu` | Graph backend: `kuzu` or `none` |

### Scheduler Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_ENABLE_SCHEDULER` | `true` | Enable background maintenance |
| `CEMS_NIGHTLY_HOUR` | `3` | Hour for nightly consolidation (0-23) |
| `CEMS_WEEKLY_DAY` | `sun` | Day for weekly summarization |
| `CEMS_WEEKLY_HOUR` | `4` | Hour for weekly summarization |
| `CEMS_MONTHLY_DAY` | `1` | Day of month for monthly reindex |
| `CEMS_MONTHLY_HOUR` | `5` | Hour for monthly reindex |

### Decay Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_STALE_DAYS` | `90` | Days before memory is stale |
| `CEMS_ARCHIVE_DAYS` | `180` | Days before memory is archived |
| `CEMS_HOT_ACCESS_THRESHOLD` | `5` | Access count to consider memory "hot" |
| `CEMS_DUPLICATE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity for duplicate detection |

## MCP Tools (5 Essential Tools)

| Tool | Description |
|------|-------------|
| `memory_add` | Store memories (personal or shared namespace) |
| `memory_search` | Unified search with 5-stage retrieval pipeline |
| `memory_forget` | Delete or archive a memory |
| `memory_update` | Update a memory's content |
| `memory_maintenance` | Run maintenance jobs (consolidation, summarization, reindex) |

## MCP Resources (3 Essential)

| Resource | Description |
|----------|-------------|
| `memory://status` | System status and configuration |
| `memory://personal/summary` | Personal memory overview |
| `memory://shared/summary` | Shared memory overview |

## 5-Stage Inference Retrieval Pipeline

The `memory_search` tool implements a sophisticated retrieval pipeline:

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Query Synthesis                               │
│ - LLM expands query for better retrieval               │
│ - "coding preferences" → multiple search terms         │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Candidate Retrieval (top_k=20)                │
│ - Vector search via Mem0/Qdrant                        │
│ - Graph traversal via Kuzu (if enabled)                │
│ - Merge all candidates                                 │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Relevance Filtering                           │
│ - Filter candidates with score < 0.3 threshold         │
│ - Reject low-confidence matches                        │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Temporal Ranking                              │
│ - time_decay = 1.0 / (1.0 + (age_days / 30))          │
│ - final_score = relevance * time_decay * priority      │
│ - Sort by final_score descending                       │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Token-Budgeted Assembly                       │
│ - Select memories until max_tokens exhausted           │
│ - Format as context payload                            │
└────────────────────────────────────────────────────────┘
    │
    ▼
Formatted Results
```

## Maintenance Jobs

### Nightly Consolidation (3 AM)
- Finds semantically duplicate memories (>92% similarity)
- Uses LLM to merge duplicates into comprehensive memories
- Promotes frequently-accessed memories (>5 accesses)

### Weekly Summarization (Sunday 4 AM)
- Uses LLM to generate category summaries
- Compresses old memories (>30 days) into category summaries
- Prunes stale memories (not accessed in 90 days)

### Monthly Re-indexing (1st at 5 AM)
- Rebuilds embeddings with latest model
- Archives dead memories (not accessed in 180 days)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=cems

# Type checking
mypy src/cems

# Linting
ruff check src/cems
```

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment guide including:
- Docker Compose setup
- Kubernetes deployment
- Team management
- Backup and restore
- Monitoring

## Troubleshooting

### CLI Error: "Extra inputs are not permitted"

If you see an error about `cems_api_key` or other fields not being permitted, your `.env` file has extra variables. This is OK - the config now ignores unknown variables. Update to the latest version.

### "Not authenticated" in Claude Code MCP panel

This is cosmetic. CEMS uses header-based identification (`X-User-ID`/`X-Team-ID`), not OAuth. If the tools work, you're connected.

### Search returns no results

Check the relevance threshold (default 0.3). If similarity scores are low, results get filtered. Try more specific queries that match your stored memories semantically.

### Server won't start

```bash
# Check required environment variable
echo $OPENROUTER_API_KEY

# For Docker, check logs
docker logs cems-server
```

## License

MIT
