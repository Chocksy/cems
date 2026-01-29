# CEMS - Continuous Evolving Memory System

A dual-layer memory system (personal + shared) with scheduled maintenance and knowledge graph, built on [Mem0](https://github.com/mem0ai/mem0). Integrates with Claude Code via hooks and skills.

## Quick Start

### Option 1: Automatic Install

```bash
git clone https://github.com/yourusername/cems.git
cd cems
./install.sh
```

The installer will:
1. Install the CEMS Python package
2. Set up Claude Code hooks and skills
3. Configure environment variables

### Option 2: Manual Install

**If you have no existing `~/.claude` folder:**
```bash
cp -r claude-setup ~/.claude
```

**If you have existing Claude Code config:**
```bash
# Backup existing
cp -r ~/.claude ~/.claude.backup

# Copy hooks and skills
cp claude-setup/hooks/cems_*.py ~/.claude/hooks/
cp -r claude-setup/skills/cems ~/.claude/skills/

# Then manually add hooks to ~/.claude/settings.json (see below)
```

### Environment Variables

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export CEMS_API_URL="https://your-cems-server.com"
export CEMS_API_KEY="your-api-key"
```

Then restart your terminal and Claude Code.

## Configuration

### Hooks Configuration

If merging into existing config, add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ~/.claude/hooks/cems_user_prompts_submit.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ~/.claude/hooks/cems_stop.py"
          }
        ]
      }
    ]
  }
}
```

### Directory Structure

After installation, your `~/.claude` folder should contain:

```
~/.claude/
├── settings.json           # Hooks configuration
├── hooks/
│   ├── cems_user_prompts_submit.py  # Memory injection on each prompt
│   └── cems_stop.py                 # Session summary storage
└── skills/
    └── cems/
        ├── remember.md     # /remember - Add personal memory
        ├── recall.md       # /recall - Search memories
        ├── share.md        # /share - Add team memory
        ├── forget.md       # /forget - Delete memory
        └── context.md      # /context - Show status
```

## Usage

### Skills (in Claude Code)

```
/remember I prefer Python for backend development
/remember The database uses snake_case column names
/recall What are my coding preferences?
/recall database conventions
/share API endpoints follow REST conventions with /api/v1/...
/forget abc123  # Memory ID from search results
/context        # Show memory system status
```

### How It Works

1. **Memory Injection**: On every prompt, the `UserPromptSubmit` hook searches CEMS and injects relevant memories as context
2. **Session Summaries**: On session end, the `Stop` hook stores a summary of your commits to CEMS
3. **Skills**: Slash commands provide direct access to memory operations

### CLI Usage

```bash
# Check status
cems status

# Add a memory
cems add "I prefer dark mode" --category preferences

# Search memories
cems search "coding preferences"

# List all memories
cems list
```

## Server Deployment

For team usage, deploy CEMS as a server:

### Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/yourusername/cems && cd cems
   cp .env.example .env
   # Edit .env with your OPENROUTER_API_KEY and CEMS_ADMIN_KEY
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Create your first user:**
   ```bash
   source .env
   curl -X POST http://localhost:8765/admin/users \
     -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
     -H "Content-Type: application/json" \
     -d '{"username": "yourname"}'
   # Returns: {"api_key": "cems_usr_abc123..."}
   ```

4. **Configure your IDE** (see Quick Start section above)

### Services

- **cems-server**: Python API server on port 8765 (internal)
- **cems-mcp**: Express MCP wrapper on port 8766 (public-facing)
- **cems-qdrant**: Vector database on port 6333
- **cems-postgres**: Metadata storage on port 5432

### Environment Variables

Required variables (set in `.env`):
- `OPENROUTER_API_KEY` - Get from https://openrouter.ai/keys
- `CEMS_ADMIN_KEY` - Generate with: `openssl rand -hex 32`

Optional:
- `POSTGRES_PASSWORD` - Default: `cems_secure_password` (change in production!)

## Features

- **Dual-Layer Memory**: Personal (per-user) and Shared (team) namespaces
- **Mem0 Backend**: Automatic fact extraction and deduplication
- **Knowledge Graph**: Kuzu-based relationship tracking
- **Scheduled Maintenance**:
  - Nightly: Merge duplicates, promote frequent memories
  - Weekly: Compress old memories, prune stale ones
  - Monthly: Rebuild embeddings, archive dead memories
- **9-Stage Retrieval Pipeline**: See [Search Architecture](#search-architecture) below

## Search Architecture

CEMS implements a sophisticated 9-stage retrieval pipeline that **fuses** results from multiple search methods using Reciprocal Rank Fusion (RRF). This is a funnel approach where results are combined, not filtered separately.

### Pipeline Overview

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Query Understanding (auto mode)                        │
│ LLM analyzes intent, complexity, domains → routes to strategy   │
│ File: retrieval.py:480-538                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Query Synthesis (hybrid mode)                          │
│ LLM expands query into 2-3 related search terms                 │
│ File: retrieval.py:36-68                                        │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: HyDE - Hypothetical Document Embeddings (hybrid mode)  │
│ LLM generates what an ideal answer would look like              │
│ File: retrieval.py:287-317                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Parallel Candidate Retrieval                           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Vector Search    │  │ Graph Traversal  │  │ Category      │ │
│  │ (Qdrant/Mem0)    │  │ (Kuzu)           │  │ Summaries     │ │
│  │                  │  │                  │  │ (LLM match)   │ │
│  │ 5 queries ×      │  │ Top-3 seeds →    │  │               │ │
│  │ 20 results each  │  │ 2-hop traversal  │  │ Category      │ │
│  └────────┬─────────┘  └────────┬─────────┘  │ boost map     │ │
│           │                     │            └───────┬───────┘ │
│           └──────────┬──────────┘                    │         │
│                      ↓                               │         │
│              Collect all results                     │         │
└──────────────────────┼───────────────────────────────┼─────────┘
                       ↓                               │
┌──────────────────────┴───────────────────────────────┘
│
↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: RRF Fusion                                             │
│ Reciprocal Rank Fusion combines all result lists                │
│ Formula: score = Σ(1 / (60 + rank_i)) across all retrievers    │
│ Blend: 30% RRF + 70% original vector score                      │
│ File: retrieval.py:324-380                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: LLM Re-ranking (hybrid mode, >3 candidates)            │
│ LLM evaluates ACTUAL relevance, not just similarity             │
│ Blend: 70% LLM rank + 30% previous score                        │
│ File: retrieval.py:388-473                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: Relevance Filtering                                    │
│ Filter results below threshold (default: 0.4)                   │
│ File: memory.py:960-964                                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 8: Unified Scoring Adjustments                            │
│ • Priority boost (1.0-2.0x)                                     │
│ • Time decay (50% per month)                                    │
│ • Pinned memory boost (+10%)                                    │
│ • Cross-category penalty (-20%)                                 │
│ • Project-scoped boost (+30% same, -20% different)             │
│ • Category summary boost (up to +30%)                           │
│ File: retrieval.py:230-279                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 9: Token-Budgeted Assembly                                │
│ Greedily select results until token budget exhausted            │
│ Default: 2000 tokens                                            │
│ File: retrieval.py:107-156                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final Results (formatted context for LLM injection)
```

### Search Modes

| Mode | Stages Active | LLM Calls | Use Case |
|------|---------------|-----------|----------|
| `vector` | 4, 7, 8, 9 | 0 | Simple queries, speed-critical |
| `hybrid` | All (1-9) | 4-5 | Complex queries, accuracy-critical |
| `auto` | 1 → routes | 1+ | Default - smart routing based on query |

### API Configuration

```json
POST /api/memory/search
{
  "query": "deployment process",
  "mode": "auto",                    // auto, vector, hybrid
  "enable_query_synthesis": true,   // Stage 2
  "enable_hyde": true,              // Stage 3
  "enable_rerank": true,            // Stage 6
  "enable_graph": true,             // Graph traversal in Stage 4
  "max_tokens": 2000,               // Token budget
  "raw": false                      // Bypass filtering (debug)
}
```

### Cost Characteristics

- **Vector mode**: ~100-300ms, 0 LLM tokens
- **Hybrid mode**: ~2-5s, ~1,200 tokens per query
- **Estimated annual cost**: ~$82/year at 1,000 searches/day (GPT-4o-mini pricing)

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_API_URL` | - | CEMS server URL (for hooks) |
| `CEMS_API_KEY` | - | Your API key (for hooks) |
| `OPENROUTER_API_KEY` | - | OpenRouter key (for server) |
| `CEMS_USER_ID` | `default` | User identifier (stdio mode) |
| `CEMS_TEAM_ID` | - | Team ID for shared memory |
| `CEMS_DATABASE_URL` | - | PostgreSQL URL (HTTP mode) |
| `CEMS_QDRANT_URL` | - | Qdrant server URL |

## Troubleshooting

### Memory not being recalled

1. Check environment variables are set: `echo $CEMS_API_URL`
2. Test API manually: `curl -X POST $CEMS_API_URL/api/memory/search -H "Authorization: Bearer $CEMS_API_KEY" -H "Content-Type: application/json" -d '{"query": "test"}'`
3. Check hook is running: Look for `<memory-recall>` tags in Claude's context

### Skills not appearing

1. Verify skills are in `~/.claude/skills/cems/`
2. Restart Claude Code
3. Type `/` and look for `remember`, `recall`, etc.

### Hook errors

Check the hook scripts directly:
```bash
echo '{"prompt": "test query"}' | uv run ~/.claude/hooks/cems_user_prompts_submit.py
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/cems
```

## License

MIT
