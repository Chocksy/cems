<p align="center">
  <img src="assets/banner-chef.png" alt="CEMS — Continuous Evolving Memory System" width="800">
</p>

<h1 align="center">CEMS</h1>
<p align="center"><strong>Continuous Evolving Memory System</strong></p>
<p align="center">Persistent memory for AI coding assistants. Works with Claude Code, Cursor, Codex, Goose, and any MCP-compatible agent.</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-compatible-green.svg" alt="MCP Compatible"></a>
  <a href="https://docs.anthropic.com/en/docs/claude-code"><img src="https://img.shields.io/badge/Claude_Code-hooks-blueviolet.svg" alt="Claude Code"></a>
  <img src="https://img.shields.io/badge/Recall%405-98%25-brightgreen.svg" alt="Recall@5: 98%">
</p>

---

## Quick Start (Client Install)

You need two things from your CEMS admin: a **server URL** and an **API key**.

### Option A: Interactive install (recommended)

```bash
curl -fsSL https://getcems.com/install.sh | bash
```

It will ask for your API URL and key, then let you choose which IDEs to configure (Claude Code, Cursor, Codex, Goose, or all).

### Option B: Non-interactive install

```bash
CEMS_API_KEY=your-key-here CEMS_API_URL=https://cems.example.com \
  curl -fsSL https://getcems.com/install.sh | bash
```

### Option C: Install from source

```bash
git clone https://github.com/chocksy/cems.git && cd cems
./install.sh
```

### Option D: Install skills only (any agent)

If your agent supports [skills.sh](https://skills.sh), you can add CEMS skills without the full install:

```bash
npx skills add Chocksy/cems
```

This installs recall, remember, and foundation skills. You still need a CEMS server and API key configured separately.

### What the installer does

1. Installs [uv](https://docs.astral.sh/uv/) if missing
2. Installs the CEMS CLI (`cems`, `cems-server`, `cems-observer`) via `uv tool install`
3. Runs `cems setup` which lets you choose IDEs to configure:
   - **Claude Code** (`--claude`): 6 hooks, 6 skills, 2 commands, settings.json config
   - **Cursor** (`--cursor`): 3 hooks, 5 skills, MCP config in `mcp.json`
   - **Codex** (`--codex`): 3 commands, 2 skills, MCP config in `config.toml`
   - **Goose** (`--goose`): MCP extension in `config.yaml`
4. Saves credentials to `~/.cems/credentials` (chmod 600)
5. Saves IDE choices to `~/.cems/install.conf` (used by `cems update` for non-interactive re-deploys)

### After install

```bash
cems --version    # Verify CLI is installed
cems health       # Check server connection
cems update       # Pull latest version + re-deploy hooks/skills
cems setup        # Re-run setup (reconfigure credentials, re-install hooks)
cems uninstall    # Remove hooks/skills (keeps credentials by default)
```

### Updating

CEMS auto-updates when you start a new Claude Code session — no action needed. If your install is more than 24 hours old, the SessionStart hook pulls the latest version in the background.

To update manually:

```bash
cems update          # Pull latest + re-deploy hooks/skills
cems update --hooks  # Re-deploy hooks only (skip package upgrade)
```

Updates re-deploy to whatever IDEs you originally configured (stored in `~/.cems/install.conf`). Auto-update can be disabled by setting `CEMS_AUTO_UPDATE=0` in your environment or `~/.cems/credentials`.

### Credentials

Stored in `~/.cems/credentials` (chmod 600). Checked in order:
1. CLI flags: `--api-url`, `--api-key`
2. Environment: `CEMS_API_URL`, `CEMS_API_KEY`
3. Credentials file: `~/.cems/credentials`

## How It Works

CEMS hooks into your IDE and provides persistent memory across sessions:

1. **Memory Injection** -- On every prompt, relevant memories are searched and injected as context
2. **Session Learning** -- On session end, learnings are extracted and stored
3. **Observational Memory** -- The observer daemon watches session transcripts and extracts high-level observations about your workflow
4. **Scheduled Maintenance** -- Nightly/weekly/monthly jobs deduplicate, compress, and prune memories automatically

## What Gets Installed

<details>
<summary><strong>Claude Code</strong> (~/.claude/)</summary>

```
~/.claude/
├── settings.json           # Hooks config (merged, not overwritten)
├── hooks/
│   ├── cems_session_start.py        # Profile + context injection
│   ├── cems_user_prompts_submit.py  # Memory search + observations
│   ├── cems_post_tool_use.py        # Tool learning extraction
│   ├── cems_pre_tool_use.py         # Gate rules enforcement
│   ├── cems_stop.py                 # Session analysis + observer
│   ├── cems_pre_compact.py          # Pre-compaction hook
│   └── utils/                       # Shared utilities
├── skills/cems/
│   ├── recall.md           # /recall - Search memories
│   ├── remember.md         # /remember - Add personal memory
│   ├── share.md            # /share - Add team memory
│   ├── forget.md           # /forget - Delete memory
│   ├── context.md          # /context - Show status
│   └── memory-guide.md     # Proactive memory usage guide
└── commands/
    ├── recall.md           # /recall command
    └── remember.md         # /remember command
```
</details>

<details>
<summary><strong>Cursor</strong> (~/.cursor/)</summary>

```
~/.cursor/
├── mcp.json                # MCP server config (merged)
├── hooks/
│   ├── cems_session_start.py    # Profile injection
│   ├── cems_agent_response.py   # Agent response hook
│   └── cems_stop.py             # Session end hook
└── skills/
    ├── cems-recall/SKILL.md     # Search memories
    ├── cems-remember/SKILL.md   # Add memory
    ├── cems-forget/SKILL.md     # Delete memory
    ├── cems-share/SKILL.md      # Share with team
    └── cems-context/SKILL.md    # Memory status
```
</details>

<details>
<summary><strong>Codex</strong> (~/.codex/)</summary>

```
~/.codex/
├── config.toml             # MCP server config (merged)
├── commands/
│   ├── recall.md           # Search memories
│   ├── remember.md         # Add memory
│   └── foundation.md       # Foundation guidelines
└── skills/
    ├── recall/SKILL.md     # Search memories
    └── remember/SKILL.md   # Add memory
```
</details>

<details>
<summary><strong>Goose</strong> (~/.config/goose/)</summary>

```
~/.config/goose/
└── config.yaml             # CEMS MCP extension block (merged)
```
</details>

<details>
<summary><strong>Shared</strong> (~/.cems/)</summary>

```
~/.cems/
├── credentials             # API URL + key (chmod 600)
└── install.conf            # IDE choices for cems update
```
</details>

## Usage

### Skills (slash commands)

```
/remember I prefer Python for backend development
/remember The database uses snake_case column names
/recall What are my coding preferences?
/share API endpoints follow REST conventions with /api/v1/...
/forget abc123
/context
```

Available in Claude Code, Cursor, and Codex. Exact skill names vary by IDE.

### CLI

```bash
cems status                          # System status
cems health                          # Server health check
cems add "I prefer dark mode"        # Add a memory
cems search "coding preferences"     # Search memories
cems list                            # List all memories
cems rule add                        # Interactive constitution/playbook rule wizard
cems rule load --kind constitution   # Load default constitution rule bundle
cems update                          # Update to latest version
cems update --hooks                  # Re-deploy hooks only (no package upgrade)
cems maintenance --job consolidation # Run maintenance
cems uninstall                       # Remove hooks/skills
cems uninstall --all                 # Remove everything including credentials
```

---

## Server Deployment

For team usage, deploy CEMS as a server. Requires Docker Compose.

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **postgres** | `pgvector/pgvector:pg16` | 5432 | PostgreSQL + pgvector (vectors + metadata + auth) |
| **cems-server** | Built from `Dockerfile` | 8765 | Python REST API (Starlette + uvicorn) |
| **cems-mcp** | Built from `mcp-wrapper/` | 8766 | MCP wrapper (Express.js, Streamable HTTP) |

### Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/chocksy/cems.git && cd cems
   cp .env.example .env
   # Edit .env with your OPENROUTER_API_KEY and CEMS_ADMIN_KEY
   ```

2. **Start services:**
   ```bash
   docker compose up -d
   ```

3. **Create your first user:**
   ```bash
   source .env
   curl -X POST http://localhost:8765/admin/users \
     -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
     -H "Content-Type: application/json" \
     -d '{"username": "yourname"}'
   # Returns: {"api_key": "cems_usr_..."}
   ```

4. **Give the API key to your team member** -- they run the client install above.

### Environment Variables

Required (set in `.env`):

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Get from https://openrouter.ai/keys |
| `CEMS_ADMIN_KEY` | Generate with `openssl rand -hex 32` |

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `cems_secure_password` | Change in production |
| `CEMS_EMBEDDING_BACKEND` | `openrouter` | Embedding provider |
| `CEMS_EMBEDDING_DIMENSION` | `1536` | Embedding dimension |
| `CEMS_RERANKER_BACKEND` | `disabled` | Reranker (disabled by default) |

## Architecture

### Storage

Everything lives in **PostgreSQL with pgvector**:

- **`memory_documents`** -- Documents with content, user/team scoping, categories, tags, soft-delete
- **`memory_chunks`** -- Chunked content with 1536-dim vector embeddings (HNSW index) and full-text search (tsvector)
- **`users` / `teams`** -- Authentication via bcrypt-hashed API keys

### Embeddings

`text-embedding-3-small` via OpenRouter (1536 dimensions). Batch support for bulk operations.

### Search Pipeline

CEMS uses a multi-stage retrieval pipeline:

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

### Maintenance

Scheduled via APScheduler:

| Job | Schedule | Purpose |
|-----|----------|---------|
| Consolidation | Nightly 3 AM | Merge semantic duplicates (cosine >= 0.92) |
| Observation Reflection | Nightly 3:30 AM | Condense observations per project |
| Summarization | Weekly Sun 4 AM | Compress old memories, prune stale |
| Re-indexing | Monthly 1st 5 AM | Rebuild embeddings, archive dead memories |

### Observer Daemon

The observer (`cems-observer`) runs as a background process on the client machine:

- Polls `~/.claude/projects/*/` JSONL transcript files every 30 seconds
- When 50KB of new content accumulates, sends it to the server
- Server extracts high-level observations via Gemini 2.5 Flash
- Observations like "User deploys via Coolify" or "Project uses PostgreSQL" are stored as memories

### MCP Integration

The MCP wrapper on port 8766 exposes CEMS as an MCP server with 6 tools:

| Tool | Description |
|------|-------------|
| `memory_add` | Store a memory |
| `memory_search` | Search with the full retrieval pipeline |
| `memory_get` | Retrieve full document by ID |
| `memory_forget` | Delete or archive a memory |
| `memory_update` | Update memory content |
| `memory_maintenance` | Trigger maintenance jobs |

### API Endpoints

<details>
<summary>Full API reference</summary>

**Public API** (Bearer token auth):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/memory/add` | Add a memory |
| POST | `/api/memory/search` | Search memories |
| POST | `/api/memory/forget` | Delete memory |
| POST | `/api/memory/update` | Update memory |
| POST | `/api/memory/log-shown` | Feedback tracking |
| POST | `/api/memory/maintenance` | Run maintenance |
| GET | `/api/memory/get` | Get full document by ID |
| GET | `/api/memory/list` | List memories |
| GET | `/api/memory/status` | System status |
| GET | `/api/memory/profile` | Session profile context |
| GET | `/api/memory/foundation` | Foundation guidelines |
| GET | `/api/memory/gate-rules` | Gate rules by project |
| POST | `/api/session/summarize` | Session summary (observer daemon) |
| POST | `/api/tool/learning` | Tool learning |
| POST | `/api/index/repo` | Index git repo |

**Admin API** (`CEMS_ADMIN_KEY` auth):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/admin/users` | List/create users |
| GET/PATCH/DELETE | `/admin/users/{id}` | Manage user |
| POST | `/admin/users/{id}/reset-key` | Reset API key |
| GET/POST | `/admin/teams` | List/create teams |

</details>

## Troubleshooting

### Memory not being recalled

1. Check credentials: `cat ~/.cems/credentials`
2. Test connection: `cems health`
3. Test search: `cems search "test"`
4. Check hook output: `echo '{"prompt": "test"}' | uv run ~/.claude/hooks/cems_user_prompts_submit.py`

### Skills not appearing

1. Verify files exist:
   - Claude Code: `ls ~/.claude/skills/cems/`
   - Cursor: `ls ~/.cursor/skills/cems-recall/`
   - Codex: `ls ~/.codex/skills/recall/`
2. Restart your IDE
3. Type `/` and look for `remember`, `recall`, etc.

### Re-install hooks

```bash
cems setup    # Re-runs the full setup
```

## Development

```bash
git clone https://github.com/chocksy/cems.git && cd cems
uv pip install -e ".[dev]"
pytest                    # Run tests
mypy src/cems             # Type checking
```

## License

MIT
