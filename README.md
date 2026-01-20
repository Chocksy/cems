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

### Docker Compose

```bash
cd deploy
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

Services:
- **cems-server**: MCP server on port 8765
- **cems-qdrant**: Vector database on port 6333
- **cems-postgres**: Metadata storage on port 5432

### Server Configuration

```bash
# Required
export OPENROUTER_API_KEY="sk-or-your-key"
export CEMS_DATABASE_URL="postgresql://..."

# Optional
export CEMS_ADMIN_KEY="admin-key-for-user-management"
export CEMS_QDRANT_URL="http://cems-qdrant:6333"
```

### Creating User API Keys

```bash
curl -X POST http://localhost:8765/admin/users \
  -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "colleague-name"}'
# Returns: {"api_key": "cems_usr_abc123..."}
```

## Features

- **Dual-Layer Memory**: Personal (per-user) and Shared (team) namespaces
- **Mem0 Backend**: Automatic fact extraction and deduplication
- **Knowledge Graph**: Kuzu-based relationship tracking
- **Scheduled Maintenance**:
  - Nightly: Merge duplicates, promote frequent memories
  - Weekly: Compress old memories, prune stale ones
  - Monthly: Rebuild embeddings, archive dead memories
- **5-Stage Retrieval**: Query synthesis, vector search, graph traversal, relevance filtering, token-budgeted assembly

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
