# CEMS — Persistent Memory for AI Agents

Give your Cursor AI agent persistent memory across sessions. CEMS stores preferences, conventions, architecture decisions, and debugging insights — then recalls them when relevant.

## What You Get

- **Memory search** — Agent recalls past decisions, preferences, and patterns before starting work
- **Memory storage** — Agent stores important learnings as it discovers them
- **Gate rules** — Block or warn on dangerous shell commands and MCP operations
- **Session tracking** — Observer daemon captures high-level session observations automatically

## Setup

### 1. Get an API Key

Sign up at [cems.chocksy.com](https://cems.chocksy.com) to get your API key.

### 2. Set Environment Variable

Add to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):

```bash
export CEMS_API_KEY="cems_ak_your_api_key_here"
```

### 3. Install Plugin

Click **"Add to Cursor"** on the marketplace page, or run `/add-plugin cems` in the Cursor editor.

## How It Works

### MCP Tools

The plugin connects to the CEMS API via HTTP and exposes these tools:

| Tool | Purpose |
|------|---------|
| `memory_search` | Search memories with semantic retrieval, graph traversal, and time decay |
| `memory_add` | Store a memory (personal or shared scope) |
| `memory_forget` | Archive or permanently delete a memory |
| `memory_update` | Update an existing memory's content |
| `memory_maintenance` | Run consolidation, summarization, or reindex jobs |

### Rules

The plugin includes an always-on rule that instructs the agent to search memory before starting tasks and store important discoveries proactively. This compensates for Cursor not having per-prompt context injection hooks.

### Hooks

| Hook | Event | Purpose |
|------|-------|---------|
| Gate rules (shell) | `beforeShellExecution` | Check commands against your gate rules — block or warn |
| Gate rules (MCP) | `beforeMCPExecution` | Check MCP tool calls against gate rules |
| Observer signal | `stop` | Signal the observer daemon to finalize session summary |

### Skills

| Skill | Purpose |
|-------|---------|
| `cems-remember` | Store memories with project context and categories |
| `cems-recall` | Search memories with tips for effective retrieval |

## Self-Hosted / Alternative Setup

If you run your own CEMS server, override the MCP URL in Cursor settings:

**Settings > Tools & MCP Servers > cems > Edit Config**

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "https://your-server.com/mcp",
      "headers": {
        "Authorization": "Bearer ${CEMS_API_KEY}"
      }
    }
  }
}
```

Or use the stdio transport (requires `pip install cems`):

```json
{
  "mcpServers": {
    "cems": {
      "command": "cems-mcp",
      "args": []
    }
  }
}
```

## Team Memory

For shared team memory, add the `X-Team-ID` header:

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "https://mcp-cems.chocksy.com/mcp",
      "headers": {
        "Authorization": "Bearer ${CEMS_API_KEY}",
        "X-Team-ID": "your-team-name"
      }
    }
  }
}
```

## Requirements

- Cursor 2.5+
- `CEMS_API_KEY` environment variable
- For gate rule hooks: Python 3.11+ and [uv](https://docs.astral.sh/uv/)

## Links

- [CEMS GitHub](https://github.com/Chocksy/cems)
- [Documentation](https://github.com/Chocksy/cems#readme)
- [Report Issues](https://github.com/Chocksy/cems/issues)
