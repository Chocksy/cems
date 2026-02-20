# CEMS Installation Guide

Install CEMS (Continuous Evolving Memory System) for your AI coding tools.

## Prerequisites

- macOS or Linux
- An API key from your CEMS admin (starts with `cems_usr_`)
- Your CEMS server URL (e.g., `https://cems.yourcompany.com`)

## One-Command Install

The remote installer handles everything: installs `uv`, installs the CEMS package, and configures your tool.

### Claude Code (default)

```bash
CEMS_API_KEY=your-key \
  curl -fsSL https://getcems.com/install.sh | bash
```

### Goose

```bash
CEMS_TOOL=goose CEMS_API_KEY=your-key \
  curl -fsSL https://getcems.com/install.sh | bash
```

### Cursor

```bash
CEMS_TOOL=cursor CEMS_API_KEY=your-key \
  curl -fsSL https://getcems.com/install.sh | bash
```

### All Tools

```bash
CEMS_TOOL=all CEMS_API_KEY=your-key \
  curl -fsSL https://getcems.com/install.sh | bash
```

Set `CEMS_API_URL` if your server is not at the default URL:

```bash
CEMS_TOOL=goose CEMS_API_KEY=your-key CEMS_API_URL=https://cems.example.com \
  curl -fsSL https://getcems.com/install.sh | bash
```

## Manual Install

If you prefer step-by-step:

### 1. Install the CEMS package

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install CEMS
uv tool install "cems @ git+https://github.com/chocksy/cems.git"
```

### 2. Run setup for your tool

```bash
# Interactive (prompts for tool choice + credentials)
cems setup

# Or specify tool and credentials directly
cems setup --claude --api-url https://cems.example.com --api-key your-key
cems setup --goose --api-url https://cems.example.com --api-key your-key
cems setup --cursor --api-url https://cems.example.com --api-key your-key
```

### 3. Add to your shell profile

Add to `~/.zshrc` or `~/.bashrc`:

```bash
eval "$(cems env)"
```

Restart your shell and IDE.

### 4. Verify

```bash
cems health
```

## What Gets Installed

### Claude Code

| Component | Location |
|-----------|----------|
| Hooks | `~/.claude/hooks/cems_*.py` |
| Skills | `~/.claude/skills/cems/` |
| Settings | `~/.claude/settings.json` (merged) |
| MCP server | `~/.claude.json` (HTTP transport) |
| Credentials | `~/.cems/credentials` |

Hooks provide automatic memory search on every prompt, session learning via the observer daemon, tool-use gating, and profile injection at session start.

### Goose

| Component | Location |
|-----------|----------|
| Extension config | `~/.config/goose/config.yaml` |
| Credentials | `~/.cems/credentials` |

Goose connects to CEMS via a stdio MCP server (`cems-mcp`). The extension gives Goose access to `memory_search`, `memory_add`, `memory_forget`, `memory_update`, and `memory_maintenance` tools. Your profile is injected into Goose's system prompt at startup.

The observer daemon also passively learns from Goose sessions (polling Goose's SQLite DB every 30 seconds).

### Cursor

| Component | Location |
|-----------|----------|
| Hooks | `~/.cursor/hooks/` |
| Config | `~/.cursor/hooks.json` |
| Credentials | `~/.cems/credentials` |

## Observer Daemon

The observer daemon runs in the background and passively learns from your coding sessions across all tools. Start it with:

```bash
cems-observer
```

It discovers sessions from Claude Code, Codex CLI, Cursor, and Goose automatically. Sessions idle for 5+ minutes are finalized.

## Updating

```bash
uv tool install "cems @ git+https://github.com/chocksy/cems.git" --force
cems setup --claude  # or --goose, --cursor, --all
```

## Uninstalling

```bash
# Remove hooks and settings (keeps credentials)
cems uninstall

# Remove everything including credentials
cems uninstall --all

# Remove the package
uv tool uninstall cems
```

## Troubleshooting

**`cems: command not found`** — Add `~/.local/bin` to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Connection errors** — Check your server URL and API key:
```bash
cems health
```

**Goose doesn't show CEMS tools** — Verify the extension is in your config:
```bash
grep -A5 "cems:" ~/.config/goose/config.yaml
```

**Observer not finding sessions** — Check the daemon is running:
```bash
cems-observer --once  # Single scan, useful for debugging
```
