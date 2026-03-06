# CEMS Client Setup

## Install

You need two things from your CEMS admin: a **server URL** and an **API key**.

### Option A: Interactive install (recommended)

```bash
curl -fsSL https://getcems.com/install.sh | bash
```

Asks for your API URL and key, then lets you choose IDEs to configure.

### Option B: Non-interactive

```bash
CEMS_API_KEY=your-key CEMS_API_URL=https://cems.example.com \
  curl -fsSL https://getcems.com/install.sh | bash
```

### Option C: From source

```bash
git clone https://github.com/chocksy/cems.git && cd cems
./install.sh
```

### Option D: Skills only (any agent)

```bash
npx skills add Chocksy/cems
```

Installs recall, remember, and foundation skills. You still need a CEMS server and API key separately.

## What the installer does

1. Installs [uv](https://docs.astral.sh/uv/) if missing
2. Installs the CEMS CLI (`cems`, `cems-server`, `cems-observer`) via `uv tool install`
3. Runs `cems setup` which lets you choose IDEs to configure:
   - **Claude Code** (`--claude`): 6 hooks, 6 skills, 2 commands, settings.json config
   - **Cursor** (`--cursor`): 3 hooks, 5 skills, MCP config in `mcp.json`
   - **Codex** (`--codex`): 3 commands, 2 skills, MCP config in `config.toml`
   - **Goose** (`--goose`): MCP extension in `config.yaml`
4. Saves credentials to `~/.cems/credentials` (chmod 600)
5. Saves IDE choices to `~/.cems/install.conf` (used by `cems update`)

## CLI Commands

```bash
cems --version                       # Check version
cems health                          # Server health check
cems status                          # System status + stats
cems search "Docker port binding"    # Search memories
cems add "Always use port 8080"      # Store a memory
cems list                            # List recent memories
cems delete <id>                     # Soft-delete a memory
cems debug                           # Debug dashboard (see what hooks inject)
cems rule add                        # Interactive gate rule wizard
cems rule load --kind constitution   # Load default rule bundle
cems maintenance --job consolidation # Trigger maintenance manually
cems update                          # Update CLI + re-deploy hooks
cems update --hooks                  # Re-deploy hooks only
cems uninstall                       # Remove hooks from IDE
cems uninstall --all                 # Remove everything including credentials
```

## Skills (Slash Commands)

```
/remember I prefer Python for backend development
/remember The database uses snake_case column names
/recall What are my coding preferences?
/share API endpoints follow REST conventions with /api/v1/...
/forget abc123
/context
```

Available in Claude Code, Cursor, and Codex.

## Updating

CEMS auto-updates when you start a new Claude Code session. The SessionStart hook pulls the latest version in the background if your install is >24 hours old.

Manual update:

```bash
cems update          # Pull latest + re-deploy hooks/skills
cems update --hooks  # Re-deploy hooks only (skip package upgrade)
```

Disable auto-update: set `CEMS_AUTO_UPDATE=0` in `~/.cems/credentials`.

## Credentials

Stored in `~/.cems/credentials` (chmod 600). Checked in order:
1. CLI flags: `--api-url`, `--api-key`
2. Environment: `CEMS_API_URL`, `CEMS_API_KEY`
3. Credentials file: `~/.cems/credentials`

## How Hooks Work

After `cems setup`, your IDE automatically:
- **On session start**: Loads your profile (preferences, guidelines, gate rules)
- **On each prompt**: Searches memory for relevant context, injects it
- **On tool use**: Applies gate rules (block/warn), extracts learnings
- **On session end**: Writes an observer signal for session summarization

No manual steps needed. Memories build up and improve over time.

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

### Re-install

```bash
cems setup    # Re-runs the full setup
```
