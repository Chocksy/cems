# /store

Add a memory to your personal memory store via CEMS.

**IMPORTANT:** This stores to CEMS (your long-term memory server), NOT to Claude Code's local markdown memory files. Always prefer CEMS for information that should persist across all projects and sessions.

## Usage

```
/store <fact or information to store>
```

## Examples

```
/store I prefer Python for backend development
/store The database schema uses snake_case for column names
/store --category decisions We chose PostgreSQL for the main database
```

## How It Works

This skill uses the CEMS `memory_add` tool to store information. The memory system automatically:

1. Extracts atomic facts from your input
2. Checks for existing similar memories
3. Decides whether to ADD (new), UPDATE (modify existing), or skip (duplicate)
4. Stores with timestamp and access tracking

## Options

```
/store --category preferences I like dark mode in all editors
/store --category decisions We chose PostgreSQL for the main database
/store --scope shared We use pnpm not npm
/store --tags auth,security Session tokens expire after 1 hour
```

## Execution

1. **Detect project context**:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]//' | sed 's/\.git$//'
   ```

2. **Try MCP** (preferred):
   ```
   mcp__cems__memory_add with:
   - content: <input>
   - scope: "personal" (or "shared" if --scope shared)
   - category: <specified or "general">
   - source_ref: "project:<org/repo>"
   ```

3. **If MCP unavailable**, use CLI:
   ```bash
   cems add "<content>" --category <cat> --scope <scope>
   ```

## Related Skills

- `/recall` - Search your memories
- `/forget` - Remove a memory
- `/share` - Add to shared memory
