# Store - Save to CEMS Memory

Store a learning, pattern, decision, or any information for future recall across sessions.

**IMPORTANT:** This stores to CEMS (your long-term memory server), NOT to Claude Code's local markdown memory files. Always prefer this for information that should persist across all projects and sessions.

## Usage

```
/store <what to store>
```

Or with options:

```
/store --category decisions The auth system uses JWT tokens
/store --scope shared Team prefers Tailwind over CSS modules
```

## Arguments

- `$ARGUMENTS` - The content to store

## Execution

When this command is invoked:

1. **Parse the arguments** to extract:
   - `--category <cat>` if provided (default: "learnings")
   - `--scope <personal|shared>` if provided (default: "personal")
   - The remaining text is the content to store

2. **Detect project context**:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]//' | sed 's/\.git$//'
   ```
   Use the result as `source_ref: "project:<org/repo>"`

3. **Try MCP first** (preferred):
   ```
   Use mcp__cems__memory_add with:
   - content: <the content>
   - category: <category>
   - scope: <scope>
   - source_ref: "project:<detected org/repo>"
   ```

4. **If MCP unavailable**, use CLI:
   ```bash
   cems add "<content>" --category <cat> --scope <scope>
   ```

5. **Confirm** with what was stored.

## Categories

| Category | Use For |
|----------|---------|
| `general` | Default, miscellaneous info |
| `decisions` | Architectural/design decisions |
| `patterns` | Code patterns, conventions |
| `errors` | Error fixes, gotchas |
| `preferences` | User/team preferences |
| `learnings` | Things learned during sessions |

## Examples

User: `/store TypeScript hooks need to be compiled before they work`
→ Call MCP `memory_add` or `cems add "TypeScript hooks need to be compiled before they work" --category learnings`
→ "Stored: TypeScript hooks need to be compiled before they work"

User: `/store --category decisions --scope shared We use pnpm not npm`
→ Call MCP `memory_add` or `cems add "We use pnpm not npm" --category decisions --scope shared`
→ "Stored shared memory (decisions): We use pnpm not npm"
