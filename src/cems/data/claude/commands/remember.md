# Remember - Store Memory in CEMS

Store a learning, pattern, decision, or any information for future recall.

## Usage

```
/remember <what to remember>
```

Or with options:

```
/remember --category decisions The auth system uses JWT tokens
/remember --scope shared Team prefers Tailwind over CSS modules
```

## Arguments

- `$ARGUMENTS` - The content to remember

## Categories

| Category | Use For |
|----------|---------|
| `general` | Default, miscellaneous info |
| `decisions` | Architectural/design decisions |
| `patterns` | Code patterns, conventions |
| `errors` | Error fixes, gotchas |
| `preferences` | User/team preferences |
| `learnings` | Things learned during sessions |

## Execution

When this skill is invoked:

1. **Parse the arguments** to extract:
   - `--category <cat>` if provided (default: "learnings")
   - `--scope <personal|shared>` if provided (default: "personal")
   - The remaining text is the content to remember

2. **Try MCP first** (fast path):
   ```
   Use mcp__cems__memory_add with:
   - content: <the content>
   - category: <extracted or "learnings">
   - scope: <extracted or "personal">
   ```

3. **If MCP fails**, fall back to CLI:
   ```bash
   cd ~/Development/cems && uv run cems add "<content>" --category <cat> --scope <scope>
   ```

4. **Confirm to user** with what was stored and the memory ID if available.

## Examples

User: `/remember TypeScript hooks need to be compiled before they work`

You should:
1. Call `mcp__cems__memory_add` with content="TypeScript hooks need to be compiled before they work", category="learnings"
2. Report: "Stored memory: TypeScript hooks need to be compiled before they work"

User: `/remember --category decisions --scope shared We use pnpm not npm`

You should:
1. Call `mcp__cems__memory_add` with content="We use pnpm not npm", category="decisions", scope="shared"
2. Report: "Stored shared memory (decisions): We use pnpm not npm"
