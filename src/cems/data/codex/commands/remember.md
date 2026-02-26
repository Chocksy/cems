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

2. **Detect the current project** from the working directory:
   - Run `git remote get-url origin` to extract `org/repo` format
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`
   - Pass as `source_ref: "project:org/repo"`

3. **Store via MCP**:
   ```
   Use memory_add with:
   - content: <the content>
   - category: <extracted or "learnings">
   - scope: <extracted or "personal">
   - source_ref: "project:<detected org/repo>"
   ```

4. **Confirm to user** with what was stored and the memory ID if available.

## Examples

User: `/remember TypeScript hooks need to be compiled before they work`

You should:
1. Detect project from git remote
2. Call `memory_add` with content="TypeScript hooks need to be compiled before they work", category="learnings", source_ref="project:org/repo"
3. Report: "Stored memory: TypeScript hooks need to be compiled before they work"

User: `/remember --category decisions --scope shared We use pnpm not npm`

You should:
1. Detect project from git remote
2. Call `memory_add` with content="We use pnpm not npm", category="decisions", scope="shared", source_ref="project:org/repo"
3. Report: "Stored shared memory (decisions): We use pnpm not npm"
