# Foundation - View Foundation Guidelines

Display the foundation guidelines (rules, principles, constraints) stored in CEMS.

## Usage

```
/foundation
```

Or with a project scope:

```
/foundation --project org/repo
```

## Arguments

- `$ARGUMENTS` - Optional project identifier (org/repo format)

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | (auto-detect) | Project scope for project-specific guidelines |

## Execution

When this skill is invoked:

1. **Parse arguments** to extract:
   - `--project <org/repo>` if provided
   - If no project specified, try to auto-detect from the current git remote

2. **Search for foundation guidelines** via MCP:
   ```
   Use mcp__cems__memory_search with:
   - query: "foundation guidelines rules principles constitution"
   - max_results: 20
   - scope: "both"
   ```

3. **Filter results** to only include memories that:
   - Have tags containing "foundation" or "constitution"
   - OR have category "guidelines"
   - These are the foundational principles the user has defined

4. **Present results** clearly:
   ```
   ## Foundation Guidelines

   Found N foundation guidelines:

   1. <guideline content>
   2. <guideline content>
   ...

   These are your foundational principles. They apply to all sessions.
   ```

5. **If no guidelines found**, respond:
   "No foundation guidelines found. Use `cems rule add` to create foundation guidelines, or use `/remember` with tags like 'foundation' to store principles."

## Examples

User: `/foundation`

You should:
1. Call `mcp__cems__memory_search` with query="foundation guidelines rules principles constitution", max_results=20
2. Filter to only foundation-tagged or guidelines-category memories
3. Display the guidelines

User: `/foundation --project myorg/myrepo`

You should:
1. Call `mcp__cems__memory_search` with query="foundation guidelines rules principles constitution", max_results=20, project="myorg/myrepo"
2. Filter and display

## Context

Foundation guidelines are stable rules that apply across all sessions. They are typically:
- Coding standards and conventions
- Project-specific constraints
- Team agreements and principles
- Quality gates and review criteria

They differ from regular memories in that they are meant to be always-on constraints, not just recalled information.
