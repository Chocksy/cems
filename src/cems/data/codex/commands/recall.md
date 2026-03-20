# Recall - Search CEMS Memory

Search your memories for relevant information from past sessions.

## Usage

```
/recall <query>
```

Or with options:

```
/recall --limit 10 authentication patterns
/recall --scope shared team conventions
```

## Arguments

- `$ARGUMENTS` - The search query (natural language)

## Execution

When this command is invoked:

1. **Parse the arguments** to extract:
   - `--limit <N>` if provided (default: 5)
   - `--scope <personal|shared|both>` if provided (default: "both")
   - The remaining text is the search query

2. **Detect project context**:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]//' | sed 's/\.git$//'
   ```

3. **Search via MCP**:
   ```
   Use memory_search with:
   - query: <the query>
   - max_results: <limit>
   - scope: <scope>
   - project: <detected org/repo>
   ```

4. **If MCP unavailable**, use CLI:
   ```bash
   cems search "<query>" --limit <N>
   ```

5. **For truncated results**, fetch full content:
   - MCP: `memory_get` with `memory_id`
   - CLI: `cems get <memory_id>`

6. **Present results** with category and ID.

## If No Results

"No memories found for '<query>'. Try broader terms or `/remember` to store new memories."
