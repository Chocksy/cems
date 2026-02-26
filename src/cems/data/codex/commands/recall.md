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

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 5 | Maximum results to return |
| `--scope` | both | `personal`, `shared`, or `both` |

## Execution

When this skill is invoked:

1. **Parse the arguments** to extract:
   - `--limit <N>` if provided (default: 5)
   - `--scope <personal|shared|both>` if provided (default: "both")
   - The remaining text is the search query

2. **Detect the current project** from the working directory:
   - Run `git remote get-url origin` to extract `org/repo` format
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`
   - Pass this as the `project` parameter to boost same-project results

3. **Search via MCP**:
   ```
   Use memory_search with:
   - query: <the query>
   - max_results: <limit>
   - scope: <scope>
   - project: <detected project ID, e.g., "Chocksy/cems">
   ```

4. **For truncated results**, fetch the full document:
   ```
   Use memory_get with:
   - memory_id: <the truncated result's memory_id>
   ```

5. **Present results** in a clear format:
   ```
   ## Memory Recall: "<query>"

   Found N memories:

   ### 1. [category] (id: abc123)
   <content>

   ### 2. [category] (id: def456)
   <content>
   ```

## Examples

User: `/recall hook development`

You should:
1. Detect project from git remote (e.g., "Chocksy/cems")
2. Call `memory_search` with query="hook development", max_results=5, project="Chocksy/cems"
3. Format and display results

User: `/recall --limit 10 --scope personal typescript errors`

You should:
1. Detect project from git remote
2. Call `memory_search` with query="typescript errors", max_results=10, scope="personal", project="<detected>"
3. Format and display results

## If No Results

If no memories match, respond:
"No memories found for '<query>'. Try a different search term or use `/remember` to store new memories."
