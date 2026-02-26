# /recall

Search your memories for relevant information.

## Usage

```
/recall <search query>
```

## Examples

```
/recall What do I prefer for backend development?
/recall database conventions
/recall How do we handle authentication?
/recall deployment process
```

## How It Works

This skill uses the CEMS `memory_search` MCP tool to find relevant memories using semantic search. It:

1. Detects the current project from git remote (e.g., "Chocksy/cems")
2. Converts your query to embeddings
3. Searches both personal and shared memory (by default)
4. Boosts same-project results, penalizes cross-project noise
5. Ranks results by relevance score with time decay
6. Returns the most relevant matches

Each search also updates access tracking, which helps the maintenance system identify important vs stale memories.

## Options

```
/recall --scope personal my coding preferences
/recall --scope shared team conventions
/recall --scope both authentication patterns
/recall --category decisions database choices
/recall --limit 10 all architecture decisions
```

## Search Tips

- Use natural language queries for best results
- Be specific when possible ("Python backend preferences" vs "preferences")
- Category filters help narrow results
- The system finds semantic matches, not just keyword matches

## MCP Tool Used

`memory_search` with:
- `query`: Your search text
- `scope`: "personal", "shared", or "both" (default)
- `project`: Auto-detected from git remote (org/repo) — boosts same-project memories
- `max_results`: Max results (default 5)

For truncated results, use `memory_get` with:
- `memory_id`: The ID from the truncated search result

## Related Skills

- `/remember` - Add personal memories
- `/share` - Add shared memories
- `/forget` - Remove memories
- `/context` - Show memory system status
