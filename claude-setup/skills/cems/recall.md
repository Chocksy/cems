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

1. Converts your query to embeddings
2. Searches both personal and shared memory (by default)
3. Ranks results by relevance score
4. Applies time decay (recent memories rank higher)
5. Returns the most relevant matches

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
- `category`: Optional filter
- `limit`: Max results (default 5)

## Related Skills

- `/remember` - Add personal memories
- `/share` - Add shared memories
- `/forget` - Remove memories
- `/context` - Show memory system status
