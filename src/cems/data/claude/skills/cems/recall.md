# /recall

Search your CEMS memories for relevant information.

## Usage

```
/recall <search query>
```

## Examples

```
/recall What do I prefer for backend development?
/recall database conventions
/recall --scope shared deployment process
/recall --limit 10 all architecture decisions
```

## Execution

1. **Detect project context**:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]//' | sed 's/\.git$//'
   ```

2. **Try MCP** (preferred):
   ```
   mcp__cems__memory_search with:
   - query: <search text>
   - scope: "both" (or specified)
   - max_results: 5 (or specified --limit)
   - project: <detected org/repo>
   ```

3. **If MCP unavailable**, use CLI:
   ```bash
   cems search "<query>" --limit <N>
   ```

4. **For results with `has_detailed: true` or `truncated: true`**, fetch full content via MCP:
   ```
   mcp__cems__memory_get with memory_id
   ```
   The response includes `content_detailed` (original full text) when the memory has been distilled.

5. **Present results** clearly with category and ID.

## Options

```
/recall --scope personal my coding preferences
/recall --scope shared conventions
/recall --category decisions database choices
/recall --limit 10 all architecture decisions
```

## Search Tips

- Use natural language for best results
- Be specific ("Python backend preferences" vs "preferences")
- The system uses semantic matching, not just keywords

## Related Skills

- `/store` - Add personal memories
- `/share` - Add shared memories
- `/forget` - Remove memories
- `/context` - Show memory system status
