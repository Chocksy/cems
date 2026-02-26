---
name: cems-recall
description: Search memories for relevant past context, decisions, and patterns
---

# Search Memories

Use the `memory_search` MCP tool to find relevant memories.

## Usage

When you need context before starting work, or the user asks to recall something:

1. **Detect the current project** from the working directory:
   - Run `git remote get-url origin` to extract `org/repo` format
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`
   - Pass this as the `project` parameter to boost same-project results
2. Formulate a natural language query
3. Call `memory_search` with appropriate parameters (always include `project`)
4. If any results are truncated, use `memory_get` to fetch the full content
5. Use the results to inform your work

## MCP Tool Call

```json
{
  "tool": "memory_search",
  "arguments": {
    "query": "authentication patterns in this project",
    "scope": "both",
    "max_results": 10,
    "max_tokens": 4000,
    "enable_graph": true,
    "enable_query_synthesis": true,
    "project": "org/repo"
  }
}
```

## Fetching Truncated Results

When search results are truncated (content ends with `...`), fetch the full document:

```json
{
  "tool": "memory_get",
  "arguments": {
    "memory_id": "the-memory-id-from-search-result"
  }
}
```

## Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `query` | (required) | Natural language search query |
| `scope` | `"both"` | `"personal"`, `"shared"`, or `"both"` |
| `max_results` | `10` | Maximum results (1-20) |
| `max_tokens` | `4000` | Token budget for results |
| `enable_graph` | `true` | Include related memories via graph traversal |
| `enable_query_synthesis` | `true` | Expand query with LLM for better retrieval |
| `raw` | `false` | Debug mode: bypass relevance filtering |
| `project` | (auto-detect) | Project ID (org/repo) — always pass this to boost same-project results |

## When to Search

- **Before coding**: Search for relevant patterns, conventions, preferences
- **Before debugging**: Search for similar issues and solutions
- **Before decisions**: Search for past architectural choices
- **Before code review**: Search for team conventions

## Search Tips

- **Always pass `project`** — auto-detect from git remote to filter cross-project noise
- Use natural language: "how do we handle authentication" > "auth"
- Be specific: "Python backend database conventions" > "conventions"
- The system uses semantic matching, not just keywords
- Results include relevance scores and time decay ranking
- Use `raw: true` to debug retrieval when results seem wrong
