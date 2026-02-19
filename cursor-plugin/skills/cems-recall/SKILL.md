---
name: cems-recall
description: Search memories for relevant past context, decisions, and patterns
---

# Search Memories

Use the `memory_search` MCP tool to find relevant memories.

## Usage

When you need context before starting work, or the user asks to recall something:

1. Formulate a natural language query
2. Call `memory_search` with appropriate parameters
3. Use the results to inform your work

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
    "enable_query_synthesis": true
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
| `project` | (optional) | Project ID to boost project-scoped memories |

## When to Search

- **Before coding**: Search for relevant patterns, conventions, preferences
- **Before debugging**: Search for similar issues and solutions
- **Before decisions**: Search for past architectural choices
- **Before code review**: Search for team conventions

## Search Tips

- Use natural language: "how do we handle authentication" > "auth"
- Be specific: "Python backend database conventions" > "conventions"
- The system uses semantic matching, not just keywords
- Results include relevance scores and time decay ranking
- Use `raw: true` to debug retrieval when results seem wrong
