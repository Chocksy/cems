---
name: cems-context
description: Show CEMS memory system status and context. Use when the user asks about memory status or memory counts.
---

# Memory System Context

Show the current memory system status and overview.

## How to Use

Search for a broad overview to gather statistics:

```json
{
  "tool": "memory_search",
  "arguments": {
    "query": "all memories overview status",
    "scope": "both",
    "max_results": 20,
    "raw": true
  }
}
```

Then summarize:
- Count of personal vs shared memories
- Categories present
- Recent additions

## Information to Present

1. **Memory Overview**: Personal count, shared count
2. **Categories**: What categories are in use and their counts
3. **Recent Activity**: Recently added or accessed memories

## Example Output

```
CEMS Memory Status
==================
Personal Memories: ~15
Shared Memories: ~8

Categories: preferences (5), decisions (3), patterns (4), general (9)

Recent: "API endpoints follow REST v2 conventions" (conventions)
```
