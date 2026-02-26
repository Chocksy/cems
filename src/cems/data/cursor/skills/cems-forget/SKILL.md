---
name: cems-forget
description: Remove a memory from CEMS. Use when the user asks to forget, remove, delete, or clear a memory.
---

# Forget a Memory

Use the `memory_forget` MCP tool to remove memories.

## MCP Tool Call

```json
{
  "tool": "memory_forget",
  "arguments": {
    "memory_id": "abc123-def456-...",
    "hard_delete": false
  }
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_id` | (required) | The ID of the memory to remove |
| `hard_delete` | `false` | If true, permanently delete; if false, soft delete |

## Finding Memory IDs

Memory IDs are returned when you:
1. Add a memory (`cems-remember`)
2. Search memories (`cems-recall`)

## Soft vs Hard Delete

- **Soft delete** (default): Marks as deleted, excluded from search results
- **Hard delete**: Permanent removal, cannot be undone

Always confirm with the user before hard deleting.
