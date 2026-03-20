# /forget

Remove a memory from the CEMS system.

## Usage

```
/forget <memory_id>
/forget --hard <memory_id>
```

## Examples

```
/forget abc123def456
/forget --hard abc123def456
```

## Execution

1. **Try MCP** (preferred):
   ```
   mcp__cems__memory_forget with:
   - memory_id: <the ID>
   - hard: false (or true if --hard)
   ```

2. **If MCP unavailable**, use CLI:
   ```bash
   cems delete <memory_id>
   cems delete --hard <memory_id>
   ```

By default, memories are **soft-deleted** (archived). Use `--hard` for permanent deletion.

## Finding Memory IDs

Memory IDs are returned by `/recall` and `/store`. Example workflow:

```
/recall old preferences       → returns IDs like "abc123..."
/forget abc123                → archives that memory
```

## Related Skills

- `/recall` - Find memories (and their IDs)
- `/store` - Add memories
