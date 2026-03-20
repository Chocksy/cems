# /context

Show the current CEMS memory system status.

## Usage

```
/context
```

## Execution

1. **Try MCP** (preferred):
   - Read resource `memory://status` for system status
   - Read resource `memory://personal/summary` for personal summary
   - Read resource `memory://shared/summary` for shared summary

2. **If MCP unavailable**, use CLI:
   ```bash
   cems status
   ```

3. **Present** user ID, team ID, memory counts, category breakdown, and scheduler status.

## Related Skills

- `/recall` - Search memories
- `/store` - Add personal memories
- `/share` - Add shared memories
