# /share

Add a memory to the shared memory store via CEMS.

## Usage

```
/share <fact or information to share with all users>
```

## Examples

```
/share API endpoints follow REST conventions with versioning (/api/v1/...)
/share Deploy process: merge to main, wait for CI, then run deploy script
/share --category architecture Microservices communicate via message queue
```

## Execution

1. **Detect project context**:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]//' | sed 's/\.git$//'
   ```

2. **Try MCP** (preferred):
   ```
   mcp__cems__memory_add with:
   - content: <input>
   - scope: "shared"
   - category: <specified or "general">
   - source_ref: "project:<org/repo>"
   ```

3. **If MCP unavailable**, use CLI:
   ```bash
   cems add "<content>" --category <cat> --scope shared
   ```

Shared memories are visible to all users on the CEMS instance.

## Related Skills

- `/store` - Add to personal memory
- `/recall` - Search memories (both personal and shared)
