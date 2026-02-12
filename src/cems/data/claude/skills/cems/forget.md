# /forget

Remove a memory from the system.

## Usage

```
/forget <memory_id>
```

## Examples

```
/forget abc123def456
/forget --hard abc123def456  # Permanent deletion
```

## How It Works

This skill uses the CEMS `memory_forget` MCP tool to remove memories.

By default, memories are **archived** (soft delete), which means:
- They won't appear in searches
- They can be recovered if needed
- They're preserved for audit purposes

Use `--hard` for permanent deletion when you want the memory completely removed.

## Finding Memory IDs

Memory IDs are returned when you:
- Add a memory (`/remember`)
- Search memories (`/recall`)
- List memories

Example workflow:
```
/recall old preferences
# Returns memories with IDs like "abc123..."

/forget abc123
# Archives that memory
```

## Options

```
/forget <memory_id>         # Archive (soft delete)
/forget --hard <memory_id>  # Permanent delete
```

## MCP Tool Used

`memory_forget` with:
- `memory_id`: The ID to remove
- `hard`: true for permanent deletion

## Related Skills

- `/recall` - Find memories (and their IDs)
- `/remember` - Add memories
