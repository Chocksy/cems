# /context

Show the current memory system status and context.

## Usage

```
/context
```

## Output

Shows:
- Current user ID
- Team ID (if configured)
- Storage location
- Memory counts by scope
- Category breakdown
- Scheduler status

## Example Output

```
CEMS Memory System Status
========================================

User ID: razvan
Team ID: engineering
Storage: /Users/razvan/.cems

Personal Memories:
- Total: 47
- By category:
  - preferences: 12
  - decisions: 8
  - patterns: 15
  - context: 7
  - general: 5

Shared Memories:
- Total: 23
- By category:
  - conventions: 10
  - architecture: 8
  - processes: 5

Scheduler: Running
- Next consolidation: Tonight 3:00 AM
- Next summarization: Sunday 4:00 AM
- Next reindex: Feb 1 5:00 AM
```

## MCP Resources Used

- `memory://status` - System status
- `memory://personal/summary` - Personal memory summary
- `memory://shared/summary` - Shared memory summary

## Related Skills

- `/recall` - Search memories
- `/remember` - Add personal memories
- `/share` - Add shared memories
