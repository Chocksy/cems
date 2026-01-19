# /share

Add a memory to the shared team memory store.

## Usage

```
/share <fact or information to share with team>
```

## Examples

```
/share API endpoints follow REST conventions with versioning (/api/v1/...)
/share Deploy process: merge to main, wait for CI, then run deploy script
/share Architecture decision: microservices communicate via message queue
```

## How It Works

This skill uses the CEMS `memory_add` MCP tool with `scope: "shared"` to store information that should be accessible to all team members.

Shared memories are useful for:
- Team conventions and standards
- Architecture decisions (ADRs)
- Deployment processes
- Codebase patterns
- Onboarding knowledge

## Requirements

You must have `CEMS_TEAM_ID` configured for shared memory to work.

## Options

```
/share --category architecture We use hexagonal architecture pattern
/share --category conventions All dates are stored in UTC
/share --tags deploy,process Deploy requires manual approval for production
```

## MCP Tool Used

`memory_add` with:
- `content`: Your input
- `scope`: "shared"
- `category`: Default "general" or specified
- `tags`: As specified

## Related Skills

- `/remember` - Add to personal memory
- `/recall` - Search memories (both personal and shared)
