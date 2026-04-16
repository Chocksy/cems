# Remove team_id from CEMS

**Date:** 2026-04-15
**Status:** Brainstorm complete, ready for planning

## What We're Building

Remove the `team_id` concept entirely from CEMS. Replace team-gated shared memory with instance-wide visibility: shared memories are visible to ALL users on the instance, personal memories remain private to the creator. A new `CEMS_DEFAULT_SCOPE` env var controls whether new memories default to "shared" (company instances) or "personal" (individual instances).

## Why This Approach

- `team_id` adds complexity but delivers thin value — hooks don't even send it, wiki ignores it, maintenance jobs ignore it, observer ignores it
- The actual functional surface of `team_id` is a single method: `FilterBuilder.add_ownership_filter()`
- The "one collective memory" vision is simpler and more powerful: new members automatically see all shared knowledge
- Configurable default scope preserves flexibility for different deployment contexts

## Key Decisions

1. **Remove `team_id` everywhere** — column, config, middleware, headers, credentials
2. **Keep `scope` (personal/shared)** — personal stays private, shared means visible to all users on the instance
3. **Add `CEMS_DEFAULT_SCOPE` env var** — defaults to `"shared"`, can be `"personal"` per-instance
4. **No data migration** — existing rows keep their current scope; ENV controls default for new memories only
5. **DROP tables:** `teams`, `team_members`, `api_keys` (unused stub)
6. **DROP columns:** `memory_documents.team_id`, `index_jobs.team_id`, `index_patterns.team_id`
7. **Keep `index_jobs`/`index_patterns`** tables — actively used, just remove FK columns
8. **Wiki uses shared memories from all users** — enables collective knowledge base

## New FilterBuilder Logic

```
scope="personal"  -> WHERE user_id = X AND scope = 'personal'
scope="shared"    -> WHERE scope = 'shared'                     # everyone sees all shared
scope="both"      -> WHERE user_id = X OR scope = 'shared'      # personal + all shared
```

## Refactor Scope — Files to Change

### Database Layer
- `src/cems/db/database.py` — migration to drop team_id column, drop team tables, drop api_keys
- `src/cems/db/document_store.py` — remove team_id param from ~12 methods
- `src/cems/db/filter_builder.py` — simplify add_ownership_filter() (core change)
- `src/cems/db/models.py` — remove Team, TeamMember, ApiKey models
- `deploy/init.sql` — update reference schema

### Core Memory
- `src/cems/config.py` — remove team_id, add default_scope field
- `src/cems/mixins/write.py` — remove team_id logic
- `src/cems/mixins/search.py` — remove team_id logic
- `src/cems/mixins/crud.py` — remove team_id logic
- `src/cems/mixins/metadata.py` — remove team_id logic

### API Layer
- `src/cems/server.py` — remove team resolution from UserContextMiddleware
- `src/cems/api/deps.py` — remove request_team_id contextvar, simplify get_memory() cache
- `src/cems/api/handlers/memory.py` — simplify promote (no team_id), fix shared summary
- `src/cems/api/handlers/me.py` — remove /api/me/teams endpoint
- `src/cems/api/handlers/wiki.py` — change queries to include shared memories from all users
- `src/cems/admin/services.py` — remove TeamService class
- `src/cems/admin/routes.py` — remove team admin endpoints, _resolve_team_id()

### Client / CLI / Config
- `src/cems/client.py` — remove team_id field, X-Team-ID header
- `src/cems/commands/setup.py` — remove _discover_team(), stop writing CEMS_TEAM_ID
- `src/cems/cli.py` — remove team_id from Click context
- `src/cems/cli_utils.py` — remove team_id from CEMSClient construction

### MCP / Hooks
- `src/cems/mcp_stdio.py` — remove TEAM_ID and X-Team-Id header
- `mcp-wrapper/src/index.ts` — remove x-team-id header forwarding
- `hooks/utils/credentials.py` — remove CEMS_TEAM_ID from credentials reading
- `src/cems/shared/credentials.py` — same
- `src/cems/data/claude/skills/cems/share.md` — update docs (no team required)

### Tests
- All tests referencing team_id need updating

## Open Questions

- Should we backfill existing `scope='personal'` memories to `'shared'` on specific instances? (Currently: no, admin can do it manually via SQL if needed)
- Should wiki compilation jobs also run across all shared memories (not just user's own)? (Likely yes — but scope for planning phase)

## Risk Assessment

- **Low risk:** Hooks already don't use team_id, wiki already ignores it, maintenance already ignores it
- **Medium risk:** FilterBuilder change is the core behavioral shift — needs thorough testing
- **Medium risk:** Migration must handle FK constraints in correct order (drop FKs before dropping tables)
- **Low risk:** Client/MCP changes are removal-only (less can go wrong)
