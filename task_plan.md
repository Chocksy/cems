# CEMS Multi-Team Support — v0.8.0

## Goal
Connect the team plumbing so multi-user orgs get automatic team context, promote personal→shared, and manage team memories in the dashboard.

## Phases

### Phase 1: Auto-resolve team_id from membership [complete]
- `src/cems/server.py` middleware: after user auth, if no X-Team-Id, look up teams
- If exactly 1 team → auto-set team_id; if 0 or >1 → leave None

### Phase 2: Promote endpoint (personal→shared) [complete]
- `src/cems/db/document_store.py` — `promote_document()` method
- `src/cems/api/handlers/memory.py` — `api_memory_promote` handler
- `src/cems/api/handlers/__init__.py` — export added
- `src/cems/server.py` — route registered + import added

### Phase 3: cems setup stores team_id [complete]
- `src/cems/api/handlers/me.py` — new `GET /api/me/teams` endpoint
- `src/cems/commands/setup.py` — after auth, query teams via API, store CEMS_TEAM_ID
- Inject X-Team-Id header into MCP configs (Claude, Cursor)

### Phase 4: Dashboard team memory management [complete]
- Scope toggle (All | Personal | Team) in header
- Scope badge on memory cards (color-coded)
- "Promote to Team" button on personal memories
- Pass scope param for team scope browsing

### Phase 5: MCP wrapper fixes [complete]
- Add x-team-id forwarding to get/forget/update/maintenance tools
- Add memory_promote tool

### Phase 6: Version bump + commit + release [in_progress]

### Phase 7: Devops tutorial [pending]
