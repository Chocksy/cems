---
title: "feat: Unified Wiki Dashboard"
type: feat
date: 2026-04-07
brainstorm: docs/brainstorms/2026-04-07-unified-wiki-dashboard-brainstorm.md
---

# Unified Wiki Dashboard

## Overview

Merge the two separate dashboards (/wiki and /dashboard) into a single Notion-like
knowledge portal. Wiki-first navigation with entity pages as landing, memories as
drill-down. Support team scope when `CEMS_TEAM_ID` is set. Google OAuth for web login.

## Problem Statement

Two disconnected dashboards serve the same data:
- `/dashboard` — flat memory list (edit, search, categories)
- `/wiki` — entity pages, graph, stats, health, timeline

Users navigate between them via a "Memories" link. No unified experience.
Team members can't discover each other's knowledge. No proper auth for web access.

## Proposed Solution

### Phase 1: Merge Dashboards (this plan)

Single app at `/` with sidebar navigation:
- **Wiki** (entity pages — the landing view)
- **All Memories** (current flat list from /dashboard)
- **Graph** (D3 knowledge graph)
- **Health** (lint, conflicts, orphans)
- **Settings** (API key, user info)

### Phase 2: Team Scope (separate plan)

Default to team scope when `CEMS_TEAM_ID` env var is set. Entity pages compile
from all team members. Wiki shows team knowledge.

### Phase 3: Google OAuth (separate plan)

Google login button. Auto-create user from email. Session-based web auth.

## Technical Approach

### What Changes

**Frontend**: Single `static/wiki/` app replaces both dashboards.
- Absorb `/dashboard` functionality into "All Memories" view
- Keep all existing wiki views (entities, graph, stats, health, timeline)
- Sidebar navigation replaces the tab bar
- URL routing: `/#wiki`, `/#wiki/:id`, `/#memories`, `/#graph`, `/#health`

**Backend**: Minimal changes.
- Mount unified app at `/` (or keep at `/wiki`, redirect `/dashboard` → `/wiki/#memories`)
- All API endpoints stay the same (`/api/memory/*`, `/api/wiki/*`)
- No new endpoints needed — memories list uses existing `/api/memory/list`

**Migration path**:
- Old `/dashboard` redirects to new unified app's memories view
- Old `/wiki` keeps working (same static files, same mount)
- API unchanged — hooks, CLI, MCP all continue working

### Implementation

#### 1. Sidebar Navigation

Replace the top tab bar with a left sidebar:

```
static/wiki/index.html:

┌────────┬──────────────────────────────────────┐
│ CEMS   │  [Content Area]                      │
│ ────── │                                      │
│ Wiki   │  Entity pages / Memory list /        │
│ Mems   │  Graph / Health — depending on       │
│ Graph  │  sidebar selection                   │
│ Health │                                      │
│        │                                      │
│ ────── │                                      │
│ User   │                                      │
│ 55/100 │                                      │
└────────┴──────────────────────────────────────┘
```

Files to modify:
- `src/cems/static/wiki/index.html` — restructure layout
- `src/cems/static/wiki/style.css` — sidebar styles
- `src/cems/static/wiki/app.js` — navigation logic, absorb memory list

#### 2. Absorb Memory List

Port the flat memory list from `static/dashboard/app.js` into the unified app:
- Memory list with pagination, search, category filters
- Edit modal (content, category, tags, source_ref)
- Scope toggle (All / Personal / Team)
- All using existing `/api/memory/list` and `/api/memory/update` endpoints

Files to read:
- `src/cems/static/dashboard/app.js` — memory list logic to port
- `src/cems/static/dashboard/style.css` — memory card styles to port

#### 3. Route Management

```javascript
// Hash-based routing
/#wiki              → Entity page list (landing)
/#wiki/:entity-id   → Entity page article
/#memories          → All memories flat list
/#graph             → Knowledge graph
/#health            → Lint, conflicts, health stats
```

#### 4. Redirect Old Dashboard

```python
# src/cems/server.py — redirect /dashboard to unified app
from starlette.responses import RedirectResponse

async def dashboard_redirect(request):
    return RedirectResponse("/wiki/#memories")

routes.append(Route("/dashboard", dashboard_redirect))
# Keep old dashboard files for one release cycle, then remove
```

## Acceptance Criteria

### Functional Requirements

- [ ] Single app at `/wiki` with sidebar navigation
- [ ] Wiki view: entity pages with Wikipedia layout (existing)
- [ ] Memories view: flat list with search, edit, categories, pagination
- [ ] Graph view: D3 force-directed graph (existing)
- [ ] Health view: stats, lint report, conflicts (existing)
- [ ] Sidebar shows: Wiki, Memories, Graph, Health, user info + health score
- [ ] URL hash routing works for all views (shareable links)
- [ ] `/dashboard` redirects to `/wiki/#memories`
- [ ] All existing API endpoints unchanged
- [ ] Login with API key works as before

### Non-Functional Requirements

- [ ] No new dependencies (vanilla JS, existing CSS variables)
- [ ] Page loads under 1 second
- [ ] Memory list handles 3,700+ memories with pagination

### Quality Gates

- [ ] All 711+ tests pass
- [ ] Existing hooks and MCP tools unaffected
- [ ] Old dashboard functionality preserved (search, edit, category filter, scope toggle)

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/cems/static/wiki/index.html` | Modify | Add sidebar, memories view, restructure layout |
| `src/cems/static/wiki/app.js` | Modify | Add memory list, port edit modal, sidebar nav |
| `src/cems/static/wiki/style.css` | Modify | Sidebar styles, memory card styles |
| `src/cems/server.py` | Modify | Redirect `/dashboard` → `/wiki/#memories` |

## References

- Brainstorm: `docs/brainstorms/2026-04-07-unified-wiki-dashboard-brainstorm.md`
- Current dashboard: `src/cems/static/dashboard/` (777 lines to absorb)
- Current wiki: `src/cems/static/wiki/` (1,304 lines — the base)
- Memory list API: `/api/memory/list` (GET, paginated)
- Memory edit API: `/api/memory/update` (POST)
