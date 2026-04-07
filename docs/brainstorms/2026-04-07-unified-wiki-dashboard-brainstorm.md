---
name: Unified Wiki Dashboard
description: Merge wiki + memories into one Notion-like company knowledge portal with Google OAuth
type: brainstorm
date: 2026-04-07
---

# Unified Wiki Dashboard

## What We're Building

Transform CEMS from two disconnected dashboards (/wiki + /dashboard) into a single
Notion-like knowledge portal that serves as the company brain.

## Two Deployment Modes

### Personal Mode (cems.chocksy.com)
- No team_id — single user
- Wiki built from personal memories
- Personal knowledge base

### Company Mode (cems.ai.hbstf.co)
- team_id set as env var → all memories default to team scope
- Multiple users via Google OAuth
- Wiki compiles from ALL team members' memories
- Shared entity pages = company knowledge
- Everyone sees everything (full transparency)

## Why This Approach

Karpathy's insight: "LLMs don't get bored maintaining a wiki." In a company setting,
every developer's coding session feeds the observer → memories → entity pages → wiki.
New team members get instant context via the wiki. Knowledge compounds across the team.

The current split (two separate dashboards, personal scope, API key auth) doesn't serve
this vision. Unifying creates a product, not a tool.

## Key Decisions

### 1. Wiki-First Navigation
- Landing page: Entity pages (the compiled knowledge)
- Sidebar: Topic navigation (like Notion's page tree)
- "All Memories" tab: Raw memory list (current /dashboard functionality)
- Graph: Visualization tab showing knowledge connections
- Health: Lint, conflicts, gaps

### 2. Team Scope by Default
- If `CEMS_TEAM_ID` is set in env, ALL new memories use team scope
- Entity pages compile from team-scoped memories (all members)
- No privacy boundaries — company wiki is fully shared
- Personal mode (no team_id) works exactly as today

### 3. Google OAuth Authentication
- Google login button for team dashboard
- Match email to user record in `users` table
- Create user automatically if not found (auto-provisioning)
- API key auth stays as fallback (for CLI/MCP/hooks)
- OAuth only needed for the web dashboard

### 4. Notion-Like UX
- Clean sidebar with page tree (entity pages as "pages")
- Smooth navigation between pages
- Markdown-rendered articles (already using marked.js)
- Breadcrumbs for navigation context
- Search across all entity pages and memories

## Architecture

```
┌────────────────────────────────────────────────────┐
│  Unified Dashboard (single app at /wiki or /)       │
├────────┬───────────────────────────────────────────┤
│        │                                           │
│ Sidebar│  Main Content Area                        │
│        │                                           │
│ Wiki   │  [Entity Page / Memory List / Graph /     │
│ Topics │   Health - depending on selection]         │
│        │                                           │
│ ─────  │  Entity pages: full wiki articles         │
│ All    │  Memories: flat list with search/edit      │
│ Mems   │  Graph: D3 force-directed visualization   │
│        │  Health: lint report, conflicts            │
│ Graph  │                                           │
│ Health │                                           │
│        │                                           │
│ ─────  │                                           │
│ User   │                                           │
│ avatar │                                           │
└────────┴───────────────────────────────────────────┘
```

### Navigation Structure
```
/                    → Wiki home (entity page list or featured page)
/wiki/:entity-id     → Entity page article
/memories            → All memories flat list (current /dashboard)
/graph               → Knowledge graph visualization
/health              → Lint, conflicts, orphans
/settings            → User profile, API key, preferences
```

## How the Wiki Expands with Team

1. **Developer A** works on a feature → observer captures session → memory stored with team scope
2. **Developer B** works on related code → their sessions also get captured
3. **Scheduler** (every 10 min): RelationBuilderJob links A's and B's memories together
4. **Scheduler** (every 10 min): CompilationJob finds the cluster, generates entity page
5. **Developer C** (new hire) opens the wiki → sees compiled knowledge from A + B
6. **Claude Code** (any developer): hook injects entity index → Claude uses /recall to read pages

## What Needs to Change (from current state)

### Dashboard
- [ ] Merge /wiki and /dashboard into single app
- [ ] Sidebar navigation (wiki topics, all memories, graph, health)
- [ ] Notion-like page layout with proper routing
- [ ] User avatar and profile section
- [ ] Search across entity pages AND memories

### Authentication
- [ ] Google OAuth flow (login button, callback, session)
- [ ] Auto-create user from Google email
- [ ] Session management (cookie/token based)
- [ ] API key auth stays for CLI/hooks (parallel auth)

### Team Scope
- [ ] Default to team scope when CEMS_TEAM_ID is set
- [ ] Entity pages compile from team-scoped memories
- [ ] Wiki endpoint filters by team (not just user)
- [ ] "All Memories" shows team memories (all members)

### Entity Pages
- [ ] Better entity page quality (current ones need work)
- [ ] Project-grouped entity navigation in sidebar
- [ ] Entity page editing (manual edits preserved on recompile?)

## Open Questions

1. **Entity page editing** — Should team members be able to manually edit wiki pages?
   If yes, how do we preserve edits when the CompilationJob recompiles?
2. **Onboarding** — What does a new team member see when they first open the wiki?
   Empty until the scheduler runs? Or do we show a "building knowledge..." state?
3. **Notifications** — Should team members be notified when new entity pages are created
   or when conflicts are detected?
4. **Mobile** — Is mobile access important for the team wiki?

## Next Steps

This is a significant product evolution. Recommend breaking into phases:

**Phase 1**: Merge dashboards into single app (sidebar + content area)
**Phase 2**: Team scope (default to team when team_id set)
**Phase 3**: Google OAuth
**Phase 4**: Polish (search, notifications, mobile)
