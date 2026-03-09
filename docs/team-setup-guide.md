# Team Setup Guide

How to create teams and add members in CEMS.

## Prerequisites

- CEMS server running (Docker or standalone)
- Admin API key (`CEMS_ADMIN_KEY` environment variable)
- Server URL (e.g., `https://cems.yourcompany.com`)

Set these for the examples below:

```bash
CEMS_URL="https://cems.yourcompany.com"
ADMIN_KEY="your-admin-key"
```

## 1. Create a Team

```bash
curl -X POST "$CEMS_URL/admin/teams" \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "engineering", "company_id": "acme"}'
```

Response:
```json
{
  "team": {"id": "uuid-here", "name": "engineering", "company_id": "acme"},
  "message": "Team created"
}
```

Save the team `id` — you'll need it to add members.

## 2. Create Users

```bash
# Create a user
curl -X POST "$CEMS_URL/admin/users" \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@acme.com"}'
```

Response:
```json
{
  "user": {"id": "user-uuid", "username": "alice"},
  "api_key": "cems_abc123...",
  "message": "User created. Save the API key - it will not be shown again."
}
```

**Important:** Save the `api_key` — it's shown only once. Give it to the team member.

## 3. Add Members to Team

Use team name or ID, user name or ID:

```bash
curl -X POST "$CEMS_URL/admin/teams/engineering/members" \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "role": "member"}'
```

Roles: `member` (default), `admin`.

## 4. User Setup

Each team member runs on their machine:

```bash
pip install cems
cems setup --claude --api-url "$CEMS_URL" --api-key "their-api-key"
```

This will:
- Save credentials to `~/.cems/credentials`
- Auto-detect the user's team and store `CEMS_TEAM_ID`
- Register the MCP server with team headers
- Install hooks and skills

Then add to their shell profile:

```bash
echo 'eval "$(cems env)"' >> ~/.zshrc
source ~/.zshrc
```

## 5. Verify

```bash
cems health
```

## Quick Reference

| Action | Command |
|--------|---------|
| List teams | `curl "$CEMS_URL/admin/teams" -H "Authorization: Bearer $ADMIN_KEY"` |
| List users | `curl "$CEMS_URL/admin/users" -H "Authorization: Bearer $ADMIN_KEY"` |
| View team details | `curl "$CEMS_URL/admin/teams/engineering" -H "Authorization: Bearer $ADMIN_KEY"` |
| Remove member | `curl -X DELETE "$CEMS_URL/admin/teams/engineering/members/alice" -H "Authorization: Bearer $ADMIN_KEY"` |
| Reset user API key | `curl -X POST "$CEMS_URL/admin/users/{user_id}/reset-key" -H "Authorization: Bearer $ADMIN_KEY"` |
| Delete team | `curl -X DELETE "$CEMS_URL/admin/teams/engineering" -H "Authorization: Bearer $ADMIN_KEY"` |

## How Memory Scoping Works

- **Personal memories** — visible only to the user who created them
- **Shared memories** — visible to all members of the user's team
- **Promote** — users can promote personal memories to shared via the dashboard or MCP `memory_promote` tool
- **Auto-resolve** — if a user belongs to exactly 1 team, the server auto-sets their team context (no need for manual `X-Team-Id` header)

## Dashboard

Users can browse and manage memories at:

```
https://cems.yourcompany.com/dashboard/
```

Log in with their API key. The scope toggle (All / Personal / Team) lets them switch between views. The "Promote to Team" button on personal memories shares them with the team.
