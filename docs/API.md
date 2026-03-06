# CEMS API Reference

Base URL: `http://localhost:8765` (or your deployed server address)

## Authentication

All API endpoints require a Bearer token:

```bash
curl -H "Authorization: Bearer $CEMS_API_KEY" http://localhost:8765/api/memory/status
```

Admin endpoints use the `CEMS_ADMIN_KEY` instead.

## Memory API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/add` | Store a memory |
| POST | `/api/memory/search` | Search memories |
| POST | `/api/memory/forget` | Soft-delete a memory |
| POST | `/api/memory/update` | Update memory content |
| POST | `/api/memory/restore` | Restore a soft-deleted memory |
| POST | `/api/memory/maintenance` | Run maintenance job |
| POST | `/api/memory/log-shown` | Log shown memories (feedback) |
| GET | `/api/memory/get?id=X` | Get full document |
| GET | `/api/memory/list` | List memories |
| GET | `/api/memory/status` | Stats + health |
| GET | `/api/memory/profile` | Profile context (session start) |
| GET | `/api/memory/foundation` | Foundation guidelines |
| GET | `/api/memory/gate-rules` | Gate rules by project |

## Session & Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/summarize` | Summarize a coding session |
| POST | `/api/tool/learning` | Submit tool learning |
| POST | `/api/index/repo` | Index git repository |

## Admin API

Requires `Authorization: Bearer <CEMS_ADMIN_KEY>`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/users` | Create user (returns API key) |
| GET | `/admin/users` | List users |
| GET | `/admin/users/{id}` | Get user details |
| PATCH | `/admin/users/{id}` | Update user |
| DELETE | `/admin/users/{id}` | Revoke user |
| POST | `/admin/users/{id}/reset-key` | Reset API key |
| POST | `/admin/teams` | Create team |
| GET | `/admin/teams` | List teams |
| GET | `/admin/db/stats` | Database stats |

## Examples

### Add a memory

```bash
curl -X POST http://localhost:8765/api/memory/add \
  -H "Authorization: Bearer $CEMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Always use port 8080 for development servers", "category": "config"}'
```

### Search memories

```bash
curl -X POST http://localhost:8765/api/memory/search \
  -H "Authorization: Bearer $CEMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "what port to use", "limit": 5}'
```

### Create a user (admin)

**CLI (recommended):**
```bash
cems admin --admin-key $CEMS_ADMIN_KEY users create alice
```

**curl:**
```bash
curl -X POST http://localhost:8765/admin/users \
  -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'
# Returns: {"api_key": "cems_ak_..."}
```

### Run maintenance

```bash
curl -X POST http://localhost:8765/api/memory/maintenance \
  -H "Authorization: Bearer $CEMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"job_type": "consolidation"}'
```

## Health Check

```bash
curl http://localhost:8765/health
# Returns: {"status": "ok", "version": "0.7.8", ...}
```

No authentication required.
