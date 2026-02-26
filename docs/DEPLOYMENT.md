# CEMS Server Deployment Guide

Deploy CEMS for individual or team use with persistent memory across coding sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                          │
│  ┌─────────────┐   ┌─────────────────────────────────┐  │
│  │  PostgreSQL  │   │  CEMS Python Server (port 8765) │  │
│  │  + pgvector  │◄──│  • REST API (Starlette)         │  │
│  │  (vectors +  │   │  • APScheduler (in-process)     │  │
│  │   metadata)  │   │  • Embeddings via OpenRouter     │  │
│  └─────────────┘   └──────────────┬──────────────────┘  │
│                                    │                     │
│                    ┌───────────────┘                     │
│                    ▼                                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MCP Wrapper (port 8766) — optional              │   │
│  │  Express.js, StreamableHTTP, stateless           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Developer Machines                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Claude   │  │  Cursor  │  │  Codex   │              │
│  │  Code     │  │  IDE     │  │  CLI     │              │
│  │  + Hooks  │  │          │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Key points:**
- Single PostgreSQL with pgvector handles vectors AND metadata (no Qdrant/Redis)
- Maintenance scheduler runs in-process via APScheduler (no separate worker)
- MCP wrapper is optional (Claude Code hooks talk directly to REST API)
- Embeddings via OpenRouter `text-embedding-3-small` (1536-dim)

## Prerequisites

- Docker and Docker Compose
- 2GB+ RAM, 10GB disk
- OpenRouter API key

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/cems.git
cd cems

# Create environment file
cp deploy/.env.example .env
```

### 2. Set Environment Variables

```bash
# .env — required
POSTGRES_PASSWORD=your_secure_password
OPENROUTER_API_KEY=sk-or-your-key
CEMS_ADMIN_KEY=cems_admin_your_key_here
```

### 3. Start Services

```bash
docker compose up -d postgres cems-server

# Check health
curl http://localhost:8765/health
# → {"status": "healthy", ...}
```

### 4. Create a User

```bash
curl -X POST http://localhost:8765/admin/users \
  -H "Authorization: Bearer $CEMS_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "yourname"}'
# → {"api_key": "YOUR_KEY_HERE..."}
```

### 5. Install Client Hooks

```bash
# Install CEMS CLI + hooks on each developer machine
pip install cems
cems setup --claude --api-url https://your-server:8765 --api-key YOUR_KEY_HERE
```

Or for non-interactive install:

```bash
export CEMS_API_KEY=YOUR_KEY_HERE
curl -sSf https://your-server/install.sh | bash
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | `sk-or-your-key` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `secure_password_123` |
| `CEMS_ADMIN_KEY` | Admin API key for user/team management | `cems_admin_xxx` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_EMBEDDING_BACKEND` | `openrouter` | Embedding provider |
| `CEMS_EMBEDDING_DIMENSION` | `1536` | Embedding vector dimension |
| `CEMS_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CEMS_RERANKER_BACKEND` | `disabled` | Reranker (disabled recommended) |
| `CEMS_NIGHTLY_HOUR` | `3` | Hour for nightly consolidation |
| `CEMS_WEEKLY_DAY` | `sun` | Day for weekly summarization |
| `CEMS_WEEKLY_HOUR` | `4` | Hour for weekly summarization |
| `CEMS_MONTHLY_DAY` | `1` | Day for monthly reindex |
| `CEMS_MONTHLY_HOUR` | `5` | Hour for monthly reindex |

## API Endpoints

### Memory (authenticated — `Authorization: Bearer cems_usr_xxx`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/memory/add` | Store a memory |
| POST | `/api/memory/search` | Search memories |
| POST | `/api/memory/forget` | Soft-delete a memory |
| POST | `/api/memory/update` | Update memory content |
| POST | `/api/memory/maintenance` | Run maintenance job |
| GET | `/api/memory/get?id=X` | Get full document |
| GET | `/api/memory/list` | List memories |
| GET | `/api/memory/status` | Memory stats |
| GET | `/api/memory/profile` | Profile context |
| GET | `/api/memory/foundation` | Foundation guidelines |
| GET | `/api/memory/gate-rules` | Gate rules |
| POST | `/api/memory/log-shown` | Log shown memories |

### Session & Tools

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/session/summarize` | Summarize a session |
| POST | `/api/tool/learning` | Submit tool learning |

### Admin (requires `CEMS_ADMIN_KEY`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/users` | Create user |
| GET | `/admin/users` | List users |
| POST | `/admin/teams` | Create team |
| GET | `/admin/db/stats` | Database statistics |

### Index

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/index/repo` | Index a git repo |
| POST | `/api/index/path` | Index a local path |

## Maintenance Schedule

The scheduler runs in-process (APScheduler). No separate worker needed.

| Job | Schedule | Description |
|-----|----------|-------------|
| Consolidation | Nightly 3 AM | Merge duplicate memories |
| Reflection | Nightly 3:30 AM | Consolidate overlapping observations |
| Summarization | Weekly Sun 4 AM | Compress old memories, prune stale |
| Re-indexing | Monthly 1st 5 AM | Rebuild embeddings, archive dead |

Run manually via API:

```bash
curl -X POST http://localhost:8765/api/memory/maintenance \
  -H "Authorization: Bearer $CEMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"job_type": "consolidation"}'
```

## Database Migrations

After initial setup, run migrations in order:

```bash
# 1. Base schema (auto-runs via init.sql in Docker)
# 2. Document+chunk model
docker exec -i cems-postgres psql -U cems cems < deploy/migrate_docs_schema.sql

# 3. Soft-delete + feedback columns
docker exec -i cems-postgres psql -U cems cems < scripts/migrate_soft_delete_feedback.sql
```

## Backup and Restore

```bash
# Backup PostgreSQL (includes vectors — no separate backup needed)
docker exec cems-postgres pg_dump -U cems cems > backup_$(date +%Y%m%d).sql

# Restore
cat backup.sql | docker exec -i cems-postgres psql -U cems cems
```

## Production Deployment

### With Coolify

```bash
coolify context use production
coolify app deploy cems
```

### SSL/TLS

Put CEMS behind a reverse proxy (Caddy, nginx, Traefik):

```nginx
server {
    listen 443 ssl;
    server_name cems.yourcompany.com;

    location / {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }
}
```

### Health Check

```bash
curl http://localhost:8765/health
# → {"status": "healthy", "database": "connected", ...}
```

## Troubleshooting

### Server won't start

```bash
docker compose logs cems-server --tail 50

# Common issues:
# - Missing OPENROUTER_API_KEY
# - PostgreSQL not ready (wait for healthcheck)
# - Port 8765 in use
```

### Rebuild after code changes

```bash
docker compose build cems-server
docker compose up -d cems-server
```

## Security

- Keep PostgreSQL internal (don't expose port 5432 externally)
- Use TLS in production
- Rotate API keys regularly
- Admin key should not be shared with developers
- User API keys are per-developer; revoke via `DELETE /admin/users/{id}`
