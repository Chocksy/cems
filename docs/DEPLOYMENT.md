# CEMS Server Deployment Guide

Deploy CEMS for your team. The server runs in Docker, developers install the CLI.

## Architecture

```
  Server (Docker / Kubernetes)             Developer Machines
  +-------------------------------+        +-------------------+
  |  PostgreSQL + pgvector        |        |  cems CLI         |
  |  CEMS Server (port 8765)      | <----- |  + IDE hooks      |
  |  - REST API (Starlette)       | HTTPS  |                   |
  |  - Scheduler (APScheduler)    |        |  Claude Code      |
  |  - Embeddings (OpenRouter)    |        |  Cursor / Codex   |
  +-------------------------------+        |  Goose            |
                                           +-------------------+
```

- Single PostgreSQL with pgvector (no Redis/Qdrant needed)
- Maintenance runs in-process (no separate worker)
- Embeddings via OpenRouter `text-embedding-3-small`

## Prerequisites

- Docker and Docker Compose
- [OpenRouter API key](https://openrouter.ai/keys)

---

## Server Setup (Docker Compose)

### Option A: Docker Hub image (recommended)

No git clone needed. Just create two files:

**`.env`**
```bash
POSTGRES_PASSWORD=your_secure_password
OPENROUTER_API_KEY=sk-or-your-key
CEMS_ADMIN_KEY=cems_admin_random_string_here
```

**`docker-compose.yml`** — download or copy from [`deploy/docker-compose.yml`](../deploy/docker-compose.yml):

```bash
curl -fsSLO https://raw.githubusercontent.com/chocksy/cems/main/deploy/docker-compose.yml
```

Start:

```bash
docker compose up -d
```

### Option B: Build from source

```bash
git clone https://github.com/chocksy/cems.git
cd cems
cp deploy/.env.example .env
# Edit .env with your credentials
docker compose up -d postgres cems-server
```

### Run migrations

After starting the server (either option), run these once:

**If you cloned the repo:**
```bash
docker exec -i cems-postgres psql -U cems cems < scripts/migrate_docs_schema.sql
docker exec -i cems-postgres psql -U cems cems < scripts/migrate_soft_delete_feedback.sql
docker exec -i cems-postgres psql -U cems cems < scripts/migrate_conflicts.sql
```

**If using Docker Hub image (no clone):**
```bash
for f in migrate_docs_schema.sql migrate_soft_delete_feedback.sql migrate_conflicts.sql; do
  curl -fsSL "https://raw.githubusercontent.com/chocksy/cems/main/scripts/$f" | \
    docker exec -i cems-postgres psql -U cems cems
done
```

### Create users

```bash
cems admin --admin-key $CEMS_ADMIN_KEY users create alice
# Returns the API key — save it, shown only once!
```

Or with email: `cems admin --admin-key $CEMS_ADMIN_KEY users create alice --email alice@example.com`

Give each developer their API key. That's it for the server.

---

## Client Setup (Developer Machines)

Each developer installs the CEMS CLI and connects to your server.

Replace `cems.example.com` below with your actual server address — either your domain or `localhost:8765` for local Docker.

### Option A: One-line install

```bash
curl -fsSL https://getcems.com/install.sh | bash
```

Prompts for server URL and API key, then asks which IDEs to configure.

### Option B: Non-interactive

```bash
CEMS_API_KEY=cems_ak_... CEMS_API_URL=https://cems.example.com \
  curl -fsSL https://getcems.com/install.sh | bash
```

### Option C: Manual

```bash
pip install cems
cems setup --api-url https://cems.example.com --api-key cems_ak_...
```

### Supported IDEs

| Flag | What it installs |
|------|-----------------|
| `--claude` | 6 hooks, 6 skills, 2 commands, settings.json config |
| `--cursor` | Rules and memory integration |
| `--codex` | Commands and skills |
| `--goose` | Extension config |

Run `cems setup` without flags for interactive IDE selection.

### CEMS CLI Commands

Once installed, developers have these commands:

```
cems search "Docker port binding"     # Search memories
cems add "Always use port 8080"       # Store a memory
cems list                             # List recent memories
cems delete <id>                      # Soft-delete a memory
cems status                           # Check connection + stats
cems health                           # Server health check
cems debug                            # Debug dashboard (see what hooks inject)
cems rule                             # Create gate rules
cems maintenance consolidation        # Trigger maintenance manually
cems update                           # Update CLI + re-deploy hooks
cems uninstall                        # Remove hooks from IDE
```

Credentials are stored in `~/.cems/credentials` and read automatically by the CLI and hooks.

### How hooks work

After `cems setup`, your IDE automatically:
- **On session start**: Loads your profile (preferences, guidelines, gate rules)
- **On each prompt**: Searches memory for relevant context, injects it
- **On tool use**: Applies gate rules (block/warn), extracts learnings
- **On session end**: Writes an observer signal for session summarization

No manual steps needed. Memories build up and improve over time.

---

## Updating

### Server

**Docker Hub:**
```bash
docker compose pull cems-server
docker compose up -d cems-server
```

**From source:**
```bash
cd cems && git pull
docker compose build cems-server
docker compose up -d cems-server
```

### Client

```bash
cems update
```

This runs `uv tool install cems --force` and re-deploys hooks.

---

## Kubernetes

Same concepts as Docker Compose, deployed as Kubernetes resources.

### 1. Image

Use the public Docker Hub image or build your own:

```bash
# Public image (no build needed)
docker pull chocksy/cems-server:latest

# Or build and push to your private registry
docker build -t your-registry.com/cems-server:latest .
docker push your-registry.com/cems-server:latest
```

### 2. Create namespace and secrets

```bash
kubectl create namespace cems
```

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cems-secrets
  namespace: cems
type: Opaque
stringData:
  POSTGRES_PASSWORD: "your_secure_password"
  OPENROUTER_API_KEY: "sk-or-your-key"
  CEMS_ADMIN_KEY: "cems_admin_random_string"
```

### 3. Deploy PostgreSQL

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: cems
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels: { app: postgres }
  template:
    metadata:
      labels: { app: postgres }
    spec:
      containers:
        - name: postgres
          image: pgvector/pgvector:pg16
          ports: [{ containerPort: 5432 }]
          env:
            - { name: POSTGRES_USER, value: cems }
            - { name: POSTGRES_DB, value: cems }
            - name: POSTGRES_PASSWORD
              valueFrom: { secretKeyRef: { name: cems-secrets, key: POSTGRES_PASSWORD } }
          volumeMounts:
            - { name: postgres-data, mountPath: /var/lib/postgresql/data }
          readinessProbe:
            exec: { command: ["pg_isready", "-U", "cems"] }
            periodSeconds: 10
  volumeClaimTemplates:
    - metadata: { name: postgres-data }
      spec:
        accessModes: ["ReadWriteOnce"]
        resources: { requests: { storage: 10Gi } }
---
apiVersion: v1
kind: Service
metadata: { name: postgres, namespace: cems }
spec:
  selector: { app: postgres }
  ports: [{ port: 5432 }]
  clusterIP: None
```

### 4. Deploy CEMS server

```yaml
# k8s/cems-server.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cems-server
  namespace: cems
spec:
  replicas: 1
  selector:
    matchLabels: { app: cems-server }
  template:
    metadata:
      labels: { app: cems-server }
    spec:
      containers:
        - name: cems-server
          image: chocksy/cems-server:latest  # or your-registry.com/cems-server:latest
          ports: [{ containerPort: 8765 }]
          env:
            - name: CEMS_DATABASE_URL
              value: "postgresql://cems:$(POSTGRES_PASSWORD)@postgres.cems.svc.cluster.local:5432/cems"
            - name: POSTGRES_PASSWORD
              valueFrom: { secretKeyRef: { name: cems-secrets, key: POSTGRES_PASSWORD } }
            - name: OPENROUTER_API_KEY
              valueFrom: { secretKeyRef: { name: cems-secrets, key: OPENROUTER_API_KEY } }
            - name: CEMS_ADMIN_KEY
              valueFrom: { secretKeyRef: { name: cems-secrets, key: CEMS_ADMIN_KEY } }
            - { name: CEMS_MODE, value: server }
            - { name: CEMS_SERVER_HOST, value: "0.0.0.0" }
            - { name: CEMS_SERVER_PORT, value: "8765" }
            - { name: CEMS_EMBEDDING_BACKEND, value: openrouter }
            - { name: CEMS_EMBEDDING_DIMENSION, value: "1536" }
            - { name: CEMS_RERANKER_BACKEND, value: disabled }
          readinessProbe:
            httpGet: { path: /health, port: 8765 }
            initialDelaySeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8765 }
            initialDelaySeconds: 30
          resources:
            requests: { memory: 512Mi, cpu: 250m }
            limits: { memory: 2Gi, cpu: "1" }
---
apiVersion: v1
kind: Service
metadata: { name: cems-server, namespace: cems }
spec:
  selector: { app: cems-server }
  ports: [{ port: 8765, targetPort: 8765 }]
```

### 5. Expose with Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cems-ingress
  namespace: cems
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts: [cems.example.com]
      secretName: cems-tls
  rules:
    - host: cems.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service: { name: cems-server, port: { number: 8765 } }
```

### 6. Apply and run migrations

```bash
kubectl apply -f k8s/

# Wait for ready
kubectl -n cems wait --for=condition=ready pod -l app=postgres --timeout=120s

# Run migrations
PG=$(kubectl -n cems get pod -l app=postgres -o jsonpath='{.items[0].metadata.name}')
for f in scripts/migrate_docs_schema.sql scripts/migrate_soft_delete_feedback.sql scripts/migrate_conflicts.sql; do
  kubectl -n cems cp $f $PG:/tmp/$(basename $f)
  kubectl -n cems exec $PG -- psql -U cems cems -f /tmp/$(basename $f)
done
```

Then create users the same way as Docker Compose (port-forward or use ingress URL).

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `CEMS_ADMIN_KEY` | Admin key for `/admin/*` endpoints |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_DATABASE_URL` | auto | PostgreSQL connection string |
| `CEMS_SERVER_PORT` | `8765` | Server port |
| `CEMS_EMBEDDING_BACKEND` | `openrouter` | Embedding provider |
| `CEMS_EMBEDDING_DIMENSION` | `1536` | Vector dimension |
| `CEMS_RERANKER_BACKEND` | `disabled` | Reranker (keep disabled) |
| `CEMS_NIGHTLY_HOUR` | `3` | Consolidation hour (UTC) |
| `CEMS_WEEKLY_DAY` | `sun` | Summarization day |
| `CEMS_STALE_DAYS` | `90` | Days before memory is stale |
| `CEMS_ARCHIVE_DAYS` | `180` | Days before memory is archived |

---

## API Endpoints

### Memory (requires `Authorization: Bearer <user_api_key>`)

| Method | Path | Description |
|--------|------|-------------|
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
| GET | `/api/memory/gate-rules` | Gate rules (pre-tool-use) |

### Session & Tools

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/session/summarize` | Summarize a coding session |
| POST | `/api/tool/learning` | Submit tool learning |

### Admin (requires `Authorization: Bearer <admin_key>`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/users` | Create user (returns API key) |
| GET | `/admin/users` | List users |
| DELETE | `/admin/users/{id}` | Revoke user |
| POST | `/admin/teams` | Create team |
| GET | `/admin/db/stats` | Database stats |

---

## Maintenance

Runs automatically via APScheduler. No cron or worker needed.

| Job | Schedule | Description |
|-----|----------|-------------|
| Consolidation | Nightly 3 AM | Merge duplicates, detect conflicts |
| Reflection | Nightly 3:30 AM | Consolidate observations |
| Summarization | Weekly Sun 4 AM | Compress old memories |
| Re-indexing | Monthly 1st 5 AM | Rebuild embeddings |

Trigger manually via CLI or API:

```bash
cems maintenance consolidation
# or
curl -X POST https://cems.example.com/api/memory/maintenance \
  -H "Authorization: Bearer $CEMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"job_type": "consolidation"}'
```

---

## Backup and Restore

```bash
# Backup
docker exec cems-postgres pg_dump -U cems cems > backup_$(date +%Y%m%d).sql

# Restore
cat backup.sql | docker exec -i cems-postgres psql -U cems cems
```

---

## Production Checklist

- [ ] Strong `POSTGRES_PASSWORD` and `CEMS_ADMIN_KEY` (32+ random chars)
- [ ] TLS via reverse proxy, Ingress, or Coolify
- [ ] PostgreSQL port (5432) not exposed externally
- [ ] Automated backups (pg_dump cron)
- [ ] Per-developer API keys (revoke with `DELETE /admin/users/{id}`)
- [ ] Health monitoring on `GET /health`

---

## Troubleshooting

**Server won't start:**
```bash
docker compose logs cems-server --tail 50
# Common: missing OPENROUTER_API_KEY, postgres not ready, port in use
```

**Migrations failed:**
```bash
docker exec cems-postgres psql -U cems cems -c "\dt memory_*"
# Re-run — they are idempotent
```

**Search returns nothing:**
```bash
cems status   # Check document count
cems search "test"  # Verify connectivity
```
