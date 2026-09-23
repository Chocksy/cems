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
- [OpenRouter API key](https://openrouter.ai/keys), or local models instead (see [Private mode](#private-mode))

---

## Server Setup (Docker Compose)

### Option A: Docker Hub image (recommended)

No git clone needed. Just create two files:

**`.env`**
```bash
POSTGRES_PASSWORD=your_secure_password
OPENROUTER_API_KEY=sk-or-your-key   # or run local models, see Private mode
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

### Migrations

Nothing to run. The server applies every schema migration at start, sizing the
embedding column from `CEMS_EMBEDDING_DIMENSION`. The `scripts/*.sql` files are
legacy references that hard-code 1536 dimensions. Do not apply them by hand.

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
cems status                           # Server health + system status
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

## Private mode

Private mode runs the whole memory pipeline on models you control. Memories always live in your PostgreSQL. Private mode moves the model calls too.

### What leaves your network

| | Default (OpenRouter) | Private cloud (Bedrock, Azure, Vertex via a gateway) | Private mode (Ollama) |
|---|---|---|---|
| Memories, embeddings at rest | Your Postgres | Your Postgres | Your Postgres |
| Extraction, consolidation, lint | OpenRouter, then the model vendor | Your cloud account | This box |
| Embedding calls | OpenRouter (OpenAI model) | Your cloud account | This box |
| Recall-time query synthesis, agentic search | OpenRouter | Your cloud account | This box (GPU preset) or off (CPU preset) |
| Your coding agent (Claude Code, Cursor, Codex) | Its vendor | Its vendor, or your cloud if the agent supports it | Its vendor, or a local model if the agent supports it |

CEMS does not change what your coding agent sends to its vendor. See [Running your agent privately](CLIENT.md#running-your-agent-privately).

### Three commands

```bash
curl -fsSL https://getcems.com/install-server.sh -o install-server.sh
bash install-server.sh --private --yes        # or --private-gpu
cems admin --admin-key <printed key> users create alice
```

The first boot pulls `gemma4:e4b` (9.6 GB on disk) and `embeddinggemma` (621 MB), about 10 GB in total. The health wait covers that.

Installer flags:

| Flag | Effect |
|------|--------|
| `--private` | CPU preset (`deploy/.env.private-cpu.example`) |
| `--private-gpu` | GPU preset plus the NVIDIA compose override |
| `--openrouter-key <key>` | Key for default mode. Also read from `OPENROUTER_API_KEY` |
| `--dir <path>` | Install directory, default `/opt/cems` |
| `--yes` | Skip the confirmation prompt |
| `--dry-run` | Print the commands, change nothing |

The installer generates `POSTGRES_PASSWORD` and `CEMS_ADMIN_KEY`, writes them to `<dir>/deploy/.env` with mode 600, and prints the admin key once. An existing `.env` is kept as is.

Without the installer, the same thing by hand:

```bash
cp deploy/.env.private-cpu.example .env
docker compose --profile private up -d
# GPU preset:
cp deploy/.env.private-gpu.example .env
docker compose --profile private -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The GPU override needs the NVIDIA Container Toolkit on the host.

### Cloud-init (boot a ready server)

Paste one of these into the user-data field when creating the VM. The box installs Docker, starts CEMS in private mode, and is ready on port 8765.

- [AWS EC2](../deploy/cloud-init/aws.yaml)
- [Hetzner Cloud](../deploy/cloud-init/hetzner.yaml)
- [DigitalOcean](../deploy/cloud-init/digitalocean.yaml)

Read the admin key afterwards from `/opt/cems/deploy/.env`.

### Network exposure

The cloud-init snippets open port 8765: Hetzner and DigitalOcean run `ufw allow 8765/tcp`, AWS expects you to open it in the instance security group. CEMS authenticates every request with the admin key or a user API key, but it serves plain HTTP. Put a TLS reverse proxy (Caddy, or your load balancer) in front of it for anything beyond a private network or Tailscale.

### Hardware and cost

| Tier | Example box | What runs | Approx. monthly |
|---|---|---|---|
| CPU | Hetzner CX33 (4 vCPU, 8 GB) or CX43 (8 vCPU, 16 GB) | Extraction, consolidation, embeddings. Plain hybrid recall | EUR 8.49 or EUR 15.99 [^prices] |
| GPU, single card | AWS g6.xlarge (1 NVIDIA L4, 24 GB VRAM) | Everything on, `qwen3.8:27b`, 32K context by default. Not yet verified on hardware | USD 0.8048 per hour, about USD 588 [^prices] |
| GPU, large | Hetzner GEX131 (RTX PRO 6000, 96 GB) or AWS p4d.24xlarge | Everything on, 1M-context model. Not yet verified on hardware | EUR 1,199 plus EUR 599 setup, or USD 21.96 per hour [^prices] |

The GPU preset caps the context at 32K (`OLLAMA_CONTEXT_LENGTH=32768`). `qwen3.8:27b` is about 18 GB of weights, so a 24 GB card has room for that and not much more. Raise the cap (up to 262144) only on a larger card. A 20 GB card (Hetzner GEX44, RTX 4000 SFF Ada) may run the preset at 32K or with a smaller model. Only the CPU tier has been tested on real hardware so far.

### Context length

Ollama's default context is 4096 tokens. It silently truncates longer prompts and logs `truncating input prompt limit=... prompt=...`. CEMS extraction prompts run past 4096, so a truncated prompt means lost memories with no error.

The compose file passes `OLLAMA_CONTEXT_LENGTH` to the `ollama` service. The default is 8192; both presets set 32768. A bigger context uses more RAM (or VRAM), so size it to the box. If you see the truncation line in `docker compose logs ollama`, raise the value and run `docker compose --profile private up -d` again.

### Verified

Tested on 2026-09-23 on a Hetzner CX33 (4 vCPU, 8 GB RAM, Ubuntu 24.04) with the CPU preset:

- Installer start to healthy: about 3 minutes with Docker already installed, model pulls included.
- RAM in use with everything running at a 32K context: about 6.2 GB of 7.7 GB.
- Add and search round trip passed. Admin health reported LLM ok, embeddings ok, dimension 768.
- With outbound ports 80 and 443 blocked after the first boot, add and search still worked.
- `gemma4:e4b` processes prompts at about 20 tokens per second on this CPU. That is fine for CEMS background jobs.

[^prices]: List prices checked 2026-09-22: [Hetzner Cloud](https://www.hetzner.com/cloud/), [Hetzner GEX131](https://www.hetzner.com/dedicated-rootserver/gex131/), [AWS EC2 on-demand](https://aws.amazon.com/ec2/pricing/on-demand/) (g6.xlarge and p4d.24xlarge, us-east-1). Hetzner raised cloud prices on 15 June 2026, so check before you budget.

### Limits

- The embedding dimension is fixed when the database is first created. Switching from OpenRouter (1536) to Ollama (768) needs a fresh database; the server refuses to start otherwise with `Embedding dimension mismatch`.
- Ollama downloads models from the internet on first boot. After that the box needs no outbound access for CEMS to work.
- The CPU preset turns off query synthesis, preference synthesis, forced synthesis (`CEMS_ENABLE_FORCED_SYNTHESIS`, the LLM expansion that temporal, preference and aggregation queries otherwise always get), query decomposition and agentic search. Set the `CEMS_ENABLE_*` variables to `true` to turn them back on if the box can take it.
- A CPU box serves CEMS fine but is too slow to also run your coding agent's model. On the CX33 an OpenCode turn with a 27K-token prompt took over 20 minutes. Use a GPU box or a hosted model for the agent.

### Bring your own endpoint

Any OpenAI-compatible server works without the Ollama profile. Set `CEMS_LLM_BASE_URL`, `CEMS_LLM_API_KEY`, `CEMS_LLM_MODEL`, `CEMS_EMBEDDING_MODEL` and `CEMS_EMBEDDING_DIMENSION` in `.env` and run `docker compose up -d`. For Bedrock or Azure OpenAI put a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) in front and point CEMS at it.

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
          # Some libraries (tiktoken, matplotlib, huggingface) write to the
          # directory returned by tempfile.gettempdir(). With
          # readOnlyRootFilesystem: true they fail unless /tmp is writable.
          # The image bakes TIKTOKEN_CACHE_DIR=/home/cems/.cache/tiktoken, so
          # tiktoken is covered, but keep this emptyDir for everything else.
          volumeMounts:
            - { name: tmp, mountPath: /tmp }
          readinessProbe:
            httpGet: { path: /health, port: 8765 }
            initialDelaySeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8765 }
            initialDelaySeconds: 30
          resources:
            requests: { memory: 512Mi, cpu: 250m }
            limits: { memory: 2Gi, cpu: "1" }
      volumes:
        - { name: tmp, emptyDir: {} }
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

### 6. Apply

```bash
kubectl apply -f k8s/

# Wait for ready
kubectl -n cems wait --for=condition=ready pod -l app=postgres --timeout=120s
```

The cems-server pod runs schema migrations itself at start. No psql step.

Then create users the same way as Docker Compose (port-forward or use ingress URL).

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `CEMS_ADMIN_KEY` | Admin key for `/admin/*` endpoints |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_DATABASE_URL` | auto | PostgreSQL connection string |
| `CEMS_SERVER_PORT` | `8765` | Server port |
| `CEMS_LLM_MODEL` | `qwen/qwen3-32b` | LLM for maintenance jobs (lint, consolidation, compilation). Any [OpenRouter model](https://openrouter.ai/models) |
| `CEMS_AGENTIC_MODEL` | `google/gemini-2.5-flash-lite` | LLM for agentic search agents. Needs 1M+ context — agents receive the full memory dump |
| `OPENROUTER_API_KEY` | none | Required only when using OpenRouter (the default). Fallback for `CEMS_LLM_API_KEY` |
| `CEMS_LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat endpoint (Ollama, vLLM, LiteLLM, Azure OpenAI) |
| `CEMS_LLM_API_KEY` | `OPENROUTER_API_KEY` | Key for the LLM endpoint. Ollama ignores it but needs a value |
| `CEMS_EMBEDDING_BASE_URL` | same as LLM | Separate embeddings endpoint if needed |
| `CEMS_EMBEDDING_API_KEY` | same as LLM | Key for the embeddings endpoint |
| `CEMS_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `CEMS_EMBEDDING_DIMENSION` | `1536` | Must match the model. Fixed at first boot, see [Private mode](#private-mode) |
| `CEMS_ENABLE_AGENTIC_SEARCH` | `true` | Allow `mode=agentic` search. Needs a long-context model |
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
