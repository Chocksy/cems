# CEMS Server Deployment Guide

This guide covers deploying CEMS for company-wide use with shared memory.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Infrastructure                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│   │  PostgreSQL  │  │    Qdrant    │  │   CEMS MCP Server        │  │
│   │  (metadata)  │  │  (vectors)   │  │   (port 8765)            │  │
│   └──────────────┘  └──────────────┘  └────────────┬─────────────┘  │
│          │                │                        │                 │
│          └────────────────┼────────────────────────┘                 │
│                           │                                          │
│   ┌───────────────────────┼───────────────────────────────────────┐ │
│   │               CEMS Background Worker                           │ │
│   │   • Nightly consolidation (3 AM)                              │ │
│   │   • Weekly summarization (Sunday)                             │ │
│   │   • Monthly re-indexing (1st of month)                        │ │
│   └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTPS/WSS
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Developer Machines                           │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   Developer 1    │   Developer 2    │   Developer N                │
│   Claude Code    │   Cursor IDE     │   VS Code                    │
│   + MCP Client   │   + MCP Client   │   + MCP Client               │
└──────────────────┴──────────────────┴──────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- A server with at least 4GB RAM, 20GB disk
- LLM API key (OpenAI, Anthropic, or OpenRouter)
- Domain name (optional, for HTTPS)

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | `secure_password_123` |
| `CEMS_API_KEY` | Server API key for authentication | `cems_your_api_key_here` |
| `CEMS_COMPANY_ID` | Your company identifier | `acme-corp` |

### LLM Provider Configuration

Choose **one** of the following provider configurations:

#### Option 1: OpenAI (Default)

```bash
CEMS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
CEMS_LLM_MODEL=gpt-4o-mini
```

#### Option 2: Anthropic

```bash
CEMS_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-key
CEMS_LLM_MODEL=claude-3-haiku-20240307
```

#### Option 3: OpenRouter (Recommended for Enterprise)

```bash
CEMS_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-key
CEMS_LLM_MODEL=anthropic/claude-3-haiku

# Optional: Attribution for OpenRouter dashboard
CEMS_OPENROUTER_SITE_URL=https://your-company.com
CEMS_OPENROUTER_SITE_NAME=YourCompany CEMS
```

**Why OpenRouter for Enterprise?**
- Centralized billing across all LLM providers
- Easy model switching without code changes
- Enterprise SSO (SAML) support
- Team management with role-based permissions
- BYOK (Bring Your Own Key) - use existing provider keys
- Automatic failover/fallbacks
- No platform-level rate limits

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CEMS_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for vector embeddings |
| `CEMS_STALE_DAYS` | `90` | Days before memory considered stale |
| `CEMS_ARCHIVE_DAYS` | `180` | Days before memory archived |
| `CEMS_NIGHTLY_HOUR` | `3` | Hour for nightly consolidation (0-23) |
| `CEMS_WEEKLY_DAY` | `sun` | Day for weekly summarization |
| `CEMS_WEEKLY_HOUR` | `4` | Hour for weekly summarization |
| `CEMS_MONTHLY_DAY` | `1` | Day of month for monthly reindex |
| `CEMS_MONTHLY_HOUR` | `5` | Hour for monthly reindex |

## Quick Start (Docker Compose)

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-memory.git
cd llm-memory

# Create environment file
cp deploy/.env.example .env
```

### 2. Edit Environment Variables

```bash
# .env file
POSTGRES_PASSWORD=your_secure_password_here
CEMS_API_KEY=cems_your_api_key_here
CEMS_COMPANY_ID=your-company

# Choose your LLM provider (see options above)
CEMS_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-key
CEMS_LLM_MODEL=anthropic/claude-3-haiku
```

### 3. Start Services

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cems-server
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8765/health

# Should return: {"status": "healthy"}
```

## Local Development Setup

For local development without Docker:

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### 2. Set Environment Variables

```bash
# Create a local .env file
cat > .env << 'EOF'
CEMS_USER_ID=developer
CEMS_STORAGE_DIR=~/.cems

# Choose your provider
CEMS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key

# Or for OpenRouter
# CEMS_LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=your-key
# CEMS_LLM_MODEL=anthropic/claude-3-haiku
EOF
```

### 3. Run the Server

```bash
# Start the MCP server (stdio mode for local use)
python -m cems.server

# Or run in HTTP mode for remote clients
CEMS_MODE=http python -m cems.server
```

### 4. Configure Your IDE

For Claude Code (stdio mode), add to `~/.claude.json` in the `mcpServers` section:

```json
{
  "mcpServers": {
    "cems": {
      "command": "python",
      "args": ["-m", "cems.server"],
      "env": {
        "CEMS_USER_ID": "your-username",
        "OPENAI_API_KEY": "sk-your-key",
        "OPENROUTER_API_KEY": "your-openrouter-key"
      }
    }
  }
}
```

For HTTP mode (Docker/server deployment), use:

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {
        "X-API-Key": "your-cems-api-key",
        "X-User-ID": "your-username",
        "X-Team-ID": "your-team"
      }
    }
  }
}
```

The `X-API-Key` must match the `CEMS_API_KEY` environment variable on the server.

For Cursor, add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "cems": {
      "command": "python",
      "args": ["-m", "cems.server"],
      "env": {
        "CEMS_USER_ID": "your-username",
        "OPENAI_API_KEY": "sk-your-key"
      }
    }
  }
}
```

## Production Deployment

### With Kubernetes

```yaml
# k8s/cems-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cems-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cems
  template:
    metadata:
      labels:
        app: cems
    spec:
      containers:
      - name: cems
        image: your-registry/cems:latest
        ports:
        - containerPort: 8765
        env:
        - name: CEMS_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cems-secrets
              key: database-url
        - name: CEMS_LLM_PROVIDER
          value: "openrouter"
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: cems-secrets
              key: openrouter-api-key
        - name: CEMS_LLM_MODEL
          value: "anthropic/claude-3-haiku"
```

### With Coolify

If you use Coolify for deployments:

```bash
# Add context
coolify context add cems --url https://your-coolify.com --token YOUR_TOKEN

# Deploy
coolify app deploy cems
```

### SSL/TLS Configuration

For production, put CEMS behind a reverse proxy (nginx, Traefik):

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name cems.yourcompany.com;

    ssl_certificate /etc/ssl/certs/cems.crt;
    ssl_certificate_key /etc/ssl/private/cems.key;

    location / {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Team Setup

### 1. Create Team

```bash
# Using the CLI (run on server)
docker exec -it cems-server cems-admin create-team \
    --name "engineering" \
    --company "your-company"
```

### 2. Generate API Keys for Users

```bash
# Generate key for each developer
docker exec -it cems-server cems-admin create-user \
    --username "jane.doe" \
    --email "jane@company.com" \
    --team "engineering"

# Output:
# User created!
# Username: jane.doe
# API Key: cems_usr_abc123...
# Team: engineering
```

### 3. Index Company Repositories

```bash
# Index your main backend repo
docker exec -it cems-server cems index git \
    https://github.com/company/backend.git \
    --team engineering \
    --scope shared

# Index frontend repo
docker exec -it cems-server cems index git \
    https://github.com/company/frontend.git \
    --team engineering \
    --scope shared
```

## Backup and Restore

### Backup

```bash
# Backup PostgreSQL
docker exec cems-postgres pg_dump -U cems cems > backup.sql

# Backup Qdrant
docker exec cems-qdrant tar -czf /qdrant/storage/backup.tar.gz /qdrant/storage/

# Copy backup
docker cp cems-qdrant:/qdrant/storage/backup.tar.gz ./qdrant-backup.tar.gz
```

### Restore

```bash
# Restore PostgreSQL
cat backup.sql | docker exec -i cems-postgres psql -U cems cems

# Restore Qdrant
docker cp qdrant-backup.tar.gz cems-qdrant:/qdrant/storage/
docker exec cems-qdrant tar -xzf /qdrant/storage/backup.tar.gz -C /
```

## Monitoring

### Health Endpoints

- `GET /health` - Basic health check
- `GET /metrics` - Prometheus metrics (if enabled)

### Logging

```bash
# View all logs
docker-compose logs -f

# Just the server
docker-compose logs -f cems-server

# Worker logs
docker-compose logs -f cems-worker
```

### Alerts (with Prometheus/Alertmanager)

```yaml
# prometheus/alerts.yml
groups:
  - name: cems
    rules:
      - alert: CEMSServerDown
        expr: up{job="cems"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CEMS server is down"

      - alert: CEMSHighMemoryUsage
        expr: container_memory_usage_bytes{name="cems-server"} > 3e9
        for: 5m
        labels:
          severity: warning
```

## Security Considerations

1. **API Keys**: Rotate regularly, never commit to git
2. **Network**: Keep PostgreSQL and Qdrant internal (not exposed)
3. **HTTPS**: Always use TLS in production
4. **Audit Log**: Enable for compliance (`CEMS_ENABLE_AUDIT=true`)
5. **Backups**: Automate daily backups
6. **LLM Keys**: Use OpenRouter for centralized key management in enterprise

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  cems-server:
    deploy:
      replicas: 3
```

### Qdrant Cluster

For high availability, run Qdrant in cluster mode:

```yaml
# qdrant-cluster.yml
services:
  qdrant-1:
    image: qdrant/qdrant
    environment:
      QDRANT__CLUSTER__ENABLED: true
      QDRANT__CLUSTER__P2P__PORT: 6335
    # ... additional config
```

## Troubleshooting

### Server won't start

```bash
# Check logs
docker-compose logs cems-server

# Common issues:
# - Missing LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY)
# - PostgreSQL not ready (wait for health check)
# - Port 8765 already in use
```

### LLM errors

```bash
# Check which provider is configured
docker exec cems-server printenv | grep CEMS_LLM

# Common issues:
# - Wrong API key format
# - Invalid model name (for OpenRouter, use provider/model format)
# - Rate limiting (check provider dashboard)
```

### Memory errors

```bash
# Increase memory limits
docker-compose up -d --scale cems-server=1 \
    -e CEMS_MAX_MEMORY=4g
```

### Slow queries

```bash
# Check Qdrant status
curl http://localhost:6333/collections

# Rebuild indexes
docker exec cems-server cems maintenance run reindex
```

## LLM Provider Migration

### Switching from OpenAI to OpenRouter

1. Get an OpenRouter API key from https://openrouter.ai
2. Update environment variables:
   ```bash
   CEMS_LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=your-new-key
   CEMS_LLM_MODEL=openai/gpt-4o-mini  # or anthropic/claude-3-haiku
   ```
3. Restart the server
4. No code changes needed - existing memories remain compatible

### Using OpenRouter with Existing Provider Keys (BYOK)

OpenRouter supports BYOK (Bring Your Own Key) for direct provider access with centralized billing:

1. Log into OpenRouter dashboard
2. Go to Settings → Keys
3. Add your existing OpenAI or Anthropic keys
4. Requests will use your keys directly while being billed through OpenRouter
