# Coolify Docker Compose Deployment Instructions

## Quick Steps

### 1. Create Docker Compose App in Coolify UI

1. Go to Coolify UI → Project "cems" (fsgsks8go4c4k0w8ko8844ok)
2. Click **"New"** → **"Docker Compose"**
3. Configure:
   - **GitHub Repository**: `chocksy/cems`
   - **Branch**: `main`
   - **Docker Compose File Path**: `docker-compose.coolify.yml`
   - **Name**: `cems-stack` (or any name you prefer)
4. Click **"Create"**

### 2. Set Environment Variables

After the app is created, set these environment variables (get UUID from app list):

```bash
# Replace NEW_APP_UUID with the UUID from the newly created app
# Replace placeholder values with your actual secrets (NEVER commit real secrets!)
coolify app env set NEW_APP_UUID CEMS_DATABASE_URL="postgres://cems:YOUR_DB_PASSWORD@YOUR_DB_HOST:5432/cems"
coolify app env set NEW_APP_UUID CEMS_QDRANT_URL="http://cems-qdrant:6333"
coolify app env set NEW_APP_UUID OPENROUTER_API_KEY="sk-or-YOUR-API-KEY"
coolify app env set NEW_APP_UUID CEMS_ADMIN_KEY="YOUR_ADMIN_KEY"
```

### 3. Configure Domains in Coolify UI

After deployment, configure domains in the Coolify UI:

1. Go to the app → **"Domains"** tab
2. Add domains:
   - `cems.chocksy.com` → Route to `cems-server` service (port 8765)
   - `mcp-cems.chocksy.com` → Route to `cems-mcp` service (port 8766)

**Note**: The Traefik labels in `docker-compose.coolify.yml` should handle routing automatically, but you may need to configure domains in Coolify UI as well.

### 4. Deploy

Click **"Deploy"** in the Coolify UI.

### 5. Verify

```bash
# Test Python API
curl https://cems.chocksy.com/health

# Test Express MCP wrapper
curl https://mcp-cems.chocksy.com/health

# Test admin API
curl https://cems.chocksy.com/admin \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

## Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `CEMS_DATABASE_URL` | PostgreSQL connection string (get from Coolify managed DB) |
| `CEMS_QDRANT_URL` | `http://cems-qdrant:6333` |
| `OPENROUTER_API_KEY` | Your OpenRouter API key (from openrouter.ai/keys) |
| `CEMS_ADMIN_KEY` | Random admin key for API authentication |

## Domains

- **cems.chocksy.com** → Python REST API (admin + hooks + memory API)
- **mcp-cems.chocksy.com** → Express MCP wrapper (Cursor MCP protocol)

## Backup

PostgreSQL backup created at: `~/cems_backup_20260121.sql` on Hetzner server
