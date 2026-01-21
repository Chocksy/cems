# Coolify Docker Compose Deployment Instructions

## Quick Steps

### 1. Create Docker Compose App in Coolify UI

1. Go to Coolify UI → Create or select your project
2. Click **"New"** → **"Docker Compose"**
3. Configure:
   - **GitHub Repository**: `your-org/cems` (or your fork)
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
   - `cems.yourdomain.com` → Route to `cems-server` service (port 8765)
   - `mcp-cems.yourdomain.com` → Route to `cems-mcp` service (port 8766)

**Note**: The Traefik labels in `docker-compose.coolify.yml` should handle routing automatically, but you may need to configure domains in Coolify UI as well.

### 4. Deploy

Click **"Deploy"** in the Coolify UI.

### 5. Verify

```bash
# Test Python API
curl https://cems.yourdomain.com/health

# Test Express MCP wrapper
curl https://mcp-cems.yourdomain.com/health

# Test admin API
curl https://cems.yourdomain.com/admin \
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

- **cems.yourdomain.com** → Python REST API (admin + hooks + memory API)
- **mcp-cems.yourdomain.com** → Express MCP wrapper (Cursor MCP protocol)

## Backup

Set up automated backups for PostgreSQL and Qdrant volumes according to your infrastructure preferences.
