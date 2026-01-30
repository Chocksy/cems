#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Strategy A Migration: Remote DB → Local Docker (768-dim)
# =============================================================================
# Required env:
#   REMOTE_DB_URL=postgresql://user:pass@cems.chocksy.com:5432/cems
#   LOCAL_DB_URL=postgresql://user:pass@localhost:5432/cems
#   DUMP_FILE=./cems_remote.dump (optional)
# =============================================================================

DUMP_FILE="${DUMP_FILE:-./cems_remote.dump}"

echo "[1/5] Dump remote DB..."
pg_dump "$REMOTE_DB_URL" -Fc -f "$DUMP_FILE"

echo "[2/5] Restore into local DB..."
pg_restore -d "$LOCAL_DB_URL" --clean --if-exists "$DUMP_FILE"

echo "[3/5] Ensure pgvector extension..."
psql "$LOCAL_DB_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "[4/5] Migrate embedding column to 768-dim (DESTRUCTIVE)..."
psql "$LOCAL_DB_URL" -c "ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);"

echo "[5/5] Re-embed all memories using llama.cpp server..."
CEMS_DATABASE_URL="$LOCAL_DB_URL" \
CEMS_EMBEDDING_BACKEND=llamacpp_server \
CEMS_EMBEDDING_DIMENSION=768 \
uv run python scripts/reembed_all_memories.py

echo "Done."
