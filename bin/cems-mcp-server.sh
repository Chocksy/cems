#!/bin/bash
# MCP server wrapper that sets up environment with 1Password secrets

cd /Users/razvan/Development/llm-memory

# Load .env (filtering out comments)
export $(grep -v "^#" .env | xargs)

# Run server with 1Password secrets injection
exec op run --env-file=deploy/.env.secrets -- uv run cems-server
