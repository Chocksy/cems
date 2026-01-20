#!/bin/bash
# CEMS Client Setup Script
# Configures Claude Code to use the CEMS MCP server via HTTP

set -e

# Configuration
CEMS_SERVER_URL="${CEMS_SERVER_URL:-http://localhost:8765/mcp}"
CEMS_API_KEY="${CEMS_API_KEY:-}"
CEMS_TEAM_ID="${CEMS_TEAM_ID:-default}"

echo "CEMS Client Setup"
echo "================="
echo "Server URL: $CEMS_SERVER_URL"
echo "Team ID: $CEMS_TEAM_ID"
echo ""

if [ -z "$CEMS_API_KEY" ]; then
    echo "Error: CEMS_API_KEY is required"
    echo ""
    echo "Get your API key from your CEMS admin:"
    echo "  curl -X POST http://localhost:8765/admin/users \\"
    echo "    -H \"Authorization: Bearer \$CEMS_ADMIN_KEY\" \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"username\": \"your-username\"}'"
    echo ""
    echo "Then run: CEMS_API_KEY=cems_usr_xxx ./setup-client.sh"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

# Path to Claude Code config
CLAUDE_CONFIG="$HOME/.claude.json"

if [ ! -f "$CLAUDE_CONFIG" ]; then
    echo "Error: Claude Code config not found at $CLAUDE_CONFIG"
    echo "Please run Claude Code at least once first."
    exit 1
fi

# Create backup
cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.bak"
echo "Backed up config to $CLAUDE_CONFIG.bak"

# Add CEMS MCP server to config
jq --arg url "$CEMS_SERVER_URL" \
   --arg api_key "$CEMS_API_KEY" \
   --arg team_id "$CEMS_TEAM_ID" \
   '.mcpServers.cems = {
      "type": "http",
      "url": $url,
      "headers": {
        "Authorization": ("Bearer " + $api_key),
        "X-Team-ID": $team_id
      }
    }' "$CLAUDE_CONFIG" > "$CLAUDE_CONFIG.tmp" && mv "$CLAUDE_CONFIG.tmp" "$CLAUDE_CONFIG"

echo ""
echo "✓ CEMS MCP server configured successfully!"
echo ""
echo "Configuration added to $CLAUDE_CONFIG:"
jq '.mcpServers.cems' "$CLAUDE_CONFIG"
echo ""
echo "Restart Claude Code to use the new MCP server."
echo ""
echo "To verify, run: claude /mcp"
