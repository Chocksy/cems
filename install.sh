#!/bin/bash
# CEMS Installation Script
# Installs CEMS and configures it for Claude Code

set -e

echo "========================================="
echo "CEMS - Continuous Evolving Memory System"
echo "========================================="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found: $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the llm-memory directory"
    exit 1
fi

# Install the package
echo
echo "Installing CEMS..."
pip install -e .

# Create storage directory
CEMS_DIR="${CEMS_STORAGE_DIR:-$HOME/.cems}"
echo
echo "Creating storage directory: $CEMS_DIR"
mkdir -p "$CEMS_DIR"

# Get user info
USER_ID="${CEMS_USER_ID:-$USER}"
TEAM_ID="${CEMS_TEAM_ID:-}"

# Configure for Claude Code
CLAUDE_DIR="$HOME/.claude"
MCP_CONFIG="$CLAUDE_DIR/mcp_config.json"

echo
echo "Configuring Claude Code integration..."

# Create Claude directory if needed
mkdir -p "$CLAUDE_DIR"

# Check if mcp_config.json exists
if [ -f "$MCP_CONFIG" ]; then
    echo "Found existing mcp_config.json"
    echo "Adding CEMS server configuration..."

    # Check if CEMS is already configured
    if grep -q '"cems"' "$MCP_CONFIG"; then
        echo "CEMS is already configured in mcp_config.json"
    else
        # Add CEMS to existing config using Python
        python3 << EOF
import json

with open("$MCP_CONFIG", "r") as f:
    config = json.load(f)

if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"]["cems"] = {
    "command": "cems-server",
    "args": [],
    "env": {
        "CEMS_USER_ID": "$USER_ID",
        "CEMS_STORAGE_DIR": "$CEMS_DIR"
    }
}

with open("$MCP_CONFIG", "w") as f:
    json.dump(config, f, indent=2)

print("Added CEMS to mcp_config.json")
EOF
    fi
else
    # Create new config
    echo "Creating new mcp_config.json..."
    cat > "$MCP_CONFIG" << EOF
{
  "mcpServers": {
    "cems": {
      "command": "cems-server",
      "args": [],
      "env": {
        "CEMS_USER_ID": "$USER_ID",
        "CEMS_STORAGE_DIR": "$CEMS_DIR"
      }
    }
  }
}
EOF
fi

# Add team ID if specified
if [ -n "$TEAM_ID" ]; then
    echo "Adding team configuration: $TEAM_ID"
    python3 << EOF
import json

with open("$MCP_CONFIG", "r") as f:
    config = json.load(f)

config["mcpServers"]["cems"]["env"]["CEMS_TEAM_ID"] = "$TEAM_ID"

with open("$MCP_CONFIG", "w") as f:
    json.dump(config, f, indent=2)
EOF
fi

# Copy skills to Claude Code skills directory
SKILLS_DIR="$CLAUDE_DIR/skills/cems"
echo
echo "Installing skills to $SKILLS_DIR..."
mkdir -p "$SKILLS_DIR"
cp skills/*.md "$SKILLS_DIR/"

# Verify installation
echo
echo "Verifying installation..."
if command -v cems &> /dev/null; then
    echo "CLI installed successfully"
    cems status
else
    echo "Warning: CLI not found in PATH"
    echo "You may need to add pip's bin directory to your PATH"
fi

echo
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo
echo "Next steps:"
echo "1. Restart Claude Code to load the MCP server"
echo "2. Try these commands in Claude Code:"
echo "   /remember I prefer Python for backend work"
echo "   /recall What do I prefer?"
echo
echo "Configuration:"
echo "  User ID: $USER_ID"
echo "  Storage: $CEMS_DIR"
echo "  MCP Config: $MCP_CONFIG"
if [ -n "$TEAM_ID" ]; then
    echo "  Team ID: $TEAM_ID"
fi
echo
echo "For more options, run: cems --help"
