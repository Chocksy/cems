#!/bin/bash
# CEMS Installation Script
# Installs CEMS CLI and configures Claude Code and/or Cursor integration
#
# IMPORTANT: The CLI requires a running CEMS server.
# Deploy the server first, then run this script to install the CLI.

set -e

echo "========================================="
echo "CEMS - Continuous Evolving Memory System"
echo "========================================="
echo
echo "NOTE: This CLI requires a running CEMS server."
echo "      See DEPLOYMENT.md for server setup."
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track what was installed
INSTALLED_CLAUDE=false
INSTALLED_CURSOR=false

# 1. Check prerequisites
echo "Checking prerequisites..."

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}Error: uv is required but not installed.${NC}"
    echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}  uv: OK${NC}"

# uv manages Python versions automatically based on pyproject.toml (requires-python >= 3.11)
# It will download the correct version if needed
UV_PYTHON=$(uv python find 2>/dev/null || echo "")
if [ -n "$UV_PYTHON" ] && [ -x "$UV_PYTHON" ]; then
    PYTHON_VERSION=$("$UV_PYTHON" --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${GREEN}  python: $PYTHON_VERSION (via uv)${NC}"
else
    echo -e "${YELLOW}  python: uv will download Python 3.11+ as needed${NC}"
fi

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the CEMS directory${NC}"
    exit 1
fi

# 2. Install CEMS Python package as a CLI tool
echo
echo "Installing CEMS package..."

# Remove any stale cems installations first
if command -v cems >/dev/null 2>&1; then
    EXISTING_CEMS=$(which cems 2>/dev/null || true)
    if [ -n "$EXISTING_CEMS" ] && [[ "$EXISTING_CEMS" == /opt/homebrew/* ]]; then
        echo -e "${YELLOW}Removing stale cems installation at $EXISTING_CEMS${NC}"
        rm -f "$EXISTING_CEMS" 2>/dev/null || true
    fi
fi

# Uninstall any existing uv tool installation
uv tool uninstall cems 2>/dev/null || true

# Install using uv tool for global CLI access (editable mode for development)
uv tool install -e . --force 2>&1 | grep -v "^Resolved\|^Prepared\|^Installed" || true
echo -e "${GREEN}CEMS package installed${NC}"

# =============================================================================
# IDE Selection
# =============================================================================
echo
echo "========================================="
echo "IDE Configuration"
echo "========================================="
echo
echo "Which IDE(s) would you like to configure?"
echo "  1) Claude Code only"
echo "  2) Cursor only"
echo "  3) Both Claude Code and Cursor"
echo "  4) Skip IDE configuration"
echo
read -p "Choose [1/2/3/4]: " ide_choice

# =============================================================================
# Claude Code Installation Function
# =============================================================================
install_claude_code() {
    echo
    echo -e "${BLUE}Configuring Claude Code...${NC}"
    
    CLAUDE_DIR="$HOME/.claude"

    if [ -d "$CLAUDE_DIR" ]; then
        echo -e "${YELLOW}Existing ~/.claude folder detected!${NC}"
        echo
        echo "Options:"
        echo "  1) Fresh install (backup existing to ~/.claude.backup)"
        echo "  2) Merge (add CEMS hooks/skills to existing config)"
        echo "  3) Skip Claude Code"
        echo
        read -p "Choose [1/2/3]: " choice

        case $choice in
            1)
                echo
                echo "Backing up existing config to ~/.claude.backup..."
                if [ -d "$HOME/.claude.backup" ]; then
                    rm -rf "$HOME/.claude.backup"
                fi
                mv "$CLAUDE_DIR" "$HOME/.claude.backup"
                echo "Installing fresh CEMS config..."
                cp -r claude-setup "$CLAUDE_DIR"
                echo -e "${GREEN}Claude Code: Fresh install complete${NC}"
                INSTALLED_CLAUDE=true
                ;;
            2)
                echo
                echo "Merging CEMS into existing config..."

                # Create directories if needed
                mkdir -p "$CLAUDE_DIR/hooks"
                mkdir -p "$CLAUDE_DIR/skills/cems"

                # Copy hooks
                cp claude-setup/hooks/cems_*.py "$CLAUDE_DIR/hooks/"
                echo "  Copied CEMS hooks to ~/.claude/hooks/"

                # Copy skills
                cp -r claude-setup/skills/cems/* "$CLAUDE_DIR/skills/cems/"
                echo "  Copied CEMS skills to ~/.claude/skills/cems/"

                echo
                echo -e "${YELLOW}IMPORTANT: You need to manually add hooks to ~/.claude/settings.json${NC}"
                echo
                echo "Add these entries to your settings.json 'hooks' section:"
                echo
                cat << 'HOOKS'
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ~/.claude/hooks/cems_user_prompts_submit.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ~/.claude/hooks/cems_stop.py"
          }
        ]
      }
    ]
  }
}
HOOKS
                echo
                echo -e "${GREEN}Claude Code: Merge complete${NC}"
                INSTALLED_CLAUDE=true
                ;;
            3)
                echo "Skipping Claude Code configuration."
                ;;
            *)
                echo "Invalid choice. Skipping Claude Code."
                ;;
        esac
    else
        echo "No existing ~/.claude config found."
        echo "Installing fresh CEMS config..."
        cp -r claude-setup "$CLAUDE_DIR"
        echo -e "${GREEN}Claude Code: Fresh install complete${NC}"
        INSTALLED_CLAUDE=true
    fi
}

# =============================================================================
# Cursor Installation Function
# =============================================================================
install_cursor() {
    echo
    echo -e "${BLUE}Configuring Cursor...${NC}"
    
    CURSOR_DIR="$HOME/.cursor"
    CURSOR_HOOKS_JSON="$CURSOR_DIR/hooks.json"

    # Check if Cursor directory exists
    if [ ! -d "$CURSOR_DIR" ]; then
        echo -e "${YELLOW}~/.cursor directory not found. Creating it...${NC}"
        mkdir -p "$CURSOR_DIR"
    fi

    # Check for existing hooks
    if [ -d "$CURSOR_DIR/hooks" ] || [ -f "$CURSOR_HOOKS_JSON" ]; then
        echo -e "${YELLOW}Existing Cursor hooks detected!${NC}"
        echo
        echo "Options:"
        echo "  1) Fresh install (backup existing to ~/.cursor/hooks.backup)"
        echo "  2) Merge (add CEMS hooks to existing config)"
        echo "  3) Skip Cursor"
        echo
        read -p "Choose [1/2/3]: " choice

        case $choice in
            1)
                echo
                echo "Backing up existing hooks..."
                if [ -d "$CURSOR_DIR/hooks" ]; then
                    if [ -d "$CURSOR_DIR/hooks.backup" ]; then
                        rm -rf "$CURSOR_DIR/hooks.backup"
                    fi
                    mv "$CURSOR_DIR/hooks" "$CURSOR_DIR/hooks.backup"
                fi
                if [ -f "$CURSOR_HOOKS_JSON" ]; then
                    mv "$CURSOR_HOOKS_JSON" "$CURSOR_DIR/hooks.json.backup"
                fi
                
                echo "Installing CEMS hooks..."
                cp -r cursor-setup/hooks "$CURSOR_DIR/"
                cp cursor-setup/hooks.json "$CURSOR_HOOKS_JSON"
                echo -e "${GREEN}Cursor: Fresh install complete${NC}"
                INSTALLED_CURSOR=true
                ;;
            2)
                echo
                echo "Merging CEMS hooks..."

                # Create hooks directory if needed
                mkdir -p "$CURSOR_DIR/hooks"

                # Copy hook scripts
                cp cursor-setup/hooks/cems_*.py "$CURSOR_DIR/hooks/"
                echo "  Copied CEMS hooks to ~/.cursor/hooks/"

                echo
                echo -e "${YELLOW}IMPORTANT: You need to manually merge hooks into ~/.cursor/hooks.json${NC}"
                echo
                echo "Add these entries to your hooks.json:"
                echo
                cat cursor-setup/hooks.json
                echo
                echo -e "${GREEN}Cursor: Merge complete${NC}"
                INSTALLED_CURSOR=true
                ;;
            3)
                echo "Skipping Cursor configuration."
                ;;
            *)
                echo "Invalid choice. Skipping Cursor."
                ;;
        esac
    else
        echo "No existing Cursor hooks found."
        echo "Installing CEMS hooks..."
        mkdir -p "$CURSOR_DIR/hooks"
        cp -r cursor-setup/hooks "$CURSOR_DIR/"
        cp cursor-setup/hooks.json "$CURSOR_HOOKS_JSON"
        echo -e "${GREEN}Cursor: Fresh install complete${NC}"
        INSTALLED_CURSOR=true
    fi
}

# =============================================================================
# Execute IDE Installation Based on Choice
# =============================================================================
case $ide_choice in
    1)
        install_claude_code
        ;;
    2)
        install_cursor
        ;;
    3)
        install_claude_code
        install_cursor
        ;;
    4)
        echo "Skipping IDE configuration."
        ;;
    *)
        echo "Invalid choice. Skipping IDE configuration."
        ;;
esac

# =============================================================================
# Environment Variables Setup (Required for CLI)
# =============================================================================
echo
echo "========================================="
echo "Environment Variables (Required)"
echo "========================================="
echo
echo "The CEMS CLI requires these environment variables:"
echo "  CEMS_API_URL - Your CEMS server URL (required)"
echo "  CEMS_API_KEY - Your personal API key (required)"
echo
echo "For admin operations (user/team management):"
echo "  CEMS_ADMIN_KEY - Admin API key (optional, or use admin user's API key)"
echo

# Check if already set
if [ -n "$CEMS_API_URL" ] && [ -n "$CEMS_API_KEY" ]; then
    echo -e "${GREEN}Environment variables already set!${NC}"
    echo "  CEMS_API_URL=$CEMS_API_URL"
    echo "  CEMS_API_KEY=****${CEMS_API_KEY: -4}"
else
    read -p "Configure environment variables now? [y/N]: " add_env

    if [[ "$add_env" =~ ^[Yy]$ ]]; then
        echo
        echo "Get these from your CEMS server admin."
        echo "If you are the admin, create a user with:"
        echo "  curl -X POST https://your-server/admin/users \\"
        echo "    -H 'Authorization: Bearer \$CEMS_ADMIN_KEY' \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"username\": \"yourname\"}'"
        echo
        read -p "CEMS_API_URL (e.g., https://cems.example.com): " api_url
        read -p "CEMS_API_KEY: " api_key

        # Detect shell config file
        if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        else
            SHELL_RC="$HOME/.profile"
        fi

        echo "" >> "$SHELL_RC"
        echo "# CEMS Configuration" >> "$SHELL_RC"
        echo "export CEMS_API_URL=\"$api_url\"" >> "$SHELL_RC"
        echo "export CEMS_API_KEY=\"$api_key\"" >> "$SHELL_RC"

        echo
        echo -e "${GREEN}Added to $SHELL_RC${NC}"
        echo "Run: source $SHELL_RC"
    else
        echo
        echo -e "${YELLOW}Remember to set CEMS_API_URL and CEMS_API_KEY before using the CLI.${NC}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo "========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================="
echo

echo "Installed components:"
echo -e "  CEMS CLI:            ${GREEN}OK${NC}"
if [ "$INSTALLED_CLAUDE" = true ]; then
    echo -e "  Claude Code hooks:   ${GREEN}OK${NC}"
fi
if [ "$INSTALLED_CURSOR" = true ]; then
    echo -e "  Cursor hooks:        ${GREEN}OK${NC}"
fi

echo
echo "Next steps:"
echo "  1. Source your shell config: source ~/.zshrc (or ~/.bashrc)"
echo "  2. Set CEMS_API_URL and CEMS_API_KEY if not already set"
echo "  3. Test connection: cems health"

if [ "$INSTALLED_CLAUDE" = true ]; then
    echo
    echo "Claude Code:"
    echo "  - Restart Claude Code"
    echo "  - Test with: /remember I prefer TypeScript"
fi

if [ "$INSTALLED_CURSOR" = true ]; then
    echo
    echo "Cursor:"
    echo "  - Restart Cursor"
    echo "  - Hooks will inject memories at session start"
fi

echo
echo "CLI Commands:"
echo "  cems status           - Check connection and config"
echo "  cems health           - Check server health"
echo "  cems add \"content\"    - Add a memory"
echo "  cems search \"query\"   - Search memories"
echo
echo "Admin Commands (requires admin privileges):"
echo "  cems admin users list      - List all users"
echo "  cems admin users create X  - Create user (shows API key)"
echo "  cems admin teams list      - List all teams"
echo
echo "For full usage: cems --help"
