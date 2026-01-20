#!/bin/bash
# CEMS Installation Script
# Installs CEMS and configures Claude Code integration

set -e

echo "========================================="
echo "CEMS - Continuous Evolving Memory System"
echo "========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# 2. Install CEMS Python package
echo
echo "Installing CEMS package..."
uv pip install -e . 2>&1 | grep -v "^Resolved\|^Prepared\|^Installed" || true
echo -e "${GREEN}CEMS package installed${NC}"

# 3. Check for existing ~/.claude
echo
CLAUDE_DIR="$HOME/.claude"

if [ -d "$CLAUDE_DIR" ]; then
    echo -e "${YELLOW}Existing ~/.claude folder detected!${NC}"
    echo
    echo "Options:"
    echo "  1) Fresh install (backup existing to ~/.claude.backup)"
    echo "  2) Merge (add CEMS hooks/skills to existing config)"
    echo "  3) Cancel"
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
            echo -e "${GREEN}Fresh install complete${NC}"
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
            echo -e "${GREEN}Merge complete${NC}"
            ;;
        3)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "No existing ~/.claude config found."
    echo "Installing fresh CEMS config..."
    cp -r claude-setup "$CLAUDE_DIR"
    echo -e "${GREEN}Fresh install complete${NC}"
fi

# 4. Environment setup
echo
echo "========================================="
echo "Environment Variables"
echo "========================================="
echo
echo "CEMS requires these environment variables:"
echo "  CEMS_API_URL - Your CEMS server URL"
echo "  CEMS_API_KEY - Your CEMS API key"
echo
read -p "Add to shell profile now? [y/N]: " add_env

if [[ "$add_env" =~ ^[Yy]$ ]]; then
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
fi

# 5. Summary
echo
echo "========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================="
echo
echo "Next steps:"
echo "  1. Restart your terminal (or source your shell rc)"
echo "  2. Restart Claude Code"
echo "  3. Test with: /remember I prefer TypeScript"
echo
echo "Configuration:"
echo "  Claude config: $CLAUDE_DIR"
echo "  CEMS skills:   $CLAUDE_DIR/skills/cems/"
echo "  CEMS hooks:    $CLAUDE_DIR/hooks/cems_*.py"
echo
echo "For CLI usage, run: cems --help"
