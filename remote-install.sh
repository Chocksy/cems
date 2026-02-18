#!/bin/bash
# CEMS — One-command remote installer
#
# Usage (interactive — download first):
#   curl -sSf https://raw.githubusercontent.com/chocksy/cems/main/remote-install.sh -o install-cems.sh
#   bash install-cems.sh
#
# Usage (non-interactive — pipe with env vars):
#   CEMS_API_KEY=your-key CEMS_API_URL=https://cems.example.com \
#     curl -sSf https://raw.githubusercontent.com/chocksy/cems/main/remote-install.sh | bash
#
# Environment variables:
#   CEMS_API_KEY   — Your CEMS API key (required for non-interactive)
#   CEMS_API_URL   — CEMS server URL (default: https://cems.chocksy.com)
#   CEMS_TOOL      — Which tool to configure: claude, cursor, goose, all (default: claude)
#
# What it does:
#   1. Installs uv (if missing)
#   2. Installs CEMS CLI + observer daemon via uv tool
#   3. Runs `cems setup` to configure hooks, skills, and credentials
#
# No git clone required. Everything is bundled in the Python package.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "CEMS - Continuous Evolving Memory System"
echo "========================================="
echo

# ─── 1. Ensure uv is installed ───────────────────────────────────────────────

if command -v uv >/dev/null 2>&1; then
    echo -e "${GREEN}uv: already installed${NC}"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the uv env so it's available immediately
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi

    # Add to PATH for this session if not already there
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv >/dev/null 2>&1; then
        echo -e "${GREEN}uv: installed${NC}"
    else
        echo -e "${RED}Failed to install uv. Install manually: https://docs.astral.sh/uv/${NC}"
        exit 1
    fi
fi

# ─── 2. Install CEMS package ─────────────────────────────────────────────────

echo
echo "Installing CEMS..."

uv tool install "cems @ git+https://github.com/chocksy/cems.git" --force 2>&1 \
    | grep -v "^Resolved\|^Prepared\|^Installed" || true

# Verify installation
if command -v cems >/dev/null 2>&1; then
    echo -e "${GREEN}CEMS installed (cems, cems-server, cems-observer)${NC}"
else
    # uv tool bin might not be on PATH yet
    UV_BIN="$HOME/.local/bin"
    if [ -x "$UV_BIN/cems" ]; then
        export PATH="$UV_BIN:$PATH"
        echo -e "${GREEN}CEMS installed${NC}"
        echo -e "${YELLOW}Note: add $UV_BIN to your PATH (restart shell or: source ~/.zshrc)${NC}"
    else
        echo -e "${RED}Installation failed. Try manually: uv tool install \"cems @ git+https://github.com/chocksy/cems.git\"${NC}"
        exit 1
    fi
fi

# ─── 3. Run cems setup ───────────────────────────────────────────────────────

echo

# Build setup flags from environment variables
TOOL="${CEMS_TOOL:-claude}"
case "$TOOL" in
    claude)  SETUP_FLAGS="--claude" ;;
    cursor)  SETUP_FLAGS="--cursor" ;;
    goose)   SETUP_FLAGS="--goose" ;;
    all)     SETUP_FLAGS="--claude --cursor --goose" ;;
    *)
        echo -e "${RED}Unknown CEMS_TOOL value: $TOOL (expected: claude, cursor, goose, all)${NC}"
        exit 1
        ;;
esac

if [ -n "$CEMS_API_KEY" ]; then
    SETUP_FLAGS="$SETUP_FLAGS --api-key $CEMS_API_KEY"
fi
if [ -n "$CEMS_API_URL" ]; then
    SETUP_FLAGS="$SETUP_FLAGS --api-url $CEMS_API_URL"
fi

# shellcheck disable=SC2086
cems setup $SETUP_FLAGS
