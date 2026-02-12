#!/bin/bash
# CEMS — Local development install
#
# For developers who cloned the repo. Installs in editable mode.
# For non-dev users, use the remote installer instead:
#
#   curl -sSf https://raw.githubusercontent.com/chocksy/cems/main/remote-install.sh | bash
#
set -e

echo "CEMS — Development Install"
echo

# Check prerequisites
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ ! -f "pyproject.toml" ]; then
    echo "Error: run this from the CEMS repo root."
    exit 1
fi

# Install in editable mode
uv tool uninstall cems 2>/dev/null || true
uv tool install -e . --force 2>&1 | grep -v "^Resolved\|^Prepared\|^Installed" || true
echo "Package installed (cems, cems-server, cems-observer)"

# Delegate to cems setup for hooks, skills, credentials
echo
cems setup
