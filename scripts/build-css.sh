#!/usr/bin/env bash
# Build Tailwind CSS for the wiki dashboard
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WIKI_DIR="$PROJECT_DIR/src/cems/static/wiki"

# Check for tailwindcss binary
if command -v tailwindcss &>/dev/null; then
  TW="tailwindcss"
elif [ -x "$PROJECT_DIR/tailwindcss" ]; then
  TW="$PROJECT_DIR/tailwindcss"
else
  echo "tailwindcss not found. Install:"
  echo "  curl -sLo tailwindcss https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64 && chmod +x tailwindcss"
  exit 1
fi

if [ "$1" = "--watch" ]; then
  echo "Watching for changes..."
  $TW -i "$WIKI_DIR/input.css" -o "$WIKI_DIR/style.css" --watch
else
  echo "Building production CSS..."
  $TW -i "$WIKI_DIR/input.css" -o "$WIKI_DIR/style.css" --minify
  echo "Done: $(wc -c < "$WIKI_DIR/style.css" | tr -d ' ') bytes"
fi
