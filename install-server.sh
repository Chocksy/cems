#!/usr/bin/env bash
# CEMS server installer.
#
#   curl -fsSL https://getcems.com/install-server.sh | bash -s -- --private --yes
#
# Flags:
#   --private          Private mode, CPU preset (Ollama on this box)
#   --private-gpu      Private mode, GPU preset (needs NVIDIA container toolkit)
#   --openrouter-key K Default mode key (or set OPENROUTER_API_KEY)
#   --dir PATH         Install directory (default /opt/cems)
#   --yes              No prompts
#   --dry-run          Print what would run, change nothing
set -euo pipefail

RAW="https://raw.githubusercontent.com/chocksy/cems/main"
DIR="/opt/cems"
MODE="default"
YES=0
DRY=0
KEY="${OPENROUTER_API_KEY:-}"

need_value() {
  if [ "$#" -lt 2 ] || [ -z "$2" ]; then
    echo "Flag $1 needs a value." >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --private) MODE="cpu" ;;
    --private-gpu) MODE="gpu" ;;
    --openrouter-key) need_value "$@"; KEY="$2"; shift ;;
    --dir) need_value "$@"; DIR="$2"; shift ;;
    --yes|-y) YES=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

say() { printf "${GREEN}==>${NC} %s\n" "$*"; }
err() { printf "${RED}%s${NC}\n" "$*" >&2; }
run() {
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

if [ "$MODE" = "default" ] && [ -z "$KEY" ]; then
  err "Default mode needs an OpenRouter key: pass --openrouter-key or set OPENROUTER_API_KEY. Or use --private."
  exit 1
fi

case "$MODE" in
  cpu) PRESET=".env.private-cpu.example"; PROFILE="--profile private"; FILES="-f docker-compose.yml" ;;
  gpu) PRESET=".env.private-gpu.example"; PROFILE="--profile private"; FILES="-f docker-compose.yml -f docker-compose.gpu.yml" ;;
  *)   PRESET=".env.example";             PROFILE="";                  FILES="-f docker-compose.yml" ;;
esac

say "Mode: $MODE — preset $PRESET, install dir $DIR"

if [ "$YES" = 0 ] && [ "$DRY" = 0 ]; then
  printf 'Install CEMS into %s? [y/N] ' "$DIR"
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

if ! command -v docker >/dev/null 2>&1; then
  say "Installing Docker"
  run sh -c "curl -fsSL https://get.docker.com | sh"
fi
if [ "$DRY" = 0 ] && ! docker compose version >/dev/null 2>&1; then
  err "docker compose plugin missing (need 2.20 or newer). Install docker-compose-plugin and re-run."
  exit 1
fi

say "Preparing $DIR"
run mkdir -p "$DIR/deploy"
run sh -c "curl -fsSL $RAW/deploy/docker-compose.yml -o $DIR/deploy/docker-compose.yml"
if [ "$MODE" = gpu ]; then
  run sh -c "curl -fsSL $RAW/deploy/docker-compose.gpu.yml -o $DIR/deploy/docker-compose.gpu.yml"
fi

if [ -f "$DIR/deploy/.env" ]; then
  say "Keeping existing $DIR/deploy/.env"
else
  say "Writing .env from $PRESET"
  run sh -c "curl -fsSL $RAW/deploy/$PRESET -o $DIR/deploy/.env"
  if [ "$DRY" = 1 ]; then
    PG="<generated>"; ADMIN="cems_admin_<generated>"
  else
    PG=$(openssl rand -hex 16); ADMIN="cems_admin_$(openssl rand -hex 16)"
  fi
  run sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$PG/; s/^CEMS_ADMIN_KEY=.*/CEMS_ADMIN_KEY=$ADMIN/" "$DIR/deploy/.env"
  if [ "$MODE" = default ]; then
    run sed -i.bak "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=$KEY|" "$DIR/deploy/.env"
  fi
  run rm -f "$DIR/deploy/.env.bak"
fi

say "Starting CEMS"
run sh -c "cd $DIR/deploy && docker compose $PROFILE $FILES up -d"

if [ "$DRY" = 0 ]; then
  say "Waiting for /health (first boot pulls models, this can take several minutes)"
  for _ in $(seq 1 60); do
    curl -fsS http://localhost:8765/health >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -fsS http://localhost:8765/health >/dev/null 2>&1; then
    err "Server did not become healthy. Check: docker compose -f $DIR/deploy/docker-compose.yml logs"
    exit 1
  fi
  ADMIN_KEY=$(grep '^CEMS_ADMIN_KEY=' "$DIR/deploy/.env" | cut -d= -f2)
  say "CEMS is up on port 8765"
  echo "Admin key (shown once, stored in $DIR/deploy/.env): $ADMIN_KEY"
  echo "Next: cems admin --admin-key \$ADMIN_KEY users create <name>"
fi
