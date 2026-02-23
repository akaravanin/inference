#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.yml"

echo "Stopping inference server..."
docker compose -f "$COMPOSE_FILE" down

echo "All containers stopped."
