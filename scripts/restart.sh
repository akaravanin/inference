#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.yml"

echo "Stopping inference server..."
docker compose -f "$COMPOSE_FILE" down

echo "Rebuilding and starting inference server..."
docker compose -f "$COMPOSE_FILE" up --build -d

echo ""
echo "Services running:"
docker compose -f "$COMPOSE_FILE" ps
echo ""
echo "  Frontend  → http://localhost:3000"
echo "  GraphQL   → http://localhost:8080/graphql"
