#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker info --format '{{.ServerVersion}}'
docker compose build catstudio
docker compose --profile setup run --rm assets scripts/download_assets.py sd15
docker compose --profile setup run --rm assets scripts/fetch_adapter.py
docker compose up -d catstudio
printf '%s\n' 'Open http://127.0.0.1:8000 after the service becomes healthy. The first generation loads the weights.'
