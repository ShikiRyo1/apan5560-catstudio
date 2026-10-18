#!/usr/bin/env sh
set -eu
CODE_ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
PROJECT_ROOT=$(dirname "$CODE_ROOT")
export CATSTUDIO_BASE_MODEL="$PROJECT_ROOT/05_Models/base_sd15"
export CATSTUDIO_ADAPTER="$PROJECT_ROOT/05_Models/cat_lora"
export CATSTUDIO_DEVICE="${CATSTUDIO_DEVICE:-auto}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$CODE_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" -m uvicorn catstudio.api:app --host 127.0.0.1 --port "${PORT:-8000}" --workers 1
