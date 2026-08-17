#!/usr/bin/env bash
# Run the dashboard. HOST/PORT override .env; .env overrides the defaults.
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a || true

exec ./.venv/bin/python -m optionsmith ui \
  --host "${HOST:-${OPTIONSMITH_HOST:-0.0.0.0}}" \
  --port "${PORT:-${OPTIONSMITH_PORT:-8030}}"
