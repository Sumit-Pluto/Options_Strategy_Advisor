#!/usr/bin/env bash
# Create the virtualenv and install the optional dependencies.
#
# The advisor's core needs none of them — `python3 -m optionsmith demo` runs on
# a bare interpreter — so this is only required for the dashboard and for live
# data from the Gateway.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "need Python 3.10+ (found: $("$PY" -V 2>&1)); set PYTHON=/path/to/python3.11" >&2
  exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> wrote .env from the example — fill in GATEWAY_CLIENT_ID/SECRET"
fi

echo "==> install complete"
echo "    offline check : ./.venv/bin/python -m optionsmith demo"
echo "    live check    : ./.venv/bin/python -m optionsmith chain RELIANCE"
