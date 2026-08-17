#!/usr/bin/env bash
# Deploy OptionSmith to the trading server.
#
#   ./deploy.sh             sync + install + restart
#   ./deploy.sh --dry-run   show what would change, touch nothing
#
# Never touches server-side state: .env, the virtualenv, __pycache__. The
# advisor places no orders and holds no broker credentials, so a restart here
# cannot disturb the broker session — unlike a gateway deploy.
set -euo pipefail

SERVER="${SERVER:-root@192.168.133.205}"
DEST="${DEST:-/opt/optionsmith}"
UNIT="optionsmith.service"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="-n"

cd "$(dirname "$0")"
echo "deploying -> $SERVER:$DEST ${DRY:+(DRY RUN)}"

EXCLUDES=(--exclude .git --exclude .venv --exclude venv --exclude __pycache__
          --exclude .DS_Store --exclude ".env" --exclude "*.pyc")

CHANGED=$(rsync -az --delete -i $DRY "${EXCLUDES[@]}" ./ "$SERVER:$DEST/" \
          | awk '{print $2}' | grep -v '/$' || true)

if [ -n "$CHANGED" ]; then
  echo "== changed:"; echo "$CHANGED" | sed 's/^/   /' | head -20
else
  echo "== no changes"
fi

if [ -n "$DRY" ]; then echo "dry run — nothing installed or restarted."; exit 0; fi

remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$SERVER" "$@"; }

remote "test -d $DEST/.venv" || {
  echo ">> first deploy — creating the virtualenv"
  remote "cd $DEST && python3 -m venv .venv && ./.venv/bin/pip install -q --upgrade pip"
}
if echo "$CHANGED" | grep -q "requirements.txt" || ! remote "test -d $DEST/.venv/lib"; then
  echo ">> installing dependencies"
  remote "cd $DEST && ./.venv/bin/pip install -q -r requirements.txt"
fi

remote "test -f $DEST/.env" || {
  echo "!! $DEST/.env is missing — the dashboard will start but live data will"
  echo "!! be unavailable. Create it from .env.example and fill in the service"
  echo "!! credentials (register_service --name optionsmith --scopes market)."
}

echo ">> installing/refreshing the unit"
remote "install -m 644 $DEST/deploy/$UNIT /etc/systemd/system/$UNIT \
        && systemctl daemon-reload && systemctl enable --now $UNIT \
        && systemctl restart $UNIT"
sleep 3

echo "== health =="
remote "systemctl is-active $UNIT; curl -s -m 8 http://127.0.0.1:8030/health; echo; \
        curl -s -m 20 http://127.0.0.1:8030/api/gateway/status; echo"
echo "deploy done."
