#!/usr/bin/env bash
# Stop the four ranks, save one gzipped archive with their logs and keep the newest LOG_KEEP (10).
# Usage: ./stop.sh; DRY_RUN=1 ./stop.sh prints the plan and changes nothing.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ && "$LOG_KEEP" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG 'STOP_TIMEOUT and LOG_KEEP must be positive integers.'
if [[ "$DRY_RUN" == 1 ]]; then
  plan docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME"
  log "  Save all four rank logs and Docker output as $LOG_DIR/glm53-<timestamp>.tar.gz; keep newest $LOG_KEEP archives."
  plan docker rm "$CONTAINER_NAME"
  exit 0
fi
command -v docker >/dev/null || fail E_DEPENDENCY 'Missing command: docker.'
docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
if [[ "${GLM_STOP_LOCK_HELD:-0}" != 1 ]]; then
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
fi
if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  log "No container named $CONTAINER_NAME. Nothing to stop."; exit 0
fi
log "Stopping $CONTAINER_NAME (up to $STOP_TIMEOUT seconds; active requests will end)."
docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null || fail E_STOP 'Docker could not stop the container. Inspect docker ps and retry.'
run_logs=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/logs"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")
mkdir -p "$LOG_DIR"
archive="$LOG_DIR/glm53-$(date -u +%Y%m%dT%H%M%SZ)-$$.tar.gz"
if [[ -z "$run_logs" || ! -d "$run_logs" ]]; then
  fail E_LOG 'Cannot locate the rank log directory. The stopped container was retained; inspect its mounts before removing it.'
fi
docker logs --timestamps "$CONTAINER_NAME" > "$run_logs/container.log" 2>&1 || fail E_LOG 'Could not save Docker output. The stopped container was retained.'
tar -czf "$archive.tmp" -C "$run_logs" . || fail E_LOG 'Could not archive logs. Check LOG_DIR space and permissions; the stopped container was retained.'
mv -- "$archive.tmp" "$archive"
docker rm "$CONTAINER_NAME" >/dev/null || fail E_STOP 'Logs were saved, but Docker could not remove the stopped container.'
# Delete raw logs only after a successful archive and only in this recipe-owned run directory.
if [[ "$run_logs" == "$LOG_DIR"/run-* && ! -L "$run_logs" ]]; then
  rm -rf -- "$run_logs"
  [[ ! -L "$LOG_DIR/current" || "$(readlink -- "$LOG_DIR/current")" != "$run_logs" ]] || rm -- "$LOG_DIR/current"
fi
python3 - "$LOG_DIR" "$LOG_KEEP" <<'PY'
from pathlib import Path
import sys
archives = sorted(Path(sys.argv[1]).glob("glm53-*.tar.gz"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
for path in archives[int(sys.argv[2]):]:
    path.unlink()
PY
log "Stopped. Logs saved to $archive (newest $LOG_KEEP kept)."
