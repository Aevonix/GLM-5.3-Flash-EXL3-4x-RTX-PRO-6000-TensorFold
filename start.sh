#!/usr/bin/env bash
# One command: check, prepare, compile, tune, serve, wait for the API and smoke-test.
# Usage: ./start.sh [restart]
# Configuration: scripts/config.sh, scripts/local.sh, .env or exported environment.
# DRY_RUN=1 ./start.sh prints the full plan without Docker, GPUs or file changes.
# NO_ANIM=1 ./start.sh shows the static command deck; NO_COLOR=1 shows plain text.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
# shellcheck source=scripts/banner.sh
source ./scripts/banner.sh
case "${1:-}" in
  '') MODE=start ;;
  restart) MODE=restart ;;
  help|-h|--help) sed -n '2,6s/^# //p' "$0"; exit 0 ;;
  *) fail E_CONFIG 'Usage: ./start.sh [restart]. Set serving options in the environment or scripts/local.sh.' ;;
esac
(( $# <= 1 )) || fail E_CONFIG 'Usage: ./start.sh [restart].'
validate_config
set_cache_paths
banner
printf '\nAevonix Research TensorFold start\n%s | %s concurrent | %s-token context | %s dense | %s KV | port %s\n\n' \
  "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$DENSE" "$KV" "$PORT"
step() { printf '\n[%s/6] %s\n' "$1" "$2"; }
API_HOST="$HOST"
[[ "$HOST" != 0.0.0.0 && "$HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

live_message() {
  brand_style
  printf '%sGLM-5.3-Flash-EXL3 is now LIVE! on port %s%s\n%sEndpoint: %s/v1%s\n' \
    "$AE_AMBER" "$PORT" "$AE_RESET" "$AE_TEAL" "$URL" "$AE_RESET"
}

# The optional upstream key is passed by environment name; never print it in a dry run.
api_curl() {
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    curl --header @<(printf 'Authorization: Bearer %s\n' "$TENSORFOLD_API_KEY") "$@"
  else
    curl "$@"
  fi
}

finish_start() {
  step 6 'Wait for the API and run a greedy smoke test'
  SMOKE_JSON=$(python3 - "$SERVED_NAME" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: ready"}], "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
PY
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    plan curl --fail --silent --max-time 5 "$URL/v1/models"
    plan curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$SMOKE_JSON"
    log 'Would validate a nonempty completion, then print:'
    live_message
    exit 0
  fi
  # Show rank 0 while startup finishes compiling and capturing graphs. A heartbeat also
  # covers long compiler steps that have no output. EXIT always releases the log follower.
  docker logs --follow --tail 5 "$CONTAINER_NAME" &
  log_follower=$!
  trap 'kill "$log_follower" 2>/dev/null || true; wait "$log_follower" 2>/dev/null || true' EXIT
  started=$SECONDS
  heartbeat=$SECONDS
  until api_curl --fail --silent --max-time 5 "$URL/v1/models" | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")' >/dev/null 2>&1; do
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]] || fail E_START "A server rank exited. Inspect $LOG_DIR/current/rank*.log; ./stop.sh preserves all logs."
    (( SECONDS - started < WAIT_TIMEOUT )) || fail E_API "API did not become ready within $WAIT_TIMEOUT seconds. Inspect $LOG_DIR/current/rank0.log; raise WAIT_TIMEOUT if kernels are still compiling."
    if (( SECONDS - heartbeat >= 30 )); then
      log "Waiting for the API: $((SECONDS - started)) seconds elapsed (kernel compilation and graph capture can take several minutes)."
      heartbeat=$SECONDS
    fi
    sleep 5
  done
  kill "$log_follower" 2>/dev/null || true
  wait "$log_follower" 2>/dev/null || true
  trap - EXIT
  api_curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$SMOKE_JSON" > "$RUN_LOG_DIR/smoke.json" || fail E_SMOKE "Smoke request failed. Inspect $LOG_DIR/current/rank0.log and retry."
  python3 - "$RUN_LOG_DIR/smoke.json" <<'PY' || fail E_SMOKE 'The smoke test returned no answer. Inspect logs/current/smoke.json and the server log.'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    reply = json.load(f)
content = reply["choices"][0]["message"]["content"]
assert isinstance(content, str) and content.strip(), "Empty completion"
print("Smoke test passed: " + content.strip()[:120])
PY
  printf '\nHealth: %s/health\nLogs: %s/current/rank0.log\n' "$URL" "$LOG_DIR"
  live_message
}

step 1 'Check Linux, Docker, NVIDIA runtime, GPUs, disk space and optional download token'
if [[ "$DRY_RUN" == 1 ]]; then
  log '[I_DRY_RUN] Plan only: hardware checks skipped; no Docker commands, downloads or file changes are executed.'
  log "Check Linux x86_64, docker group/access, NVIDIA runtime, four idle GPUs with $MIN_GPU_FREE_MIB MiB free each, all-pairs CUDA peer access, free disk and optional HF_TOKEN."
  log 'Warn if power limits differ from the measured 250 W; never change them.'
  if [[ "$MODE" == restart ]]; then DRY_RUN=1 "$REPO_ROOT/stop.sh"; fi
else
  check_platform
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)
  if [[ "$running" == true && "$MODE" == start ]]; then
    log "$CONTAINER_NAME is already running. Endpoint: $URL/v1"
    log 'Use ./start.sh restart to apply changed settings, or ./stop.sh to stop it.'
    RUN_LOG_DIR=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/logs"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")
    [[ -d "$RUN_LOG_DIR" ]] || fail E_LOG 'The running container has no accessible log directory. Inspect its mounts.'
    finish_start
    exit 0
  fi
  # Settings are validated before an existing server is stopped. Preserve its logs first.
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
  fi
  check_gpus
  check_disk
  python3 - "$HOST" "$PORT" "$MASTER_PORT" <<'PY' || fail E_PORT 'API or rendezvous port is in use. Stop its owner or change PORT / MASTER_PORT.'
import socket, sys
for host, port in ((sys.argv[1], int(sys.argv[2])), ("127.0.0.1", int(sys.argv[3]))):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
PY
fi

step 2 'Build the pinned TensorFold image and apply all 86 release patches (skip when ready)'
prepare_image
step 3 'Download the pinned checkpoint and selected drafter (skip when ready)'
prepare_weights
step 4 'Check CUDA peer access, compile kernels and autotune the launch table (skip when ready)'
[[ "$DRY_RUN" == 1 ]] || check_gpus
check_peer_access
prepare_kernels

step 5 'Start one container with four local TensorFold ranks'
SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --reasoning-effort "$REASONING_EFFORT")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$VISION" != 1 ]] || SERVE_ARGS+=(--vision)
DRAFT_MOUNT=()
if [[ "$DRAFTER" == dflash2 ]]; then
  SERVE_ARGS=(--drafter /dflash2 "${SERVE_ARGS[@]}")
  DRAFT_MOUNT=(-v "$DFLASH2_DIR:/dflash2:ro")
else
  SERVE_ARGS=(--drafter none "${SERVE_ARGS[@]}")
  export TF_GLM_MTP=1
fi
ENV_ARGS=(-e HF_HUB_OFFLINE=1)
while IFS='=' read -r key _; do
  case "$key" in TENSORFOLD_*|TF_GLM_*|TF_NCCL_*|TF_W1_*) ENV_ARGS+=(-e "$key") ;; esac
done < <(env | sort)
NCCL_ARGS=(-e NCCL_SOCKET_IFNAME=lo -e NCCL_IB_DISABLE=1 -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" \
  -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL" -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}")
RUN_LOG_DIR="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$DRY_RUN" != 1 ]]; then
  mkdir -p "$RUN_LOG_DIR"
  cat > "$RUN_LOG_DIR/runner.sh" <<'RUNNER'
#!/usr/bin/env bash
# Each rank sees its own GPU first, with its peers visible for NCCL and CUDA IPC.
set -u
pids=()
vis() { local g; printf '%s' "$1"; for g in 0 1 2 3; do [[ "$g" == "$1" ]] || printf ',%s' "$g"; done; }
stop_ranks() { kill "${pids[@]}" 2>/dev/null || true; wait || true; }
trap 'stop_ranks; exit 143' TERM INT
for rank in 3 2 1; do
  CUDA_VISIBLE_DEVICES=$(vis "$rank") tensorfold serve /model --tp 4 --rank "$rank" \
    --master 127.0.0.1 --master-port "$MPORT" "$@" > "/logs/rank$rank.log" 2>&1 &
  pids+=("$!")
done
CUDA_VISIBLE_DEVICES=$(vis 0) tensorfold serve /model --tp 4 --rank 0 --master 127.0.0.1 --master-port "$MPORT" \
  --name "$SNAME" --host "$SHOST" --port "$SPORT" "$@" > >(tee /logs/rank0.log) 2>&1 &
pids+=("$!")
wait -n "${pids[@]}"
code=$?
printf '[runner] rank exited with code %s; stopping other ranks\n' "$code" >> /logs/rank0.log
stop_ranks
exit "$code"
RUNNER
  chmod +x "$RUN_LOG_DIR/runner.sh"
  ln -sfn -- "$RUN_LOG_DIR" "$LOG_DIR/current"
fi
log 'The wrapper launches ranks 0-3 on GPUs 0-3 inside the container, rotating visible devices for each rank.'
run docker run -d --name "$CONTAINER_NAME" --init --restart=no --gpus "\"device=$GPUS\"" --ipc=host --network host \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  "${ENV_ARGS[@]}" "${NCCL_ARGS[@]}" -e "MPORT=$MASTER_PORT" -e "SNAME=$SERVED_NAME" -e "SHOST=$HOST" -e "SPORT=$PORT" \
  -v "$MODEL_DIR:/model:ro" "${DRAFT_MOUNT[@]}" -v "$ACTIVE_CACHE:/cache" -v "$RUN_LOG_DIR:/logs" \
  --entrypoint /logs/runner.sh "$IMAGE" "${SERVE_ARGS[@]}" || fail E_START 'Docker could not launch the server. Check its error and rerun.'

finish_start
