#!/usr/bin/env bash
# Build the pinned image, download weights, compile and tune once. Also sourced by start.sh.
set -euo pipefail
# shellcheck source=scripts/config.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

log() { printf '%s\n' "$*"; }
fail() { printf '[%s] %s\n' "$1" "$2" >&2; exit 1; }
warn() { printf '[%s] %s\n' "$1" "$2" >&2; }
plan() { printf '  '; printf '%q ' "$@"; printf '\n'; }
run() { if [[ "$DRY_RUN" == 1 ]]; then plan "$@"; else "$@"; fi; }

validate_config() {
  local key path
  [[ "$DRY_RUN" =~ ^[01]$ ]] || fail E_CONFIG 'DRY_RUN must be 0 or 1.'
  [[ "$DRAFTER" =~ ^(dflash2|mtp)$ && "$DENSE" =~ ^(fp8|q4|bf16)$ && "$KV" =~ ^(fp8|bf16)$ ]] || fail E_CONFIG 'Use DRAFTER=dflash2|mtp, DENSE=fp8|q4|bf16 and KV=fp8|bf16.'
  for key in PARALLEL CONTEXT MAX_TOKENS PORT MASTER_PORT LOG_KEEP WAIT_TIMEOUT SMOKE_TIMEOUT STOP_TIMEOUT MIN_GPU_FREE_MIB MODEL_FREE_GIB DFLASH2_FREE_GIB IMAGE_FREE_GIB KERNEL_FREE_GIB; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG "$key must be a positive integer."
  done
  (( PARALLEL <= 64 && CONTEXT <= 1048576 && PORT <= 65535 && MASTER_PORT <= 65535 && PORT != MASTER_PORT )) || fail E_CONFIG 'PARALLEL must be 1-64, CONTEXT at most 1048576, and ports distinct in 1-65535.'
  [[ "$DRAFTER" != mtp || "$PARALLEL" == 1 ]] || fail E_CONFIG 'DRAFTER=mtp requires PARALLEL=1.'
  for key in THINKING VISION; do [[ "${!key}" =~ ^[01]$ ]] || fail E_CONFIG "$key must be 0 or 1."; done
  [[ "$REASONING_EFFORT" =~ ^(none|minimal|low|medium|high|max)$ ]] || fail E_CONFIG 'REASONING_EFFORT must be none, minimal, low, medium, high or max.'
  for key in MEMORY_RESERVE_GIB KV_POOL_GIB; do [[ "${!key}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail E_CONFIG "$key must be a number of GiB."; done
  [[ "$TF_GLM_MULTI_WINDOW" =~ ^[1-9][0-9]*$ ]] && (( TF_GLM_MULTI_WINDOW >= 16 && TF_GLM_MULTI_WINDOW <= 256 )) || fail E_CONFIG 'TF_GLM_MULTI_WINDOW must be 16-256.'
  [[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail E_CONFIG 'GPUS must list four distinct numeric GPU indices, such as 0,1,2,3.'
  IFS=, read -r -a GPU_IDS <<< "$GPUS"
  [[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)" == 4 ]] || fail E_CONFIG 'GPUS contains duplicate GPU indices.'
  [[ "$CONTAINER_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail E_CONFIG 'CONTAINER_NAME contains unsupported characters.'
  for key in DATA_DIR MODEL_DIR DFLASH2_DIR KERNEL_CACHE STATE_DIR LOG_DIR; do
    path=${!key}
    [[ "$path" == /* && "$path" != *:* && "$path" != *$'\n'* ]] || fail E_CONFIG "$key must be an absolute path without colons or newlines."
  done
  [[ "$TF_GLM_TUNE" == /cache/launch-table.json ]] || fail E_CONFIG 'TF_GLM_TUNE must be /cache/launch-table.json; it is prepared automatically.'
}

image_hash() {
  local patch
  mapfile -t PATCHES < <("$REPO_ROOT/scripts/apply-patches.sh" --list)
  (( ${#PATCHES[@]} == 86 )) || fail E_PATCHES 'The release requires exactly 86 patches.'
  { for patch in "${PATCHES[@]}"; do cat "$REPO_ROOT/$patch"; done
    cat "$REPO_ROOT/scripts/apply-patches.sh" "$REPO_ROOT/scripts/prepare.sh"
    printf '%s\n' "$TF_COMMIT" "$BASE_IMAGE" "$IMAGE_EXTRAS"
  } | sha256sum | cut -c1-16
}
set_cache_paths() {
  PATCHES_HASH=$(image_hash)
  ACTIVE_CACHE="$KERNEL_CACHE/$PATCHES_HASH"
}
weights_ready() { [[ -f "$1/config.json" && -f "$1/.revision" && "$(cat "$1/.revision")" == "$2" ]]; }
image_ready() { [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$PATCHES_HASH" ]]; }

check_platform() {
  local cmd
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail E_PLATFORM 'This release requires Linux x86_64.'
  for cmd in docker curl python3 flock gzip tar sha256sum nvidia-smi; do
    command -v "$cmd" >/dev/null || fail E_DEPENDENCY "Missing command: $cmd. See README requirements."
  done
  docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
  if (( EUID != 0 )) && ! id -nG | tr ' ' '\n' | grep -qx docker; then
    fail E_DOCKER_GROUP 'Your login is not in the docker group. Add it and start a new login session.'
  fi
  docker info --format '{{json .Runtimes}}' | python3 -c 'import json,sys; sys.exit(0 if "nvidia" in json.load(sys.stdin) else 1)' || fail E_NVIDIA_RUNTIME 'Docker has no NVIDIA runtime. Install and configure NVIDIA Container Toolkit.'
  log '[I_PLATFORM] Linux x86_64, Docker access and NVIDIA runtime are ready.'
}

check_gpus() {
  local gpu row free power used processes
  GPU_FINGERPRINT=''
  for gpu in "${GPU_IDS[@]}"; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=name,memory.free,memory.used,power.limit,driver_version --format=csv,noheader,nounits) || fail E_GPU "Cannot query GPU $gpu. Check the driver and GPUS selection."
    IFS=, read -r _gpu_name free used power _gpu_driver <<< "$row"
    free=${free//[[:space:]]/}; used=${used//[[:space:]]/}; power=${power//[[:space:]]/}
    [[ "$free" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || fail E_GPU "GPU $gpu returned invalid memory information."
    (( free >= MIN_GPU_FREE_MIB )) || fail E_GPU_MEMORY "GPU $gpu has $free MiB free; at least $MIN_GPU_FREE_MIB MiB is required. Stop other GPU workloads."
    (( used < 1000 )) || fail E_GPU_BUSY "GPU $gpu uses $used MiB. Stop other GPU workloads before launch or autotune."
    [[ "$power" =~ ^250([.]0+)?$ ]] || warn W_POWER "GPU $gpu power limit is $power W; results were measured at 250 W. No power setting was changed."
    GPU_FINGERPRINT+="$gpu:${_gpu_name}:${_gpu_driver};"
  done
  processes=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader,nounits) || fail E_GPU 'Cannot check GPU compute processes.'
  if [[ "$processes" =~ [0-9] ]]; then fail E_GPU_BUSY 'A selected GPU has a compute process. Stop other GPU workloads before launch or autotune.'; fi
  log '[I_GPU] Four GPUs have enough free memory and no compute workloads.'
}

check_disk() {
  local path ancestor device free need label docker_root i
  local -a paths=() sizes=() labels=()
  local -A need_by_device=() free_by_device=() names_by_device=()
  if ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then paths+=("$MODEL_DIR"); sizes+=("$MODEL_FREE_GIB"); labels+=(checkpoint); fi
  if [[ "$DRAFTER" == dflash2 ]] && ! weights_ready "$DFLASH2_DIR" "$DFLASH2_REVISION"; then paths+=("$DFLASH2_DIR"); sizes+=("$DFLASH2_FREE_GIB"); labels+=(DFlash2); fi
  if ! image_ready; then
    docker_root=$(docker info --format '{{.DockerRootDir}}') || fail E_DISK 'Cannot locate Docker storage.'
    paths+=("$docker_root"); sizes+=("$IMAGE_FREE_GIB"); labels+=(image)
  fi
  if [[ ! -f "$ACTIVE_CACHE/.autotune-ready" ]]; then paths+=("$ACTIVE_CACHE"); sizes+=("$KERNEL_FREE_GIB"); labels+=(kernels); fi
  for (( i=0; i<${#paths[@]}; i++ )); do
    path=${paths[i]}; need=${sizes[i]}; label=${labels[i]}; ancestor=$path
    while [[ ! -e "$ancestor" ]]; do ancestor=$(dirname -- "$ancestor"); done
    read -r device free < <(df -Pk -- "$ancestor" | awk 'END {print $1, $4}')
    [[ "$free" =~ ^[0-9]+$ ]] || fail E_DISK "Cannot inspect available disk space for $label."
    need_by_device[$device]=$(( ${need_by_device[$device]:-0} + need * 1024 * 1024 ))
    free_by_device[$device]=$free
    names_by_device[$device]="${names_by_device[$device]:-} $label"
  done
  for device in "${!need_by_device[@]}"; do
    (( free_by_device[$device] >= need_by_device[$device] )) || fail E_DISK "Need $((need_by_device[$device]/1024/1024)) GiB free for${names_by_device[$device]}; only $((free_by_device[$device]/1024/1024)) GiB available on that filesystem."
  done
  log '[I_DISK] Enough disk space for missing weights, image and kernel cache.'
  load_hf_token
  if [[ -n "${HF_TOKEN:-}" ]]; then log '[I_HF_TOKEN] Optional Hugging Face token is set; its value will not be printed.'
  else log '[I_HF_TOKEN] No Hugging Face token set; public downloads work without one. Set HF_TOKEN if access or rate limits require it.'; fi
}

prepare_image() {
  if [[ "$DRY_RUN" != 1 ]] && image_ready; then log "Image is ready: $IMAGE ($PATCHES_HASH)"; return; fi
  log "Build $IMAGE: clone TensorFold $TF_VERSION at $TF_COMMIT, apply 86 patches, install runtime."
  run docker build -t "$IMAGE" --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "TF_REPO=$TF_REPO" \
    --build-arg "TF_COMMIT=$TF_COMMIT" --build-arg "IMAGE_EXTRAS=$IMAGE_EXTRAS" --build-arg "PATCHES_HASH=$PATCHES_HASH" \
    -f - "$REPO_ROOT" <<'DOCKERFILE' || fail E_IMAGE 'Image build failed. Check disk space, network access and the first build error.'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG TF_REPO
ARG TF_COMMIT
ARG IMAGE_EXTRAS
RUN git clone --no-checkout ${TF_REPO} /opt/TensorFold && \
    git -C /opt/TensorFold checkout --detach ${TF_COMMIT} && \
    test "$(git -C /opt/TensorFold rev-parse HEAD)" = "${TF_COMMIT}"
COPY scripts/apply-patches.sh /opt/tf-patches/scripts/apply-patches.sh
COPY patches /opt/tf-patches/patches
# pip's package data omits C/CUDA headers that runtime-compiled extensions include (cuda/ipc.h): copy them in.
RUN bash /opt/tf-patches/scripts/apply-patches.sh /opt/TensorFold && \
    pip install --no-cache-dir /opt/TensorFold && pip install --no-cache-dir ${IMAGE_EXTRAS} && \
    site="$(python -c 'import os, tensorfold; print(os.path.dirname(tensorfold.__file__))')" && \
    (cd /opt/TensorFold/src/tensorfold && find . \( -name '*.h' -o -name '*.cuh' -o -name '*.hpp' \) -exec cp --parents {} "$site/" \;) && \
    test -f "$site/cuda/ipc.h" && \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar" && \
    rm -rf /opt/TensorFold
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
}

load_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -r "$HF_TOKEN_PATH" ]]; then
    IFS= read -r HF_TOKEN < "$HF_TOKEN_PATH" || true
  fi
  [[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
}

fetch_weights() {
  local repo=$1 rev=$2 dir=$3
  if weights_ready "$dir" "$rev"; then log "Weights are ready: $repo @ $rev"; return; fi
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$dir"
  log "Download $repo at pinned revision $rev."
  # Pass the token by name so neither shell tracing here nor dry-run output prints its value.
  load_hf_token
  local -a token_args=(); [[ -z "${HF_TOKEN:-}" ]] || token_args=(-e HF_TOKEN)
  run docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/huggingface "${token_args[@]}" -v "$dir:/dst" --entrypoint python "$IMAGE" -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir="/dst")' "$repo" "$rev" || fail E_DOWNLOAD 'Weight download failed. Check free space, access and optional HF_TOKEN; rerun to resume.'
  [[ "$DRY_RUN" == 1 ]] || printf '%s\n' "$rev" > "$dir/.revision"
}
prepare_weights() {
  fetch_weights "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR"
  if [[ "$DRAFTER" == dflash2 ]]; then
    log 'DFlash2 weights: CC BY-NC-ND 4.0, non-commercial. DRAFTER=mtp skips them.'
    fetch_weights "$DFLASH2_ID" "$DFLASH2_REVISION" "$DFLASH2_DIR"
  fi
}

check_peer_access() {
  run docker run --rm --gpus "\"device=$GPUS\"" --network none --entrypoint python "$IMAGE" -c \
    'import torch; assert torch.cuda.device_count() == 4, "Expected four CUDA GPUs"; missing = [(a,b) for a in range(4) for b in range(4) if a != b and not torch.cuda.can_device_access_peer(a,b)]; assert not missing, f"CUDA peer access unavailable: {missing}"; print("[I_P2P] CUDA peer access is available between all four GPUs.")' || fail E_P2P 'CUDA peer access failed. Check the GPU topology, driver and platform PCIe/IOMMU configuration.'
}

prepare_kernels() {
  local signature
  signature=$({ printf '%s\n' "$PATCHES_HASH" "${GPU_FINGERPRINT:-dry-run}" "$TF_GLM_DENSE" "$TF_GLM_EXL3_LOADS"; cat "$REPO_ROOT/scripts/glm_autotune.py"; } | sha256sum | cut -d' ' -f1)
  if [[ "$DRY_RUN" != 1 && -f "$ACTIVE_CACHE/launch-table.json" && -f "$ACTIVE_CACHE/.autotune-ready" && "$(cat "$ACTIVE_CACHE/.autotune-ready")" == "$signature" ]]; then
    log 'Compiled kernel cache and launch table are ready.'; return
  fi
  log 'Compile CUDA kernels and run one-time launch-table autotune on the first selected GPU (GPUs must be idle).'
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$ACTIVE_CACHE"
  run docker run --rm --gpus "\"device=${GPU_IDS[0]}\"" --ipc=host --network none \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e "TF_GLM_EXL3_LOADS=$TF_GLM_EXL3_LOADS" -e "TF_GLM_DENSE=$TF_GLM_DENSE" \
    -v "$ACTIVE_CACHE:/cache" -v "$REPO_ROOT/scripts:/tools:ro" -v "$MODEL_DIR/config.json:/model/config.json:ro" \
    --entrypoint python "$IMAGE" /tools/glm_autotune.py --config /model/config.json --tp 4 --rank 0 \
    --kernels exl3_dec,kda_step,q4_prefill,exl3_prompt --prefill-rows 256,512,2048,4096 --out /cache/launch-table.json || fail E_AUTOTUNE 'Kernel compilation or autotune failed. Inspect the preceding error, leave GPUs idle and rerun.'
  if [[ "$DRY_RUN" != 1 ]]; then
    python3 - "$ACTIVE_CACHE/launch-table.json" <<'PY' || fail E_AUTOTUNE 'The launch table reports incomplete tuning. Check compiler errors above and rerun with idle GPUs.'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)
assert doc.get("tables"), "Autotune produced no table"
assert all(not t.get("measured", {}).get("errors") for t in doc["tables"]), "Autotune reported failed kernels"
PY
    printf '%s\n' "$signature" > "$ACTIVE_CACHE/.autotune-ready"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cd "$REPO_ROOT"
  case "${1:-}" in
    --label) set_cache_paths; printf '%s\n' "$PATCHES_HASH"; exit 0 ;;
    ''|--autotune|--image) ;;
    *) printf 'Usage: scripts/prepare.sh [--autotune|--image|--label]\n' >&2; exit 2 ;;
  esac
  validate_config
  set_cache_paths
  if [[ "$DRY_RUN" == 1 ]]; then log '[I_DRY_RUN] Plan only: no Docker, hardware access, downloads or file changes.'
  else
    check_platform
    mkdir -p "$STATE_DIR"
    exec 8>"$STATE_DIR/start.lock"
    flock -n 8 || fail E_LOCK 'Another start or preparation is in progress.'
    [[ "${1:-}" == --image ]] || check_gpus
    check_disk
  fi
  prepare_image
  [[ "${1:-}" != --image ]] || exit 0
  prepare_weights
  [[ "$DRY_RUN" == 1 ]] || check_gpus
  check_peer_access
  prepare_kernels
  log 'Prepared. Start the API with ./start.sh.'
fi
