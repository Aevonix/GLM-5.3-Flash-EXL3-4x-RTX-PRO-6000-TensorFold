#!/usr/bin/env bash
# Settings in precedence order: exported environment, scripts/local.sh, .env, defaults.
# local.sh is trusted Bash. .env is read as literal KEY=value lines, never executed.
# DENSE=fp8 is the measured fidelity setting; DENSE=q4 is the faster, lower-fidelity option.
[[ "${_GLM_CONFIG_LOADED:-0}" == 1 ]] && return 0
_GLM_CONFIG_LOADED=1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$REPO_ROOT/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _cfg_key; do _cfg_env[$_cfg_key]=${!_cfg_key}; done < <(compgen -e)
  # shellcheck source=/dev/null
  source "$REPO_ROOT/scripts/local.sh"
  for _cfg_key in "${!_cfg_env[@]}"; do export "$_cfg_key=${_cfg_env[$_cfg_key]}"; done
  unset _cfg_env
fi
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
    [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _cfg_key=${BASH_REMATCH[2]}; _cfg_value=${BASH_REMATCH[3]}
    if [[ "$_cfg_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_cfg_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _cfg_value=${BASH_REMATCH[1]}
    else
      _cfg_value=${_cfg_value%%#*}; _cfg_value=${_cfg_value%"${_cfg_value##*[![:space:]]}"}
    fi
    [[ -n "${!_cfg_key+set}" ]] || export "$_cfg_key=$_cfg_value"
  done < "$REPO_ROOT/.env"
fi
unset _cfg_key _cfg_line _cfg_value

# Pinned software and weights. Downloads are made directly from their publishers.
TF_VERSION=v0.6.6
TF_COMMIT=cb2ebf0540f42604e2759b2ddef497861e928248
TF_REPO=https://github.com/ashhart/TensorFold.git
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}"
IMAGE="${IMAGE:-tensorfold-glm53:1.2.0}"
IMAGE_EXTRAS='av==18.1.0 xgrammar==0.2.8'
CONTAINER_NAME="${CONTAINER_NAME:-glm53-tf}"
MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold
MODEL_REVISION=78353f1f6eb2c96fa6c62de57b44345d1f497c38
DFLASH2_ID=incoai/GLM-5.3-Flash-DFlash2
DFLASH2_REVISION=bf582e4eacc1810f76656d1811693ff6c6737d2a
DATA_DIR="${DATA_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/model}"
DFLASH2_DIR="${DFLASH2_DIR:-$DATA_DIR/dflash2}"
KERNEL_CACHE="${KERNEL_CACHE:-$DATA_DIR/kernels}" # subdirectory per image hash
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}" # optional local login token
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.state}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_KEEP="${LOG_KEEP:-10}"                      # newest 10 compressed session logs

# One host, four GPUs; no remote ranks or external network transport.
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8020}"
HOST="${HOST:-0.0.0.0}"
MASTER_PORT="${MASTER_PORT:-29551}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-flash}"
DRAFTER="${DRAFTER:-dflash2}"                  # CC BY-NC-ND 4.0 weights; non-commercial
# mtp uses the checkpoint's own MTP head and skips DFlash2 entirely (one request at a time).
_default_parallel=40; [[ "$DRAFTER" == mtp ]] && _default_parallel=1
PARALLEL="${PARALLEL:-$_default_parallel}"
unset _default_parallel
CONTEXT="${CONTEXT:-1048576}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
VISION="${VISION:-1}"
DENSE="${DENSE:-${TF_GLM_DENSE:-fp8}}"
KV="${KV:-${TF_GLM_KV:-fp8}}"
DRAFT_POLICY="${DRAFT_POLICY:-${TF_GLM_DFLASH_POLICY:-fnc7:0.3}}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-7.5}"
KV_POOL_GIB="${KV_POOL_GIB:-60}"

# Setup and checks. DRY_RUN prints plans and commands without running Docker or touching GPUs.
DRY_RUN="${DRY_RUN:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"           # seconds for API readiness, including remaining JIT builds
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-120}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-90000}"   # per GPU; measured on 96 GB cards
MODEL_FREE_GIB="${MODEL_FREE_GIB:-180}"         # conservative first-download budgets
DFLASH2_FREE_GIB="${DFLASH2_FREE_GIB:-10}"
IMAGE_FREE_GIB="${IMAGE_FREE_GIB:-40}"
KERNEL_FREE_GIB="${KERNEL_FREE_GIB:-10}"

# Engine switches. Draft and scheduling optimizations preserve engine outputs; dense/KV quantization is lossy.
# Chunked KDA changes prompt arithmetic. See docs/PATCHES.md for exactness boundaries.
export TF_GLM_MTP="${TF_GLM_MTP:-auto}" TF_GLM_KV="$KV" TF_GLM_DENSE="$DENSE"
export TF_GLM_DFLASH_POLICY="$DRAFT_POLICY"
export TF_GLM_COPY_DRAFTS="${TF_GLM_COPY_DRAFTS:-1}" TF_GLM_COPY_MAX="${TF_GLM_COPY_MAX:-15}"
export TF_GLM_WIDE_GRAPHS="${TF_GLM_WIDE_GRAPHS:-16}" TF_GLM_COPY_REPLY_MATCH="${TF_GLM_COPY_REPLY_MATCH:-16}"
export TF_GLM_KDA_CHUNKED="${TF_GLM_KDA_CHUNKED:-1}" TF_GLM_SHARED_PREFIX="${TF_GLM_SHARED_PREFIX:-1}"
export TF_GLM_MULTI_PREFILL="${TF_GLM_MULTI_PREFILL:-1}" TF_GLM_MULTI_LONE="${TF_GLM_MULTI_LONE:-0}"
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-32}" TF_GLM_L2PF="${TF_GLM_L2PF:-1}" TF_GLM_EXL3_LOADS="${TF_GLM_EXL3_LOADS:-nc}"
# prompt chunks: rows split between the ranks, exchanged by the GPUs' copy engines over CUDA IPC (patch 0062), two
# lanes a chunk (0063), the indexer's token selection split between the ranks (0060), 4,096-row chunks
export TF_GLM_HC_SPLIT="${TF_GLM_HC_SPLIT:-1}" TF_GLM_PREFILL_OVERLAP="${TF_GLM_PREFILL_OVERLAP:-2}"
export TF_GLM_OVERLAP_PIECES="${TF_GLM_OVERLAP_PIECES:-4}" TF_GLM_PREFILL_ROWS="${TF_GLM_PREFILL_ROWS:-4096}"
export TF_GLM_HC_EXCHANGE="${TF_GLM_HC_EXCHANGE:-ce}" TF_GLM_PREFILL_LANES="${TF_GLM_PREFILL_LANES:-2}"
export TF_GLM_INDEX_SPLIT="${TF_GLM_INDEX_SPLIT:-1}" TF_GLM_FILL_ROWS="${TF_GLM_FILL_ROWS:-2048}"
# chunks under 512 rows run unsplit (a short remainder after a kept prompt: no exchange handshakes; same bits)
export TF_GLM_HC_SPLIT_MIN_ROWS="${TF_GLM_HC_SPLIT_MIN_ROWS:-512}"
# the KDA prompt recurrences on a stream of their own (0071), and the drafter skips prompt rows no draft reads
export TF_GLM_KDA_OVERLAP="${TF_GLM_KDA_OVERLAP:-2}" TF_GLM_DRAFT_PROMPT_SKIP="${TF_GLM_DRAFT_PROMPT_SKIP:-1}"
# prompt kernels (this release): the KDA recurrence over 64 SMs in 32-column pieces (0072), the routed experts' K12 kernels on
# blocks of up to 80 members (0073), the MoE glue (0074), DSA's fused writes, the warp-role sparse attention and the
# two-read top-512 (0076, 0080), the index split's pools over CUDA IPC (0076), KDA lanes swapping their bf16 inputs
# and the peers' copy engines writing lane rows in place (0081)
export TF_GLM_KDA_SPLIT="${TF_GLM_KDA_SPLIT:-32:512}" TF_GLM_EXPERT_PROMPT_KERNEL="${TF_GLM_EXPERT_PROMPT_KERNEL:-k12-p80}"
export TF_GLM_MOE_GLUE="${TF_GLM_MOE_GLUE:-side,defer,combine,shared,rowsplit}" TF_GLM_MOE_SHARED_TILE="${TF_GLM_MOE_SHARED_TILE:-1}"
export TF_GLM_DSA_FUSE="${TF_GLM_DSA_FUSE:-1}" TF_GLM_SPARSE_FAST="${TF_GLM_SPARSE_FAST:-2}" TF_GLM_TOPK_FAST="${TF_GLM_TOPK_FAST:-1}"
export TF_GLM_INDEX_SPLIT_GATHER="${TF_GLM_INDEX_SPLIT_GATHER:-ipc}"
_lane_inputs_default=0; [[ "$TF_GLM_DENSE" == q4 ]] && _lane_inputs_default=kda
export TF_GLM_LANE_INPUTS="${TF_GLM_LANE_INPUTS:-$_lane_inputs_default}" TF_GLM_LANE_DIRECT="${TF_GLM_LANE_DIRECT:-1}"
# concurrent rounds: a 128-row batched window with CUDA graphs every 8 rows past 64 (0077), async round messages,
# packed sampler, sampling on the GPU (0065)
export TF_GLM_MULTI_WINDOW="${TF_GLM_MULTI_WINDOW:-128}" TF_GLM_MULTI_GRAPH_STEP="${TF_GLM_MULTI_GRAPH_STEP:-8}"
export TF_GLM_MULTI_ASYNC="${TF_GLM_MULTI_ASYNC:-1}"
export TF_GLM_MULTI_SAMPLER="${TF_GLM_MULTI_SAMPLER:-packed}" TENSORFOLD_NUCLEUS_UNION="${TENSORFOLD_NUCLEUS_UNION:-1}"
export TENSORFOLD_GPU_SAMPLE="${TENSORFOLD_GPU_SAMPLE:-1}" TENSORFOLD_PRIORITY="${TENSORFOLD_PRIORITY:-1}"
# decode rounds (this release): partials of windows of 8 rows and more exchanged in two shots, the scatter on the second
# communicator (0078); the drafter's fused GPU pass with all-reduce rank sums, block rows by stream count and the
# next round's pass launched at a round's end (0075, 0083)
export TF_GLM_TWOSHOT_ROWS="${TF_GLM_TWOSHOT_ROWS:-8}" TF_GLM_TWOSHOT_KIND="${TF_GLM_TWOSHOT_KIND:-bf16}"
export TF_GLM_TWOSHOT_SCATTER="${TF_GLM_TWOSHOT_SCATTER:-exchange}" TF_GLM_TWOSHOT_GATHER="${TF_GLM_TWOSHOT_GATHER:-comm}"
export TF_GLM_DRAFT_FAST="${TF_GLM_DRAFT_FAST:-all}" TF_GLM_DRAFT_FAST_GATHER="${TF_GLM_DRAFT_FAST_GATHER:-auto16}"
export TF_GLM_DRAFT_FAST_BLOCK="${TF_GLM_DRAFT_FAST_BLOCK:-auto}" TF_GLM_DRAFT_FAST_AHEAD="${TF_GLM_DRAFT_FAST_AHEAD:-1}"
# decode fusions (patch 0069): each is checked bit for bit on the GPU at start and turns itself off on any difference
export TF_GLM_FUSE="${TF_GLM_FUSE:-all}"
# warm turns: the host reuses a kept turn's rendering, token ids and hashes (0079)
export TF_W1_INCREMENTAL="${TF_W1_INCREMENTAL:-1}"
# per-GPU launch table written by scripts/glm_autotune.py into KERNEL_CACHE (the first start); unset if absent
export TF_GLM_TUNE="${TF_GLM_TUNE:-/cache/launch-table.json}"
export TENSORFOLD_NO_UPDATE_CHECK=1
export TF_GLM_COMM=nccl
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB" TF_GLM_CACHE_GIB="$KV_POOL_GIB"

# --- the GPUs' links ----------------------------------------------------------------------------------------------
# NCCL over PCIe peer-to-peer for every collective (decode all-gathers), and a second NCCL communicator with
# peer-to-peer off for the paired sends and receives (patch 0059; also the two-shot scatter): on these cards NCCL's
# peer-to-peer send/receive runs two channels a peer and is several times slower than shared memory, its all-gathers
# are faster.
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export TF_NCCL_EXCHANGE_ENV="${TF_NCCL_EXCHANGE_ENV:-NCCL_P2P_DISABLE=1}"
NCCL_CHANNELS="${NCCL_CHANNELS:-4}"

# Preserve the pinned checkpoint's earlier-turn rendering across the engine upgrade.
export TF_GLM_CLEAR_THINKING="${TF_GLM_CLEAR_THINKING:-1}"
