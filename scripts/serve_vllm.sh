#!/usr/bin/env bash
# Serve a Gemma 4 competition checkpoint with vLLM (same version as the Kaggle scorer) on deb-gpu.
# Rootless podman, GPU via CDI. The image ships CUDA-575 *compat* libs that break host driver 550 (Error 804);
# the wrapper below deletes their ld.so config inside the (ephemeral) container so the host libcuda is used.
# Model ids match what `swegemma eval` expects:
#   base alias  = $ALIAS (e.g. gemma-4-e4b-it-qat-w4a16-ct)
#   adapters    = every dir under $ADAPTERS_DIR (name=path) → served as that name
#
# Usage: scripts/serve_vllm.sh [ALIAS] [MODEL_DIR] [ADAPTERS_DIR]
#   GPU=1 MAX_LEN=32768 PORT=8000 scripts/serve_vllm.sh
set -euo pipefail

ALIAS="${1:-gemma-4-e4b-it-qat-w4a16-ct}"
MODEL_DIR="${2:-/mnt/nvme/kaggle/models/gemma-4-E4B-it-qat-w4a16-ct}"
ADAPTERS_DIR="${3:-}"
shift $(( $# < 3 ? $# : 3 ))   # remaining "$@" are extra vllm flags
GPU="${GPU:-1}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:v0.19.1}"
NAME="${NAME:-vllm-gemma}"

lora_args=()
mounts=(-v "$MODEL_DIR:/model:ro")
if [[ -n "$ADAPTERS_DIR" && -d "$ADAPTERS_DIR" ]]; then
  mounts+=(-v "$ADAPTERS_DIR:/adapters:ro")
  mods=()
  max_rank=16
  for d in "$ADAPTERS_DIR"/*/; do
    n=$(basename "$d")
    [[ -f "$d/adapter_config.json" ]] || continue
    mods+=("$n=/adapters/$n")
    r=$(python3 -c "import json;print(json.load(open('$d/adapter_config.json')).get('r',16))")
    (( r > max_rank )) && max_rank=$r
  done
  if (( ${#mods[@]} )); then
    lora_args=(--enable-lora --max-loras "${#mods[@]}" --max-lora-rank "$max_rank" --lora-modules "${mods[@]}")
  fi
fi

podman rm -f "$NAME" >/dev/null 2>&1 || true
exec podman run --rm --name "$NAME" \
  --device "nvidia.com/gpu=$GPU" --security-opt=label=disable \
  --shm-size 2g -p "127.0.0.1:$PORT:8000" \
  "${mounts[@]}" \
  -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
  --entrypoint sh "$IMAGE" \
  -c 'rm -f /etc/ld.so.conf.d/00-compat-*.conf; ldconfig; exec vllm serve "$@"' _ \
  /model --served-model-name "$ALIAS" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  "${lora_args[@]}" \
  "$@"
