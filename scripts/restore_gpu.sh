#!/usr/bin/env bash
# Restart the ollama container stopped by scripts/free_gpu.sh.
set -euo pipefail
TARGET="${1:-academy-ollama}"
HOST="${GPU_HOST:-user@gpu-host}"
ssh "$HOST" "podman start '$TARGET' && sleep 3 && podman ps --filter name='$TARGET' --format '{{.Names}} {{.Status}}' && nvidia-smi --query-gpu=index,memory.used --format=csv"
