#!/usr/bin/env bash
# Free one RTX 3060 on deb-gpu for training by stopping the ollama container that occupies it.
# Rootless Podman as user jc only — never sudo. Restore with scripts/restore_gpu.sh.
set -euo pipefail
TARGET="${1:-academy-ollama}"   # academy-ollama -> GPU1 (10 GB); platform-ollama -> GPU0 (8 GB)
HOST="${GPU_HOST:-user@gpu-host}"
ssh "$HOST" bash -s "$TARGET" <<'REMOTE'
set -euo pipefail
TARGET="$1"
echo "Stopping $TARGET ..."
podman stop "$TARGET"
sleep 2
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
REMOTE
