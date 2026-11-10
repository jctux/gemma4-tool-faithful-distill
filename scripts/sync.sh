#!/usr/bin/env bash
# Push repo code (not data) to the deb-gpu working copy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --delete --exclude .git --exclude data --exclude .venv --exclude .remember --exclude runs --exclude logs --exclude "adapters_*" --exclude eval_subs --exclude "*.pyc" \
  "$ROOT/" ${GPU_HOST:-user@gpu-host}:/home/jc/kagle/
