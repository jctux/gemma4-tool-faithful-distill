#!/usr/bin/env bash
# Push competition data from the Mac to deb-gpu (/mnt/nvme/kaggle/gemma4). Idempotent, resumable.
set -euo pipefail
SRC="$(dirname "$0")/../data/gemma4/"
rsync -a --partial  --exclude '*.kaggle-partial' --exclude filelist.tsv \
  "$SRC" ${GPU_HOST:-user@gpu-host}:/mnt/nvme/kaggle/gemma4/
