#!/usr/bin/env bash
# Remove leftover swebench sandbox containers on deb-gpu (from killed teacher/gold runs).
# Usage: scripts/clean_sandboxes.sh [MAX_AGE=40m] [--dry-run]   (rootless podman, user jc — never sudo)
set -uo pipefail
MAX_AGE="${1:-40m}"
DRY="${2:-}"
ssh -o BatchMode=yes ${GPU_HOST:-user@gpu-host} bash -s "$MAX_AGE" "$DRY" <<'REMOTE'
ids=$(podman ps -a -q --filter ancestor=localhost/swebench-sandbox:latest --filter "until=$1")
n=$(echo "$ids" | grep -c . || true)
if [ -n "$2" ]; then echo "would remove $n sandbox containers older than $1"; exit 0; fi
[ -n "$ids" ] && echo "$ids" | xargs -r podman rm -f >/dev/null 2>&1
echo "removed $n sandbox containers older than $1; remaining: $(podman ps -q --filter ancestor=localhost/swebench-sandbox:latest | wc -l)"
REMOTE
