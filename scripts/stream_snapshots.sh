#!/usr/bin/env bash
# Stream all competition snapshots through the Mac to deb-gpu one at a time (Mac disk is small).
# Skips snapshots already present remotely with the right size. Resumable; run in background.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST=data/gemma4
REMOTE="${GPU_HOST:-user@gpu-host}"
RDIR=/mnt/nvme/kaggle/gemma4/snapshots
remote_sizes=$(ssh -o BatchMode=yes $REMOTE "cd $RDIR && stat -c '%n %s' *.tgz 2>/dev/null")
awk -F'\t' '$1~/^snapshots\//{print $1"\t"$2}' $DEST/filelist.tsv | sort | while IFS=$'\t' read -r f size; do
  name=$(basename "$f")
  have=$(echo "$remote_sizes" | awk -v n="$name" '$1==n{print $2}')
  [ "$have" = "$size" ] && continue
  rm -f "$DEST/$f" "$DEST/$f.kaggle-partial"
  for try in 1 2 3; do
    kaggle competitions download gemma-4-developer-agent -f "$f" -p "$DEST" -q 2>&1 | grep -vE "NotOpenSSL|warnings.warn|Dataset URL|License"
    got=$(stat -f %z "$DEST/$f" 2>/dev/null || echo 0)
    [ "$got" = "$size" ] && break
    echo "size mismatch $name ($got != $size), retry $try"; rm -f "$DEST/$f" "$DEST/$f.kaggle-partial"
  done
  [ "$got" = "$size" ] || { echo "FAILED $name"; continue; }
  rsync -a --partial "$DEST/$f" "$REMOTE:$RDIR/" && rm -f "$DEST/$f" && echo "pushed $name ($size)"
done
echo "STREAM DONE $(date)"
