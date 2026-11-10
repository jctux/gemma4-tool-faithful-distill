#!/usr/bin/env bash
# Resumable fetch of a whole HF repo with curl: scripts/hf_fetch.sh REPO DEST (file list in DEST/.files)
set -u
REPO="$1"; DEST="$2"
while read -r f; do
  [ -z "$f" ] && continue
  [ -f "$DEST/$f" ] && continue
  mkdir -p "$DEST/$(dirname "$f")"
  "$(dirname "$0")/curl_resume.sh" "https://huggingface.co/$REPO/resolve/main/$f" "$DEST/$f" || echo "FAILED $f"
done < "$DEST/.files"
echo "FETCH DONE $REPO"
