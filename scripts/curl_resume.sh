#!/usr/bin/env bash
# Resumable download with retries: scripts/curl_resume.sh URL OUTFILE
set -u
URL="$1"; OUT="$2"
for i in $(seq 1 200); do
  curl -L -C - --retry 5 --retry-delay 5 --speed-limit 50000 --speed-time 60 -o "$OUT.part" "$URL" && { mv "$OUT.part" "$OUT"; echo "DONE $OUT"; exit 0; }
  echo "retry $i"; sleep 5
done
exit 1
