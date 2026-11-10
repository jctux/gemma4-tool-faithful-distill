#!/usr/bin/env bash
# Download every non-snapshot competition file (graphs, embeddings, wheels, docker, sandbox, sample_submission), 6-wide.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST=data/gemma4
export DEST
awk -F'\t' '$1!~/^snapshots\//{print $1}' $DEST/filelist.tsv | while read -r f; do [ -f "$DEST/$f" ] || echo "$f"; done \
 | xargs -P 6 -I{} sh -c 'kaggle competitions download gemma-4-developer-agent -f "{}" -p "$DEST" -q 2>&1 | grep -vE "NotOpenSSL|warnings.warn|Dataset URL|License"'
echo "DONE small files: $(find $DEST -type f | wc -l) files"
