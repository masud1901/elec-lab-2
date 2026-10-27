#!/usr/bin/env bash
# Import results produced elsewhere (e.g. on Google Colab) into this checkout.
#
#   scripts/import_colab_results.sh path/to/ablation_runs.zip
#
# The archive must contain one directory per run with run.json, history.csv,
# per_class.csv, confusion_test.npy and config.json.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "usage: $0 <ablation_runs.zip>" >&2
  exit 2
fi

ZIP=$1
TMP=$(mktemp -d /tmp/opencode/import.XXXXXX)
unzip -q -o "$ZIP" -d "$TMP"

# the archive contains run directories at the top level or under collected/
if [ -d "$TMP/collected" ]; then
  SRC="$TMP/collected"
else
  SRC="$TMP"
fi

count=0
for d in "$SRC"/*/; do
  [ -f "$d/run.json" ] || continue
  id=$(basename "$d")
  if [ -f "runs/$id/run.json" ]; then
    echo "[skip] $id already present"
    continue
  fi
  mkdir -p "runs/$id"
  cp "$d"/* "runs/$id/" 2>/dev/null || true
  echo "[import] $id"
  count=$((count + 1))
done

echo "[import] $count run(s) imported into runs/"
echo "[import] now run: python summarize.py"

rm -rf "$TMP"
