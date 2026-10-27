#!/usr/bin/env bash
# Show which runs are finished and which one is currently training.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== finished runs =="
for d in runs/*/; do
  id=$(basename "$d")
  [ "$id" = "runs" ] && continue
  if [ -f "$d/run.json" ]; then
    .venv/bin/python - "$d/run.json" "$id" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("test") or {}
v = r.get("val") or {}
print(f"  {sys.argv[2]:<18s} ep={r.get('best_epoch'):>3} "
      f"val_macro={v.get('macro_f1', float('nan')):.4f} "
      f"test_wF1={t.get('weighted_f1', float('nan')):.4f} "
      f"test_mF1={t.get('macro_f1', float('nan')):.4f} "
      f"({r.get('train_time_s', 0)/60:.0f} min, status={r.get('status','ok')})")
EOF
  elif [ -f "$d/train.log" ]; then
    last=$(grep -c "^\[epoch" "$d/train.log" 2>/dev/null || echo 0)
    echo "  $id  ...running ($last epochs logged)"
  fi
done

echo
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
