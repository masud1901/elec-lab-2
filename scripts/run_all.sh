#!/usr/bin/env bash
# Run the ablation matrix.  Existing runs (runs/<id>/run.json) are skipped.
#
#   scripts/run_all.sh core           # reference + Parts A and B
#   scripts/run_all.sh long           # Part C (long windows are slower)
#   scripts/run_all.sh regularization # Part D
#   scripts/run_all.sh optimizer      # Part E
#   scripts/run_all.sh imbalance      # Part F
#   scripts/run_all.sh all            # everything
#
# Env: PY=.venv/bin/python SEEDS="42 43 44" EPOCHS=80 PATIENCE=15 SMOKE=1
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"42 43 44"}
EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-15}
PHASE=${1:-all}
EXTRA=()

if [ "${SMOKE:-0}" = "1" ]; then
  SEEDS="42"
  EXTRA+=(--smoke)
fi

echo "[run_all] phase=$PHASE seeds='$SEEDS' epochs=$EPOCHS patience=$PATIENCE"
"$PY" - <<'EOF'
import torch
print(f"[run_all] torch={torch.__version__} cuda={torch.cuda.is_available()}",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

run () {
  local id=$1; shift
  if [ -f "runs/$id/run.json" ]; then
    echo "[skip] $id"
    return
  fi
  local resume=""
  if [ -f "runs/$id/best.pt" ]; then
    resume="--resume"
    echo "[resume] $id"
  fi
  echo "[run] $id"
  if ! "$PY" train.py --run-id "$id" --epochs "$EPOCHS" --patience "$PATIENCE" \
        $resume "${EXTRA[@]}" "$@"; then
    echo "[fail] $id (see runs/$id/train.log)"
  fi
}

run_core () {
  for seed in $SEEDS; do
    run "ref_s$seed" --part 0 --seed "$seed"
    run "A_cnn_s$seed" --part A --model cnn --seed "$seed"
    run "A_gru_s$seed" --part A --rnn gru --seed "$seed"
    run "B_conv0_s$seed" --part B --n-conv 0 --seed "$seed"
    run "B_conv1_s$seed" --part B --n-conv 1 --seed "$seed"
    run "B_conv2_s$seed" --part B --n-conv 2 --seed "$seed"
  done
}

run_long () {
  for seed in $SEEDS; do
    run "C_w20_s$seed" --part C --window 20 --seed "$seed"
    run "C_w42_s$seed" --part C --window 42 --seed "$seed"
    run "C_w60_s$seed" --part C --window 60 --seed "$seed"
  done
}

run_regularization () {
  for seed in $SEEDS; do
    run "D_no_dropout_s$seed" --part D --dropout 0 --seed "$seed"
    run "D_no_wd_s$seed" --part D --weight-decay 0 --seed "$seed"
    run "D_no_aug_s$seed" --part D --augment none --seed "$seed"
  done
}

run_optimizer () {
  for seed in $SEEDS; do
    run "E_lr1e-4_s$seed" --part E --lr 1e-4 --seed "$seed"
    run "E_lr1e-2_s$seed" --part E --lr 1e-2 --seed "$seed"
    run "E_warm1e-4_s$seed" --part E --lr 1e-4 --warmup-epochs 5 --seed "$seed"
    run "E_warm1e-2_s$seed" --part E --lr 1e-2 --warmup-epochs 5 --seed "$seed"
  done
}

run_imbalance () {
  for seed in $SEEDS; do
    run "F_focal_s$seed" --part F --loss focal --focal-gamma 2 --seed "$seed"
  done
}

case "$PHASE" in
  core) run_core ;;
  long) run_long ;;
  regularization) run_regularization ;;
  optimizer) run_optimizer ;;
  imbalance) run_imbalance ;;
  all)
    run_core
    run_long
    run_regularization
    run_optimizer
    run_imbalance
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[run_all] done phase=$PHASE"
