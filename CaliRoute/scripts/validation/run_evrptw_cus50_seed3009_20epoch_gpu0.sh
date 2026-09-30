#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../AAAI_Dataset/dataset}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3009}"
EPOCHS="${EPOCHS:-20}"
POOL="${SLPPO_POOL:-weighted}"
USE_PPO_INIT="${USE_PPO_INIT:-0}"
SKIP_PPO="${SKIP_PPO:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
METHODS_TO_RUN="${METHODS_TO_RUN:-ppo slppo dapg awbc}"

CUSTOMERS=50
CHARGING_STATIONS=10
NUM_ENVS="${NUM_ENVS:-256}"
N_TRAJ="${N_TRAJ:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-90}"
PPO_STEP_CHUNK_SIZE="${PPO_STEP_CHUNK_SIZE:-16}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_N_TRAJ="${EVAL_N_TRAJ:-50}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

VALIDATION_ROOT="results/validation/evrptw_cus50_seed3009_20epoch"
LOG_DIR="$VALIDATION_ROOT/logs"
mkdir -p "$LOG_DIR"

run_method() {
  local method="$1"
  local ppo_updates="$2"
  local run_name="CALIROUTE_VALIDATE_EVRPTW_CUS50_${method^^}_SEED${SEED}_E${EPOCHS}"
  local log_path="$LOG_DIR/${run_name}.log"
  local status_path="$LOG_DIR/${run_name}.status"
  local extra_args=()

  if [[ "$method" == "slppo" ]]; then
    extra_args+=(--pool "$POOL")
  fi
  if [[ "$USE_PPO_INIT" == "1" ]]; then
    extra_args+=(--init-checkpoint "$PPO_INIT_CKPT")
  fi

  echo "[Run] $method -> $run_name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
    --problem evrptw \
    --customers "$CUSTOMERS" \
    --charging-stations "$CHARGING_STATIONS" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --num-envs "$NUM_ENVS" \
    --n-traj "$N_TRAJ" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --ppo-step-chunk-size "$PPO_STEP_CHUNK_SIZE" \
    --num-minibatches "$NUM_MINIBATCHES" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-n-traj "$EVAL_N_TRAJ" \
    --eval-max-steps "$EVAL_MAX_STEPS" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --checkpoint-interval "$EVAL_INTERVAL" \
    --mixed-precision \
    --device cuda:0 \
    --offline-method "$method" \
    --run-name "$run_name" \
    --ppo-update-epochs "$ppo_updates" \
    "${extra_args[@]}" \
    > "$log_path" 2>&1
  echo "$?" > "$status_path"
}

PPO_RUN="CALIROUTE_VALIDATE_EVRPTW_CUS50_PPO_SEED${SEED}_E${EPOCHS}"
PPO_INIT_CKPT="${INIT_CHECKPOINT:-results/checkpoints/Cus_${CUSTOMERS}_CS_${CHARGING_STATIONS}/${PPO_RUN}/seed_${SEED}/checkpoint_epoch_$(printf '%04d' "$EVAL_INTERVAL").pt}"

for method in $METHODS_TO_RUN; do
  if [[ "$method" == "ppo" && "$SKIP_PPO" == "1" ]]; then
    continue
  fi
  case "$method" in
    ppo) run_method ppo 3 ;;
    slppo) run_method slppo 4 ;;
    dapg) run_method dapg 3 ;;
    awbc) run_method awbc 3 ;;
    *) echo "Unknown validation method: $method" >&2; exit 2 ;;
  esac
done

"$PYTHON_BIN" scripts/validation/compare_evrptw_cus50_seed3009_20epoch.py
