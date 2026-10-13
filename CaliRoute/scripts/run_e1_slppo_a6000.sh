#!/usr/bin/env bash
# Dedicated A6000 assignment: independent CVRP50 / CVRP15 SL-PPO on GPUs0 / 1.
# No native baselines, no Cus100 dependency, and no changes to training budgets.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_MODE="${1:-}"
if [[ $# -gt 1 || ( -n "$E1_MODE" && "$E1_MODE" != --execute && "$E1_MODE" != --worker ) ]]; then
  printf 'Usage: run_e1_slppo_a6000.sh [--execute]\n' >&2; exit 2
fi
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
[[ -x "$E1_PYTHON" ]] || E1_PYTHON="$(command -v python)"
export E1_PYTHON
E1_PARENT="${E1_A6000_SLPPO_ROOT:-${E1_BATCH2_ROOT:-$E1_ROOT/results/e1}}"
E1_ASSETS="${E1_BATCH2_ASSETS_ROOT:-$E1_PARENT/batch2_assets}"
E1_SCALES="${E1_A6000_SLPPO_SCALES:-50 15}"
case "$E1_SCALES" in '50 15'|50|15) ;; *) printf 'E1_A6000_SLPPO_SCALES must be 50, 15 or "50 15"\n' >&2; exit 2;; esac
E1_GPU50="${E1_CVRP50_SLPPO_GPU:-0}"
E1_GPU15="${E1_CVRP15_SLPPO_GPU:-1}"
if [[ ! "$E1_GPU50" =~ ^[0-9]+$ || ! "$E1_GPU15" =~ ^[0-9]+$ || ( "$E1_SCALES" == '50 15' && "$E1_GPU50" == "$E1_GPU15" ) ]]; then
  printf 'Concurrent scales require two distinct nonnegative GPU indices\n' >&2; exit 2
fi
for E1_SCALE in $E1_SCALES; do
  E1_PPO_KEY="E1_CVRP${E1_SCALE}_SLPPO_PPO_CHUNK"
  E1_EXPERT_KEY="E1_CVRP${E1_SCALE}_SLPPO_EXPERT_CHUNK"
  E1_PPO="${!E1_PPO_KEY:-${E1_SLPPO_PPO_CHUNK:-32}}"
  E1_EXPERT="${!E1_EXPERT_KEY:-${E1_EXPERT_CHUNK:-128}}"
  if [[ ! "$E1_PPO" =~ ^[1-9][0-9]*$ || ! "$E1_EXPERT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PPO/expert chunks must be positive integers\n' >&2; exit 2
  fi
  E1_CARD="$E1_GPU50"; [[ "$E1_SCALE" != 15 ]] || E1_CARD="$E1_GPU15"
  printf 'GPU%s: CVRP%s SL-PPO; single GPU; seed3009; scratch; 1500 online loops; batch64; n_traj50; PPO passes4; minibatches4\n' "$E1_CARD" "$E1_SCALE"
  printf '  PPO chunk=%s, expert chunk=%s (A6000 candidates; must pass hardware preflight).\n' "$E1_PPO" "$E1_EXPERT"
  printf '  Campaign: %s/E1_CVRP%s_SLPPO_S3009_A6000_V1\n' "$E1_PARENT" "$E1_SCALE"
done
if [[ -z "$E1_MODE" ]]; then
  printf 'Preview only. --execute detaches ALL preparation and queues both scales independently.\n'; exit 0
fi
E1_LAUNCH="$E1_PARENT/launchers/E1_SLPPO_CVRP50_15_A6000_S3009_V1"
mkdir -p "$E1_LAUNCH"
if [[ "$E1_MODE" == --execute ]]; then
  nohup bash "$E1_ROOT/scripts/run_e1_slppo_a6000.sh" --worker >> "$E1_LAUNCH/supervisor.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$E1_LAUNCH/last_launcher.pid"
  printf 'Background preparation PID=%s\nLog: %s/supervisor.log\nPer-scale logs: %s/cus{50,15}.log\n' "$!" "$E1_LAUNCH" "$E1_LAUNCH"
  exit 0
fi
exec 9>"$E1_LAUNCH/prepare.lock"
flock -n 9 || { printf 'A6000 SL-PPO preparation already running\n' >&2; exit 1; }
source "$E1_ROOT/scripts/e1_small_scale_hardware.sh"
e1_slppo_scale() {
  local scale="$1" gpu="$E1_GPU50"
  [[ "$scale" != 15 ]] || gpu="$E1_GPU15"
  local campaign="$E1_PARENT/E1_CVRP${scale}_SLPPO_S3009_A6000_V1"
  local audit="$E1_ASSETS/Cus${scale}"
  local ppo_key="E1_CVRP${scale}_SLPPO_PPO_CHUNK" expert_key="E1_CVRP${scale}_SLPPO_EXPERT_CHUNK"
  local ppo="${!ppo_key:-${E1_SLPPO_PPO_CHUNK:-32}}" expert="${!expert_key:-${E1_EXPERT_CHUNK:-128}}"
  local cmd=(bash "$E1_ROOT/scripts/run_e1.sh") retry=()
  # Use isolated names; never relaunch or overwrite existing CVRP/native campaigns.
  if e1_small_existing "$campaign" slppo; then return 0; fi
  if [[ ! -f "$audit/assets_audit.json" ]]; then
    mkdir -p "$E1_ASSETS"
    (
      flock -x 8
      if [[ ! -f "$audit/assets_audit.json" ]]; then
        "${cmd[@]}" preflight --customers "$scale" --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}" --output "$audit"
      fi
    ) 8>"$E1_ASSETS/Cus${scale}.audit.lock"
  fi
  if [[ ! -f "$campaign/manifest.json" ]]; then
    "${cmd[@]}" prepare --customers "$scale" --audit "$audit/assets_audit.json" --campaign "$campaign" \
      --seed 3009 --epochs 1500 --method-ppo-chunk "slppo=$ppo" --expert-chunk "$expert"
  fi
  [[ ! -d "$campaign/runs/slppo/smoke" ]] || retry=(--retry)
  "${cmd[@]}" smoke --campaign "$campaign" --method slppo "${retry[@]}"
  # Explicit method is essential: server a6000 alone would select RRNCO/RADAR.
  "${cmd[@]}" launch --campaign "$campaign" --customers "$scale" --server a6000 \
    --method slppo --gpus "$gpu" --execute
  printf 'Queued CVRP%s SL-PPO; status: %s/scheduler_a6000/status.json\n' "$scale" "$campaign"
}
E1_CHILDREN=()
for E1_SCALE in $E1_SCALES; do
  e1_slppo_scale "$E1_SCALE" > "$E1_LAUNCH/cus${E1_SCALE}.log" 2>&1 &
  E1_CHILDREN+=("$!")
done
E1_FAILED=0
for E1_CHILD in "${E1_CHILDREN[@]}"; do
  if ! wait "$E1_CHILD"; then E1_FAILED=1; fi
done
if [[ "$E1_FAILED" == 1 ]]; then
  printf 'At least one scale failed preparation; inspect per-scale logs. The other scale may already be training.\n' >&2
else
  printf 'Both requested scales processed. Training supervisors run independently in the background.\n'
fi
exit "$E1_FAILED"
