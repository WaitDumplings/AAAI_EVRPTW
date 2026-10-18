#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${E1_RADAR_PYTHON:-/data/Maojie/AAAI/.venvs/e1-radar/bin/python}"
campaign="${E2_RADAR_CAMPAIGN:-/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2_RADAR_CVRP100_S3009_A6000_V1}"
baseline="${E2_RADAR_BASELINE:-/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1/runs/radar}"
args=("$python_bin" -u "$root/scripts/run_e2_radar_variants_a6000.py" --campaign "$campaign" --baseline-run "$baseline" "$@")
if [[ " $* " == *" --execute "* ]]; then
  mkdir -p "$campaign"
  nohup setsid "${args[@]}" >> "$campaign/launch.log" 2>&1 < /dev/null &
  echo "launcher PID=$! log=$campaign/launch.log"
else
  "${args[@]}"
fi
