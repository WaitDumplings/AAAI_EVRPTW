#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${E1_RRNCO_PYTHON:-/data/Maojie/AAAI/.venvs/e1-rrnco/bin/python}"
campaign="${E2_RRNCO_CAMPAIGN:-/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2_RRNCO_VRPTW100_S3009_A6000_V1}"
baseline="${E2_RRNCO_BASELINE:-/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2/runs/rrnco}"
args=("$python_bin" -u "$root/scripts/run_e2_rrnco_variants_a6000.py" --campaign "$campaign" --baseline-run "$baseline" "$@")
if [[ " $* " == *" --execute "* ]]; then
  mkdir -p "$campaign"
  nohup setsid "${args[@]}" >> "$campaign/launch.log" 2>&1 < /dev/null &
  echo "launcher PID=$! log=$campaign/launch.log"
else
  "${args[@]}"
fi
