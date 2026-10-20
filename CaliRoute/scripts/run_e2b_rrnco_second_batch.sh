#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${E1_RRNCO_PYTHON:-/data/Maojie/AAAI/.venvs/e1-rrnco/bin/python}"
campaign="${E2B_CAMPAIGN:-/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009}"
args=("$python_bin" -u "$root/scripts/run_e2b_rrnco_second_batch.py" --campaign "$campaign" "$@")
if [[ " $* " == *" --execute "* ]]; then
  mkdir -p "$campaign"
  nohup setsid "${args[@]}" >> "$campaign/second_batch_launch.log" 2>&1 < /dev/null &
  echo "launcher PID=$! log=$campaign/second_batch_launch.log"
else
  "${args[@]}"
fi
