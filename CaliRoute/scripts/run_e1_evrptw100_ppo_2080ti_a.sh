#!/usr/bin/env bash
# EVRPTW Cus100: GPU0+1 PPO base, GPU2+3 PPO RDI+AGDA; one DDP run each.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_MODE="${1:-}"
[[ $# -le 1 && ( -z "$E1_MODE" || "$E1_MODE" == --execute ) ]] || { echo "Usage: $0 [--execute]" >&2; exit 2; }
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
[[ -x "$E1_PYTHON" ]] || E1_PYTHON="$(command -v python)"
E1_OUTPUT="${E1_EVRPTW_ROOT:-$E1_ROOT/results/e1}"
E1_ARGS=("$E1_ROOT/scripts/run_e1_vrptw_batch.py" 2080ti_a_ev100 --problem evrptw --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}" --output "$E1_OUTPUT")
if [[ -z "$E1_MODE" ]]; then exec "$E1_PYTHON" "${E1_ARGS[@]}"; fi
E1_LAUNCH="$E1_OUTPUT/launchers/EVRPTW_2080ti_a_ev100_V1"
mkdir -p "$E1_LAUNCH"
nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$E1_PYTHON" -u "${E1_ARGS[@]}" --execute >> "$E1_LAUNCH/supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$E1_LAUNCH/last_launcher.pid"
printf 'Background preparation PID=%s\nLogs: %s/cus*.log\nTraining and best-checkpoint test run automatically.\n' "$!" "$E1_LAUNCH"
