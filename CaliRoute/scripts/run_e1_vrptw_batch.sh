#!/usr/bin/env bash
# Detach all preparation and subsequent training/test. Repeated invocations do not restart runs.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_PROFILE="${1:?Usage: run_e1_vrptw_batch.sh a6000|2080ti_b|2080ti_a|2080ti_3 [--execute]}"
shift
case "$E1_PROFILE" in a6000|2080ti_b|2080ti_a|2080ti_3) ;; *) exit 2;; esac
E1_MODE="${1:-}"
[[ $# -le 1 && ( -z "$E1_MODE" || "$E1_MODE" == --execute ) ]] || exit 2
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
[[ -x "$E1_PYTHON" ]] || E1_PYTHON="$(command -v python)"
E1_OUTPUT="${E1_VRPTW_ROOT:-$E1_ROOT/results/e1}"
E1_ARGS=("$E1_ROOT/scripts/run_e1_vrptw_batch.py" "$E1_PROFILE" --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}" --output "$E1_OUTPUT")
if [[ -z "$E1_MODE" ]]; then exec "$E1_PYTHON" "${E1_ARGS[@]}"; fi
E1_LAUNCH="$E1_OUTPUT/launchers/VRPTW_${E1_PROFILE}_V1"
mkdir -p "$E1_LAUNCH"
nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$E1_PYTHON" -u "${E1_ARGS[@]}" --execute >> "$E1_LAUNCH/supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$E1_LAUNCH/last_launcher.pid"
printf 'Background preparation PID=%s\nLogs: %s/cus*.log\nTraining and best-checkpoint test run automatically.\n' "$!" "$E1_LAUNCH"
