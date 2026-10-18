#!/usr/bin/env bash
# Portable E2 queue preparation. --execute never falls back to E1 training.
set -euo pipefail
E2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E2_SERVER="${1:?Usage: run_e2_batch.sh 2080ti_a|a6000|2080ti_3 [--prepare|--check|--execute]}"
shift
case "$E2_SERVER" in 2080ti_a|a6000|2080ti_3) ;; *) echo 'Unknown E2 server' >&2; exit 2;; esac
E2_MODE="${1:-}"
[[ $# -le 1 && ( -z "$E2_MODE" || "$E2_MODE" == --prepare || "$E2_MODE" == --check || "$E2_MODE" == --execute ) ]] || { echo 'Use --prepare, --check or --execute' >&2; exit 2; }
E2_PYTHON="${E2_PYTHON:-$E2_ROOT/../.venv/bin/python}"
[[ -x "$E2_PYTHON" ]] || E2_PYTHON="$(command -v python)"
E2_OUTPUT="${E2_OUTPUT:-$E2_ROOT/results/e2}"
E2_ARGS=("$E2_ROOT/scripts/run_e2_batch.py" "$E2_SERVER" --data-root "${E2_DATA_ROOT:-$E2_ROOT/../AAAI_Dataset}" --output "$E2_OUTPUT" --baseline-policy "${E2_BASELINE_POLICY:-pending}")
if [[ "$E2_MODE" == --prepare ]]; then
    E2_LOG_DIR="$E2_OUTPUT/launchers/$E2_SERVER"
    mkdir -p "$E2_LOG_DIR"
    nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$E2_PYTHON" -u "${E2_ARGS[@]}" --prepare >> "$E2_LOG_DIR/preparation.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$E2_LOG_DIR/last_preparation.pid"
    printf 'Background preparation PID=%s\nLog: %s/preparation.log\nThis prepares the queue only; no model training has started.\n' "$!" "$E2_LOG_DIR"
else
    [[ -z "$E2_MODE" ]] || E2_ARGS+=("$E2_MODE")
    exec "$E2_PYTHON" "${E2_ARGS[@]}"
fi
