#!/usr/bin/env bash
# Entire preparation, smoke, GPU preflight and training survives terminal closure.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
[[ -x "$E1_PYTHON" ]] || E1_PYTHON="$(command -v python)"
E1_CAMPAIGN="${E1_VRPTW100_CAMPAIGN:-$E1_ROOT/results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V1}"
E1_ARGS=(--campaign "$E1_CAMPAIGN" --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}")
case "${1:-}" in
 '') exec "$E1_PYTHON" "$E1_ROOT/scripts/run_e1_vrptw_native.py" "${E1_ARGS[@]}";;
 --execute) [[ $# == 1 ]] || exit 2;;
 --resume) [[ $# == 1 ]] || exit 2; E1_ARGS+=(--resume);;
 *) printf 'Usage: run_e1_vrptw100_a6000.sh [--execute|--resume]\n' >&2; exit 2;;
esac
mkdir -p -- "$(dirname -- "$E1_CAMPAIGN")"
nohup "$E1_PYTHON" "$E1_ROOT/scripts/run_e1_vrptw_native.py" "${E1_ARGS[@]}" --execute >> "${E1_CAMPAIGN}.launch.log" 2>&1 < /dev/null &
printf 'Supervisor PID=%s\nGPU0 RRNCO / GPU1 RADAR; VRPTW Cus100, scratch seed3009, 96000 instance exposures (1500 x64), K50\nLog: %s.launch.log\nStatus: %s/status.json\n' "$!" "$E1_CAMPAIGN" "$E1_CAMPAIGN"
