#!/usr/bin/env bash
# One task, two synchronized GPUs, random initialization; no local GPU tuning.
set -euo pipefail
usage() {
  echo "Usage: bash scripts/run_graph_rdi100_dual.sh vrptw|evrptw [--gpus 0,1] [--epochs 1500] [--prepare-only] [launcher options]"
}
if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
case "$1" in
  vrptw|evrptw) task="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CODE_ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi
arguments=(--task "${task}" --variant optimized --encoder-variant graph
  --seed "${SEED:-3011}" --batch-per-gpu 32 --chunk-size 8 --expert-chunk-size 64
  --gpus "${GPUS:-0,1}")
if [[ -n "${DATA_ROOT:-}" ]]; then arguments+=(--data-root "${DATA_ROOT}"); fi
if [[ -n "${RUN_ID:-}" ]]; then arguments+=(--run-id "${RUN_ID}"); fi
launch=true
for argument in "$@"; do
  case "${argument}" in
    --prepare-only|--launch|--supervise|-h|--help) launch=false ;;
  esac
done
if ${launch}; then arguments+=(--launch); fi
exec "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_evrptw_dual_scratch.py" "${arguments[@]}" "$@"
