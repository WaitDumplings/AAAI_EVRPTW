#!/usr/bin/env bash
# One two-GPU arm; global VRPTW batch40 / EVRPTW batch32, five PPO passes.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_graph_reproduction_2080ti.sh vrptw|evrptw graph|current [--gpus 0,1] [--prepare-only] [options]' >&2
  exit 2
fi
task="$1"; encoder="$2"; shift 2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CODE_ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN=python3; fi
arguments=(--task "${task}" --encoder-variant "${encoder}" --gpus "${GPUS:-0,1}")
if [[ -n "${DATA_ROOT:-}" ]]; then arguments+=(--data-root "${DATA_ROOT}"); fi
launch=true
for argument in "$@"; do
  case "${argument}" in --prepare-only|--launch|-h|--help) launch=false ;; esac
done
if ${launch}; then arguments+=(--launch); fi
exec "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_graph_reproduction_2080ti.py" "${arguments[@]}" "$@"
