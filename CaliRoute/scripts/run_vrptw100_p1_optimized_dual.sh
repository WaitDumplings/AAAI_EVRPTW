#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CODE_ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi
arguments=(--variant optimized --gpus "${GPUS:-2,3}")
if [[ -n "${DATA_ROOT:-}" ]]; then arguments+=(--data-root "${DATA_ROOT}"); fi
if [[ -n "${RUN_ID:-}" ]]; then arguments+=(--run-id "${RUN_ID}"); fi
launch=true
for argument in "$@"; do
  case "${argument}" in
    --prepare-only|--launch|--supervise|-h|--help) launch=false ;;
  esac
done
if ${launch}; then arguments+=(--launch); fi
exec "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_original_p1_dual.py" "${arguments[@]}" "$@"
