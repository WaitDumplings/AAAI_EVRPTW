#!/usr/bin/env bash
# Prepares only unless --launch is explicit. Does not define or start E1.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CODE_ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN=python3; fi
exec "${PYTHON_BIN}" -B "${SCRIPT_DIR}/run_aaai_candidate.py" "$@"
