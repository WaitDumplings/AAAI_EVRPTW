#!/usr/bin/env bash
set -euo pipefail
CALIR_ROUTE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_PYTHON="${E1_PYTHON:-$CALIR_ROUTE_ROOT/../.venv/bin/python}"
if [[ ! -x "$E1_PYTHON" ]]; then E1_PYTHON="$(command -v python)"; fi
exec "$E1_PYTHON" "$CALIR_ROUTE_ROOT/scripts/run_e1.py" "$@"
