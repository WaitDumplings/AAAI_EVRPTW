#!/usr/bin/env bash
# E2 server profile: 2080ti_a. Preview by default; --prepare runs detached preparation.
set -euo pipefail
E2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$E2_SCRIPT_DIR/run_e2_batch.sh" 2080ti_a "$@"
