#!/usr/bin/env bash
# Three-card host: one distributed SL-PPO run on GPU0+1; GPU2 stays available.
# No arguments previews the plan. --execute detaches preparation/train/best test.
set -euo pipefail
E1_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$E1_SCRIPT_DIR/run_e1_vrptw_batch.sh" 2080ti_3 "$@"
