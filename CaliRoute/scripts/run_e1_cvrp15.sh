#!/usr/bin/env bash
# Cus15 only, same frozen-budget hardware profile as the batch-two launcher.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export E1_BATCH2_SCALES=15
exec bash "$E1_ROOT/scripts/run_e1_batch2.sh" "$@"
