#!/usr/bin/env bash
# Prepare/audit/CPU-smoke first; the detached watcher starts training only later.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
E1_CAMPAIGN="${E1_VRPTW_CONTROLLED_CAMPAIGN:-$E1_ROOT/results/e1/E1_VRPTW100_PPO_COMPARE_S3009_E1500_2080TI_A_20261013_R1}"
E1_ARGS=(--campaign "$E1_CAMPAIGN")
if [[ ! -f "$E1_CAMPAIGN/manifest.json" ]]; then
  E1_ARGS+=(--prepare --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}"
    --after "$E1_ROOT/results/e1/E1_CVRP100_BASE_NO_ROAD_S3009_E1500_20261012_R2"
    --after "$E1_ROOT/results/e1/E1_CVRP50_BASE_NO_ROAD_S3009_E1500_20261012_R1"
    --after "$E1_ROOT/results/e1/E1_CVRP15_BASE_NO_ROAD_S3009_E1500_20261012_R1")
fi
cd "$E1_ROOT"
exec "$E1_PYTHON" -m e1.vrptw_controlled "${E1_ARGS[@]}" --execute
