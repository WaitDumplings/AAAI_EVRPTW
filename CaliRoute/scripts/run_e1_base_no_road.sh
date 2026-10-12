#!/usr/bin/env bash
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_SCALE="${1:-100}"
case "$E1_SCALE" in 15|50|100) ;; *) printf 'Scale must be 15, 50 or 100\n' >&2; exit 2;; esac
if [[ $# -gt 0 ]]; then shift; fi
source "$E1_ROOT/scripts/e1_small_scale_hardware.sh"
e1_small_scale_hardware "$E1_SCALE"
E1_AUDIT="${E1_BASE_AUDIT:-$E1_ROOT/results/e1/assets_cvrp${E1_SCALE}_20261010}"
E1_CAMPAIGN="${E1_BASE_CAMPAIGN:-$E1_ROOT/results/e1/E1_CVRP${E1_SCALE}_BASE_NO_ROAD_S3009_E1500_20261012_R2}"
printf 'Corrected CVRP%s PPO base, scratch, E1500, seed3009: %s\n' "$E1_SCALE" "$E1_CAMPAIGN"
if [[ $# == 0 ]]; then printf 'Preview only; add --execute.\n'; exit 0; fi
if [[ $# != 1 || "$1" != --execute ]]; then printf 'Only --execute is accepted\n' >&2; exit 2; fi
E1_CMD=(bash "$E1_ROOT/scripts/run_e1.sh")
if [[ ! -f "$E1_AUDIT/assets_audit.json" ]]; then
 "${E1_CMD[@]}" preflight --customers "$E1_SCALE" --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}" --output "$E1_AUDIT"
fi
"${E1_CMD[@]}" prepare --customers "$E1_SCALE" --audit "$E1_AUDIT/assets_audit.json" --campaign "$E1_CAMPAIGN" "${E1_HARDWARE_ARGS[@]}"
"${E1_CMD[@]}" smoke --campaign "$E1_CAMPAIGN" --method ppo_base
"${E1_CMD[@]}" launch --campaign "$E1_CAMPAIGN" --customers "$E1_SCALE" --server 2080ti_a --method ppo_base --execute
