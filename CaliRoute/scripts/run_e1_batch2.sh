#!/usr/bin/env bash
# Batch two: CVRP50 + CVRP15. No long jobs without --execute.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_SERVER="${1:?Usage: run_e1_batch2.sh 2080ti_a|2080ti_b|2080ti_3|a6000 [--execute]}"
shift
E1_EXECUTE=false
if [[ $# == 1 && "$1" == --execute ]]; then E1_EXECUTE=true
elif [[ $# != 0 ]]; then printf 'Only --execute is accepted.\n' >&2; exit 2; fi
case "$E1_SERVER" in
  2080ti_a) E1_METHODS=(ppo_base ppo_rdi_agda);;
  2080ti_b) E1_METHODS=(awbc dapg);;
  2080ti_3) E1_METHODS=(slppo);;
  a6000) E1_METHODS=(rrnco radar);;
  2080ti_c) printf 'Four-card server C is reserved in batch two.\n'; exit 0;;
  *) printf 'Unknown server profile\n' >&2; exit 2;;
esac
E1_BATCH2_ROOT="${E1_BATCH2_ROOT:-$E1_ROOT/results/e1}"
E1_BATCH2_ASSETS_ROOT="${E1_BATCH2_ASSETS_ROOT:-$E1_BATCH2_ROOT/batch2_assets}"
E1_DATA_ROOT="${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}"
E1_CMD=(bash "$E1_ROOT/scripts/run_e1.sh")
E1_PREVIOUS=""
for E1_SCALE in 50 15; do
  E1_CAMPAIGN="$E1_BATCH2_ROOT/E1_BATCH2_CVRP${E1_SCALE}_S3009_${E1_SERVER}"
  E1_AUDIT="$E1_BATCH2_ASSETS_ROOT/Cus${E1_SCALE}"
  if [[ "$E1_SERVER" == a6000 || "$E1_SCALE" == 50 ]]; then E1_OFFSET=0
  elif [[ "$E1_SERVER" == 2080ti_3 ]]; then E1_OFFSET=1
  else E1_OFFSET=2; fi
  printf 'CVRP%s: methods=%s; first GPU=%s; campaign=%s\n' "$E1_SCALE" "${E1_METHODS[*]}" "$E1_OFFSET" "$E1_CAMPAIGN"
  if [[ "$E1_SERVER" == a6000 && "$E1_SCALE" == 15 ]]; then
    printf 'Wait for BOTH CVRP50 native methods to complete successfully before acquiring GPUs 0,1.\n'
  fi
  if ! "$E1_EXECUTE"; then continue; fi
  if [[ "$E1_SERVER" == a6000 && "$E1_SCALE" == 50 ]]; then "${E1_CMD[@]}" native-status; fi
  if [[ ! -f "$E1_AUDIT/assets_audit.json" ]]; then
    "${E1_CMD[@]}" preflight --customers "$E1_SCALE" --data-root "$E1_DATA_ROOT" --output "$E1_AUDIT"
  fi
  if [[ ! -f "$E1_CAMPAIGN/manifest.json" ]]; then
    "${E1_CMD[@]}" prepare --customers "$E1_SCALE" --audit "$E1_AUDIT/assets_audit.json" --campaign "$E1_CAMPAIGN"
  fi
  for E1_METHOD in "${E1_METHODS[@]}"; do
    E1_RETRY=()
    if [[ -d "$E1_CAMPAIGN/runs/$E1_METHOD/smoke" ]]; then E1_RETRY=(--retry); fi
    "${E1_CMD[@]}" smoke --campaign "$E1_CAMPAIGN" --method "$E1_METHOD" "${E1_RETRY[@]}"
  done
  E1_DEPENDENCY=()
  if [[ "$E1_SERVER" == a6000 && "$E1_SCALE" == 15 ]]; then E1_DEPENDENCY=(--after-campaign "$E1_PREVIOUS"); fi
  "${E1_CMD[@]}" launch --campaign "$E1_CAMPAIGN" --customers "$E1_SCALE" --server "$E1_SERVER" "${E1_DEPENDENCY[@]}" --execute
  E1_PREVIOUS="$E1_CAMPAIGN"
done
if ! "$E1_EXECUTE"; then printf 'Dry run only. Add --execute to audit, smoke and queue; GPU preflight precedes fresh formal training.\n'; fi
