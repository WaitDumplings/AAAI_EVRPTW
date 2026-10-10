#!/usr/bin/env bash
# An explicit --execute performs asset audit + selected smokes, then queues this
# server's allocation. With no --execute it only prints the proposed commands.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_SERVER="${1:?Usage: run_e1_server.sh 2080ti_a|2080ti_b|2080ti_c|a6000|2080ti_3 [--execute]}"
shift
E1_EXECUTE=false
if [[ "${1:-}" == --execute && $# == 1 ]]; then E1_EXECUTE=true
elif [[ $# != 0 ]]; then printf 'Unsupported arguments; only --execute is accepted.\n' >&2; exit 2; fi
case "$E1_SERVER" in
  2080ti_a) E1_METHODS=(ppo_base ppo_rdi_agda);;
  2080ti_b) E1_METHODS=(awbc dapg);;
  2080ti_c) E1_METHODS=(slppo);;
  a6000) E1_METHODS=(rrnco radar);;
  2080ti_3) printf 'Reserved server; no E1 training allocated.\n';exit 0;;
  *) printf 'Unknown server profile\n' >&2;exit 2;;
esac
E1_AUDIT="${E1_AUDIT:-$E1_ROOT/results/e1/assets_cvrp100}"
E1_CAMPAIGN="${E1_CAMPAIGN:-$E1_ROOT/results/e1/E1_CVRP100_S3009_V1_$E1_SERVER}"
E1_DATA_ROOT="${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}"
if ! "$E1_EXECUTE"; then
  printf 'Plan: %s; methods: %s; campaign: %s\n' "$E1_SERVER" "${E1_METHODS[*]}" "$E1_CAMPAIGN"
  printf 'Use --execute to audit, prepare, smoke, and queue after GPUs become idle.\n'
  exit 0
fi
if [[ "$E1_SERVER" == a6000 ]]; then
  bash "$E1_ROOT/scripts/run_e1.sh" native-status
fi
if [[ ! -f "$E1_AUDIT/assets_audit.json" ]]; then
  bash "$E1_ROOT/scripts/run_e1.sh" preflight --data-root "$E1_DATA_ROOT" --output "$E1_AUDIT"
fi
if [[ ! -f "$E1_CAMPAIGN/manifest.json" ]]; then
  bash "$E1_ROOT/scripts/run_e1.sh" prepare --audit "$E1_AUDIT/assets_audit.json" --campaign "$E1_CAMPAIGN"
fi
for E1_METHOD in "${E1_METHODS[@]}"; do
  if [[ -d "$E1_CAMPAIGN/runs/$E1_METHOD/smoke" ]]; then
    bash "$E1_ROOT/scripts/run_e1.sh" smoke --campaign "$E1_CAMPAIGN" --method "$E1_METHOD" --retry
  else
    bash "$E1_ROOT/scripts/run_e1.sh" smoke --campaign "$E1_CAMPAIGN" --method "$E1_METHOD"
  fi
done
bash "$E1_ROOT/scripts/run_e1.sh" launch --campaign "$E1_CAMPAIGN" --server "$E1_SERVER" --execute
