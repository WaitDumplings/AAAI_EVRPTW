#!/usr/bin/env bash
# Physical deployment alias. Keep CaliRoute's frozen E1 source unchanged.
set -euo pipefail
E1_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then
  printf 'Usage: bash scripts/run_e1_3x2080ti.sh [--execute]\n' >&2
  exit 2
fi
export E1_CAMPAIGN="${E1_CAMPAIGN:-$E1_REPO_ROOT/CaliRoute/results/e1/E1_CVRP100_S3009_V1_2080TI_3}"
printf 'Physical server: 3x2080Ti; SL-PPO on GPUs 0,1; GPU 2 reserved.\n'
printf 'Frozen task profile: 2080ti_c (a task label, not a four-GPU requirement).\n'
exec bash "$E1_REPO_ROOT/CaliRoute/scripts/run_e1_server.sh" 2080ti_c "$@"
