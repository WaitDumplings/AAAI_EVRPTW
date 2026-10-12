#!/usr/bin/env bash
# Cus50 only. Reuse batch-two campaign names to avoid duplicate experiments.
set -euo pipefail
E1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_SERVER="${1:?Usage: run_e1_cvrp50.sh 2080ti_a|2080ti_b|2080ti_3|a6000 [--execute]}"
shift
E1_EXECUTE=false
if [[ $# == 1 && "$1" == --execute ]]; then E1_EXECUTE=true
elif [[ $# != 0 ]]; then printf 'Only --execute is accepted.\n' >&2; exit 2; fi
case "$E1_SERVER" in
  2080ti_a)
    printf 'Local A: corrected Cus50 PPO base already scheduled on GPU2; RDI+AGDA training/test completed.\n'
    printf 'Base: %s/results/e1/E1_CVRP50_BASE_NO_ROAD_S3009_E1500_20261012_R1/scheduler_2080ti_a/status.json\n' "$E1_ROOT"
    printf 'Enhanced: %s/results/e1/E1_BATCH2_CVRP50_S3009_2080ti_a/runs/ppo_rdi_agda/test/manifest.json\n' "$E1_ROOT"
    printf 'No duplicate jobs launched by this profile.\n'; exit 0;;
  2080ti_b) E1_METHODS=(awbc dapg);;
  2080ti_3) E1_METHODS=(slppo);;
  a6000) E1_METHODS=(rrnco radar);;
  2080ti_c) printf 'Four-card C is reserved. Use 2080ti_3 on the three-card SL-PPO server.\n'; exit 0;;
  *) printf 'Unknown server profile\n' >&2; exit 2;;
esac
E1_PYTHON="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
if [[ ! -x "$E1_PYTHON" ]]; then E1_PYTHON="$(command -v python)"; fi
E1_PARENT="${E1_BATCH2_ROOT:-$E1_ROOT/results/e1}"
E1_CAMPAIGN="${E1_CVRP50_CAMPAIGN:-$E1_PARENT/E1_BATCH2_CVRP50_S3009_${E1_SERVER}}"
E1_AUDIT="${E1_BATCH2_ASSETS_ROOT:-$E1_PARENT/batch2_assets}/Cus50"
E1_CMD=(bash "$E1_ROOT/scripts/run_e1.sh")
E1_GPU_ARGS=()
if [[ "$E1_SERVER" == 2080ti_3 ]]; then
 E1_SL_GPU="${E1_CVRP50_SLPPO_GPU:-2}"
 if [[ ! "$E1_SL_GPU" =~ ^[0-9]+$ ]]; then printf 'SL-PPO GPU must be a nonnegative index\n' >&2; exit 2; fi
 E1_GPU_ARGS=(--gpus "$E1_SL_GPU")
fi
printf 'CVRP Cus50 only: %s; seed3009; scratch; 1500 online loops; one GPU per method\n' "${E1_METHODS[*]}"
for E1_INDEX in "${!E1_METHODS[@]}"; do
 E1_CARD="$E1_INDEX"
 if [[ "$E1_SERVER" == 2080ti_3 ]]; then E1_CARD="$E1_SL_GPU"; fi
 printf 'GPU%s: %s\n' "$E1_CARD" "${E1_METHODS[$E1_INDEX]}"
done
printf 'Campaign: %s\nStatus: %s/scheduler_%s/status.json\n' "$E1_CAMPAIGN" "$E1_CAMPAIGN" "$E1_SERVER"
if ! "$E1_EXECUTE"; then printf 'Preview only. Add --execute to audit, smoke and queue. No Cus15 jobs.\n'; exit 0; fi
# Verify existing frozen files and refuse duplicate/partial formal writes. A live
# or finished run needs no relaunch. Failed runs require explicit frozen resume.
if [[ -f "$E1_CAMPAIGN/manifest.json" ]]; then
 E1_DECISION=$(PYTHONPATH="$E1_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$E1_PYTHON" - "$E1_CAMPAIGN" "${E1_METHODS[@]}" <<'PY'
import sys,json
from pathlib import Path
from e1.runner import verify_campaign
from e1.lifecycle import assert_can_start
p=Path(sys.argv[1]).resolve();m=verify_campaign(p)
if m['schema']!='aaai_e1_cvrp50_v1' or m['training_seed']!=3009:
    raise ValueError('Existing campaign has a different scale/seed')
existing=[]
for method in sys.argv[2:]:
    try:assert_can_start(m,method)
    except FileExistsError:existing.append(method)
    if (Path(m['methods'][method]['run_dir'])/'hardware_preflight').exists() and method not in existing:
        existing.append(method)
print('existing:'+','.join(existing) if existing else 'fresh')
PY
 )
 if [[ "$E1_DECISION" != fresh ]]; then
   printf 'Existing runs/preflights detected (%s). Preserved; no duplicate launch.\n' "$E1_DECISION"
   printf 'Inspect status and train.log/native_status.json; for interruption use explicit run_e1.sh resume.\n'
   exit 0
 fi
fi
if [[ "$E1_SERVER" == a6000 ]]; then "${E1_CMD[@]}" native-status; fi
if [[ ! -f "$E1_AUDIT/assets_audit.json" ]]; then
 "${E1_CMD[@]}" preflight --customers 50 --data-root "${E1_DATA_ROOT:-$E1_ROOT/../AAAI_Dataset}" --output "$E1_AUDIT"
fi
if [[ ! -f "$E1_CAMPAIGN/manifest.json" ]]; then
 "${E1_CMD[@]}" prepare --customers 50 --audit "$E1_AUDIT/assets_audit.json" --campaign "$E1_CAMPAIGN"
fi
for E1_METHOD in "${E1_METHODS[@]}"; do
 E1_RESULT="$E1_CAMPAIGN/runs/$E1_METHOD/smoke/smoke_result.json"
 if [[ -f "$E1_RESULT" ]] && "$E1_PYTHON" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("state")!="passed")' "$E1_RESULT"; then
   printf 'Existing smoke passed: %s\n' "$E1_METHOD"; continue
 fi
 E1_RETRY=()
 if [[ -d "$E1_CAMPAIGN/runs/$E1_METHOD/smoke" ]]; then E1_RETRY=(--retry); fi
 "${E1_CMD[@]}" smoke --campaign "$E1_CAMPAIGN" --method "$E1_METHOD" "${E1_RETRY[@]}"
done
"${E1_CMD[@]}" launch --campaign "$E1_CAMPAIGN" --customers 50 --server "$E1_SERVER" "${E1_GPU_ARGS[@]}" --execute
