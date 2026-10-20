#!/usr/bin/env bash
# Five CVRP100 ablations, one true two-rank training job per GPU pair.
set -euo pipefail
E3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E3_SERVER="${1:?Usage: run_e3_initial.sh 2080ti_a|2080ti_3|2080ti_b [--execute|--resume]}"
shift
case "$E3_SERVER" in
 2080ti_a) printf 'GPU0,1: A0_no_bias; GPU2,3: AE_euclidean_bias\n';;
 2080ti_3) printf 'GPU0,1: AR_road_bias; GPU2 unused\n';;
 2080ti_b) printf 'GPU0,1: B0_static_candidates; GPU2,3: B1_feature_matched_logit\n';;
 *) printf 'Unknown profile\n' >&2; exit 2;;
esac
E3_PYTHON="${E3_PYTHON:-$E3_ROOT/../.venv/bin/python}"
if [[ ! -x "$E3_PYTHON" ]]; then E3_PYTHON="$(command -v python)"; fi
E3_RUN="${E3_RUN:-$E3_ROOT/results/e3/E3_CVRP100_S3009_E1500_${E3_SERVER}_V1}"
printf 'seed3009; scratch; 1500 online loops; global batch64; n_traj50; PPO passes4; FP32\nRun: %s\n' "$E3_RUN"
if [[ $# == 0 ]]; then printf 'Preview only. Add --execute to prepare, preflight and train; best-test is automatic.\n'; exit 0; fi
E3_EXTRA=()
if [[ $# == 1 && "$1" == --resume ]]; then E3_EXTRA=(--resume)
elif [[ $# != 1 || "$1" != --execute ]]; then printf 'Only --execute or --resume accepted\n' >&2; exit 2
elif [[ -e "$E3_RUN" ]]; then printf 'Existing campaign preserved. Inspect it before using --resume.\n' >&2; exit 2; fi
mkdir -p "$(dirname "$E3_RUN")"
# Log outside the run directory: preparation refuses to overwrite existing runs.
nohup setsid env PYTHONPATH="$E3_ROOT${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 "$E3_PYTHON" -u -m e3.runner --server "$E3_SERVER" --output "$E3_RUN" \
 --data-root "${E3_DATA_ROOT:-$E3_ROOT/../AAAI_Dataset}" "${E3_EXTRA[@]}" \
 > "$E3_RUN.supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$E3_RUN.supervisor.pid"
printf 'Supervisor PID=%s\nLog=%s.supervisor.log\nStatus=%s/status.json\n' "$!" "$E3_RUN" "$E3_RUN"
