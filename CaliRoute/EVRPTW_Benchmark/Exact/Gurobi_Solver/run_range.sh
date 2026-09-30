#!/usr/bin/env bash
set -Eeuo pipefail

PROBLEM="${1:?Expected CVRP, VRPTW, or EVRPTW}"
shift
case "$PROBLEM" in CVRP|VRPTW|EVRPTW) ;; *) echo "Unknown problem: $PROBLEM" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIROUTE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$SCRIPT_DIR/$PROBLEM/run_range.py"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: run_gurobi_range.sh --cus 15 --split val --start 0 --end 100 [options]
  -w, --workers N       Parallel workers (default: 24)
  -t, --threads N       Threads per worker (default: 1)
  -s, --start N         Inclusive instance index (default: 0)
  -e, --end N           Exclusive instance index (default: 100)
  -c, --cus N|CusN      Customer scale (default: Cus15)
  --split NAME         train, val, or test (legacy eval also accepted)
  --dataset-path PATH  Explicit input bundle or split/scale directory
  --dataset-root PATH  Problem root containing split/CusN directories
  --output-path PATH   Override results/gurobi/<problem>/<split>/CusN
  --time-limit N       Solve time in seconds (default: 7200)
  --checkpoints SECS   Comma-separated checkpoint times
  --mip-gap X          Relative MIP gap (default: 0)
  --cs-copies N        EVRPTW charging-station copies (shell default: 4)
  --no-skip-completed  Recompute existing result rows
  --python PATH        Python executable (default: PYTHON_BIN or python)
  --conda-env NAME     Optionally use conda run in a named environment
  --log-dir PATH       Log directory (default: results/gurobi/<problem>/logs)
  --log-file PATH      Explicit log filename
  --detach             Run in the background
  --dry-run            Print resolved paths and runner arguments without solving
USAGE
}
die() { echo "ERROR: $*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "$1 requires a value"; }

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-100}"
WORKERS="${WORKERS:-24}"
THREADS="${THREADS:-1}"
CUS="${CUS:-${SCALE:-Cus15}}"
DEFAULT_SPLIT=train
[[ "$PROBLEM" == CVRP ]] && DEFAULT_SPLIT=val
SPLIT="${SPLIT:-$DEFAULT_SPLIT}"
CS_COPIES="${CS_COPIES:-4}"
TIME_LIMIT_S="${TIME_LIMIT_S:-7200}"
MIP_GAP="${MIP_GAP:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
LOG_DIR="${LOG_DIR:-$CALIROUTE_ROOT/results/gurobi/${PROBLEM,,}/logs}"
LOG_FILE="${LOG_FILE:-}"
DETACH="${DETACH:-0}"
DRY_RUN=0
EXTRA_ARGS=()
[[ -n "${DATASET_PATH:-}" ]] && EXTRA_ARGS+=(--dataset_path "$DATASET_PATH")
[[ -n "${DATASET_ROOT:-}" ]] && EXTRA_ARGS+=(--dataset_root "$DATASET_ROOT")
[[ -n "${OUTPUT_PATH:-}" ]] && EXTRA_ARGS+=(--output_path "$OUTPUT_PATH")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workers) need_value "$@"; WORKERS="$2"; shift 2 ;;
    -t|--threads) need_value "$@"; THREADS="$2"; shift 2 ;;
    -s|--start|--start-index) need_value "$@"; START_INDEX="$2"; shift 2 ;;
    -e|--end|--end-index) need_value "$@"; END_INDEX="$2"; shift 2 ;;
    -c|--cus|--scale) need_value "$@"; CUS="$2"; shift 2 ;;
    --split) need_value "$@"; SPLIT="$2"; shift 2 ;;
    --time-limit|--time-limit-s) need_value "$@"; TIME_LIMIT_S="$2"; shift 2 ;;
    --mip-gap|--mip_gap) need_value "$@"; MIP_GAP="$2"; shift 2 ;;
    --cs-copies) need_value "$@"; [[ "$PROBLEM" == EVRPTW ]] || die "--cs-copies requires EVRPTW"; CS_COPIES="$2"; shift 2 ;;
    --dataset-path) need_value "$@"; EXTRA_ARGS+=(--dataset_path "$2"); shift 2 ;;
    --dataset-root) need_value "$@"; EXTRA_ARGS+=(--dataset_root "$2"); shift 2 ;;
    --output-path) need_value "$@"; EXTRA_ARGS+=(--output_path "$2"); shift 2 ;;
    --checkpoints) need_value "$@"; EXTRA_ARGS+=(--checkpoints_s "$2"); shift 2 ;;
    --no-skip-completed) EXTRA_ARGS+=(--no_skip_completed); shift ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) need_value "$@"; CONDA_ENV="$2"; shift 2 ;;
    --log-dir) need_value "$@"; LOG_DIR="$2"; shift 2 ;;
    --log-file) need_value "$@"; LOG_FILE="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "$START_INDEX" =~ ^[0-9]+$ && "$END_INDEX" =~ ^[0-9]+$ ]] || die "Indices must be non-negative integers"
(( START_INDEX < END_INDEX )) || die "--start must be less than --end"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "Workers and threads must be positive integers"
if [[ "${CUS,,}" =~ ^(cus)?([1-9][0-9]*)$ ]]; then SCALE="Cus${BASH_REMATCH[2]}"; else die "Invalid customer scale: $CUS"; fi
CMD=("$PYTHON_BIN" -u "$RUNNER")
if [[ -n "$CONDA_ENV" ]]; then CMD=(conda run --no-capture-output -n "$CONDA_ENV" "${CMD[@]}"); fi
CMD+=(--split "$SPLIT" --scale "$SCALE" --start_index "$START_INDEX" --end_index "$END_INDEX"
      --workers "$WORKERS" --threads "$THREADS" --time_limit_s "$TIME_LIMIT_S" --mip_gap "$MIP_GAP")
[[ "$PROBLEM" == EVRPTW ]] && CMD+=(--cs_copies "$CS_COPIES")
CMD+=("${EXTRA_ARGS[@]}" --verbose)
if [[ "$DRY_RUN" == 1 ]]; then exec "${CMD[@]}" --dry-run; fi

LOG_FILE="${LOG_FILE:-$LOG_DIR/gurobi_${PROBLEM,,}_${SCALE}_${SPLIT}_${START_INDEX}_${END_INDEX}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
if [[ "$DETACH" == 1 && "${_GUROBI_DETACHED:-0}" != 1 ]]; then
  export _GUROBI_DETACHED=1 LOG_FILE
  nohup bash "$SCRIPT_DIR/run_range.sh" "$PROBLEM" "${ORIGINAL_ARGS[@]}" > "${LOG_FILE}.launcher" 2>&1 &
  echo "Started $PROBLEM Gurobi: PID=$!; log=$LOG_FILE"
  exit 0
fi
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
