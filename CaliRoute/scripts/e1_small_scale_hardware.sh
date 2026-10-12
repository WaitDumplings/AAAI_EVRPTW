# Sourced by small-scale launchers. No jobs or files are created here.
e1_small_scale_hardware() {
  E1_HARDWARE_ARGS=()
  case "$1" in 15|50) ;; *) return 0;; esac
  local names=(BASE RDI_AGDA AWBC DAPG SLPPO)
  local methods=(ppo_base ppo_rdi_agda awbc dapg slppo)
  local defaults=(32 14 8 8 8)
  local i key value
  printf 'Execution tuning Cus%s (global batch64, n_traj50, PPO passes4, minibatches4 unchanged):\n' "$1"
  for i in "${!names[@]}"; do
    key="E1_${names[$i]}_PPO_CHUNK"
    value="${!key:-${defaults[$i]}}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then printf 'Invalid %s: positive integer required\n' "$key" >&2; return 2; fi
    E1_HARDWARE_ARGS+=(--method-ppo-chunk "${methods[$i]}=$value")
    printf '  %s: PPO chunk=%s\n' "${methods[$i]}" "$value"
  done
  value="${E1_EXPERT_CHUNK:-32}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then printf 'Invalid E1_EXPERT_CHUNK\n' >&2; return 2; fi
  E1_HARDWARE_ARGS+=(--expert-chunk "$value")
  printf '  Expert chunk=%s; native RRNCO/RADAR parameters unchanged.\n' "$value"
  printf 'Cus50 PPO chunks32/14 short-profiled; Cus15 candidates require hardware preflight. Expert methods retain8 pending their own profiling.\n'
}

# Exit success means some selected run already has artifacts; caller skips it.
# An empty frozen campaign may still be launched with its original settings.
e1_small_existing() {
  local campaign="$1"; shift
  [[ -f "$campaign/manifest.json" ]] || return 1
  local python="${E1_PYTHON:-$E1_ROOT/../.venv/bin/python}"
  [[ -x "$python" ]] || python="$(command -v python)"
  local decision
  decision=$(PYTHONPATH="$E1_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$python" - "$campaign" "$@" <<'PY'
import sys
from pathlib import Path
from e1.runner import verify_campaign
from e1.lifecycle import assert_can_start
m=verify_campaign(Path(sys.argv[1]))
existing=[]
for method in sys.argv[2:]:
    try: assert_can_start(m,method)
    except FileExistsError: existing.append(method)
    if (Path(m['methods'][method]['run_dir'])/'hardware_preflight').exists() and method not in existing:
        existing.append(method)
print(','.join(existing))
PY
  ) || { printf 'Frozen campaign verification failed\n' >&2; exit 2; }
  if [[ -n "$decision" ]]; then
    printf 'Preserved existing runs/preflights: %s. No restart; use explicit resume for stopped runs.\n' "$decision"
    return 0
  fi
  printf 'Using existing frozen configs; new chunk defaults apply ONLY to newly prepared campaigns.\n'
  return 1
}
