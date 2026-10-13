# VRPTW controlled E1 batch

Run from `CaliRoute` after pulling the updated branch. Each command detaches all
preparation with `nohup` and a separate session. CPU audit and smoke precede GPU
preflight; training starts only on idle, locked GPUs. Existing runs are preserved.

| Server profile | GPU | Experiment |
|---|---|---|
| `a6000` | 0 | Cus50 SL-PPO |
| `a6000` | 1 | Cus15 SL-PPO |
| `2080ti_b` | 0 / 1 | Cus50 AWBC / DAPG |
| `2080ti_b` | 2 / 3 | Cus15 AWBC / DAPG |
| `2080ti_a` (local) | 0 / 1 | Cus15 PPO base / PPO + RDI + AGDA, then Cus50 |
| `2080ti_3` | 0 + 1 | Cus100 SL-PPO, one distributed run |

The third card on `2080ti_3` is unassigned. Two ranks retain global batch64 and
four equal minibatches; three ranks would change that protocol. Cus15 and Cus50
use **one GPU per method**, not two GPUs per run. Local GPU2/3 keep the existing
Cus100 PPO + RDI + AGDA run. Cus50 PPO base and PPO + RDI + AGDA are prepared next and wait for both Cus15
training and accepted tests/reporting. This dependency does not include Cus100
or GPUs2/3. See `<Cus50 campaign>/handoff_status.json` for queue status.

```bash
bash scripts/run_e1_vrptw_batch.sh a6000 --execute
# On B:
bash scripts/run_e1_vrptw_batch.sh 2080ti_b --execute
# On the three-card server:
bash scripts/run_e1_vrptw_batch.sh 2080ti_3 --execute
# Local A (already started; do not duplicate):
bash scripts/run_e1_vrptw_batch.sh 2080ti_a --execute
```

Omit `--execute` for a preview. `E1_PYTHON`, `E1_DATA_ROOT` and `E1_VRPTW_ROOT`
can override interpreter, dataset root and campaign parent. Chunks may be
configured before preparing a new campaign with `E1_<METHOD>_PPO_CHUNK` (upper
case method, e.g. `E1_PPO_RDI_AGDA_PPO_CHUNK`) and `E1_EXPERT_CHUNK`.
A6000 SL-PPO defaults: PPO32/expert128; B expert methods and two-rank SL-PPO:
PPO8/expert32; local base32/graph14. These are execution chunk candidates, not
batch changes or a measured guarantee of 80% memory occupancy. Hardware preflight
must pass; failed runs remain blocked for inspection rather than auto-resuming.

Protocol: scratch seed3009, 1500 online loops, global batch64, n_traj50, four PPO
passes, four minibatches, gamma0.99, fixed LR1e-4, FP32. AWBC/DAPG retain their
existing method-specific BC warmup; online budget remains 1500. Validation every
50 online loops, K50. The base excludes explicit road features but retains actual
time windows, service duration, clock and physical feasibility constraints.
Experts come only from the corresponding VRPTW training split; independent route
validation and teacher forcing precede publication of the expert pool. Shared
physical distance unit is fitted on the corresponding training split only.

Logs: `results/e1/launchers/VRPTW_<profile>_V1/cus{15,50,100}.log`.
Campaigns: `results/e1/E1_VRPTW<scale>_S3009_E1500_<PROFILE>_V1`.
Training progress: `<campaign>/scheduler_<profile>/status.json`.
Automatic test progress: `<campaign>/test_watcher/status.json`.

After completed training, the watcher tests validation-selected best weights on
1000 independent test instances, K50, batch8; frozen model/environment code is
used. Only accepted results update per-scale main/partial CSVs, evidence JSON,
`CVRP_VRPTW_test_overview_20261013.csv`, and the VRPTW HTML/PNG/PDF table. Common
subset metrics remain TBD until aligned per-instance results are collected.
Remote tables initially contain placeholders for unsynchronized results; they
are not a replacement for the consolidated report on the local server.
