# EVRPTW SL-PPO: Cus15 and Cus50 on two RTX A6000 cards

From the repository root:

```bash
git fetch origin
git switch fix/e1-no-road-base-20261012
git pull --ff-only
cd CaliRoute
bash scripts/run_e1_evrptw_slppo_a6000.sh           # preview only
bash scripts/run_e1_evrptw_slppo_a6000.sh --execute # nohup + separate session
```

GPU 0 runs Cus50, GPU 1 runs Cus15: one independent single-GPU experiment each.
The two scales audit data independently, then run CPU smoke/resume and GPU
preflight before formal training. GPU locks prevent training on occupied cards.
Repeated calls preserve existing campaigns; failures require inspection, not a
silent restart. Running VRPTW campaigns use their frozen source and are unaffected.

Protocol: scratch seed3009, 1500 online loops, global batch 64 per experiment,
n_traj 50, PPO passes 4, minibatches 4, gamma 0.99, constant LR 1e-4, FP32. This uses
the current E1 graph RDI/AGDA SL-PPO model, external train experts and cost-only
incumbent contract. No initialization checkpoint or extra PPO warmup is added.
Validation: every 50 loops, 1000 instances, K=50. Final test: validation-selected
best checkpoint, 1000 independent instances, K=50, batch 8.

Physical resources remain active: directed road distance, travel time, time
windows/service, battery and fixed-full charging duration. Explicit time/energy
matrices are preferred; if absent, both derive from the same road matrix and
instance speed/consumption. The train-only distance scale uses customer depot
round trips, excluding stations. Cus15 has 3 physical stations and a 76-step cap;
Cus50 has 10 stations and a 244-step cap, matching the environment's 4x terminal
limit. Gurobi station copies are solver variables; the policy uses physical
station IDs, with the existing environment rule of at most one visit per station
per vehicle route. Training expert solutions must pass independent physical
validation and teacher-forced replay before entering the expert pool.

Required data are under `AAAI_Dataset/dataset/evrptw/{train,val}/Cus{15,50}` and
`AAAI_Dataset/test_release/evrptw/test/Cus{15,50}`. Train experts must accompany
the training split. Test reference solutions are never used for training,
checkpoint selection or expert-pool preparation.

Overrides before creating a new campaign:

- `E1_PYTHON`: interpreter (defaults to repository `.venv/bin/python`, then `python`).
- `E1_DATA_ROOT`: dataset root.
- `E1_EVRPTW_ROOT`: campaign parent (defaults to `CaliRoute/results/e1`).
- `E1_SLPPO_PPO_CHUNK`: PPO time chunk, default 32.
- `E1_EXPERT_CHUNK`: expert time chunk, default 128.

Chunk defaults are starting points for the A6000 preflight, not a measured
memory-occupancy guarantee. Global batch and trajectory budget remain fixed.

Preparation logs: `results/e1/launchers/EVRPTW_a6000_V1/cus{15,50}.log`.
Campaigns: `results/e1/E1_EVRPTW{15,50}_S3009_E1500_A6000_V1`.
Within each campaign:

- `scheduler_a6000/status.json`: hardware preflight and training status.
- `test_watcher/status.json`: automatic best-checkpoint test and acceptance.
- `runs/slppo/test/instances.jsonl`: accepted per-instance output.

Accepted tests update `results/e1/reports/EVRPTW_test_overview.csv` and the
EVRPTW HTML/PNG/PDF under `results/e1/reports/visualizations`. Existing CVRP and
VRPTW tables are preserved; absent EVRPTW method results remain TBD. Transfer
per-instance results and manifests back to the main server for consolidation.

Validation before release: 62 unit/integration checks passed. Real Cus15 and
Cus50 subsets passed two CPU training epochs with optimizer/expert replay,
checkpoint save and fresh-process full resume. The checkpoint evaluation path
also generated K=50 candidates per instance on two held-out smoke instances per
scale, preserving failures and reporting zero cost mismatches. These reduced
smoke runs do not estimate solution quality, full-dataset acceptance or A6000
memory use, and are never published as E1 test results. A6000 GPU preflight runs
on the target server before the formal scratch run.

## Local four-card host: Cus15 PPO comparison

```bash
bash scripts/run_e1_evrptw15_ppo_2080ti_a.sh --execute
```

GPU 0+1 run PPO base; GPU 2+3 run PPO + RDI + AGDA. Each is a two-rank
experiment with batch 32 per GPU (global 64), n_traj 50 and PPO passes 4,
from scratch for 1500 epochs. Both PPO time chunks default to 32. CPU smoke,
full-size GPU preflight, frozen sources, checkpoint backups and automatic best
checkpoint test are shared with the existing E1 lifecycle.

The EVRPTW base preserves station coordinates, battery state/capacity, charging
duration, time windows, service time, clock and physical action masks. Road
distance/time/energy matrices and derived graph input contexts are excluded at
the actor/critic boundary. The simulator and reward retain real physical costs.

Campaign: `results/e1/E1_EVRPTW15_S3009_E1500_2080TI_A_EV15_V2`.
Progress: `scheduler_2080ti_a/status.json` inside that campaign.

## Local four-card host: Cus100 PPO comparison

```bash
bash scripts/run_e1_evrptw100_ppo_2080ti_a.sh --execute
```

GPU 0+1 run the no-road PPO base and GPU 2+3 run PPO + RDI + AGDA, each from
scratch for 1500 epochs, seed 3009, global batch 64 (32 per GPU), n_traj 50,
PPO passes 4. Cus100 uses 20 physical stations and a 484-step cap, matching the
4x-terminal environment cap. PPO time chunks are 32 for base and 8 for graph
RDI+AGDA to fit the 11 GiB cards; this does not change sampling/update budgets.
GPU preflight must pass before each formal run. Best-checkpoint test runs
automatically at the end and updates the EVRPTW report.

Campaign: `results/e1/E1_EVRPTW100_S3009_E1500_2080TI_A_EV100_V1`.
Preparation log: `results/e1/launchers/EVRPTW_2080ti_a_ev100_V1/cus100.log`.
Training/test status: `scheduler_2080ti_a/status.json` and
`test_watcher/status.json` inside the campaign.

## Local four-card host: Cus15/Cus50 AWBC and DAPG

```bash
bash scripts/run_e1_evrptw_awbc_dapg_2080ti_a.sh          # preview
bash scripts/run_e1_evrptw_awbc_dapg_2080ti_a.sh --execute # nohup + setsid
```

GPU0 runs Cus50 AWBC; GPU1 Cus50 DAPG; GPU2 Cus15 AWBC; GPU3 Cus15
DAPG. Each independent run uses one GPU, global batch64, n_traj50, four PPO
passes and 1500 online loops. The existing E1 method presets retain their own
BC warmup and expert losses; these additional loops are recorded separately
from the online budget. Models and reward definitions match the other E1
expert-method runs. PPO chunks default to 8 for Cus50 and 32 for Cus15;
these are execution settings, not changes to batch or optimizer budgets.
Cus50 keeps 512 sampled expert transitions per loss, computed in microbatches
of 64 with activation checkpointing; AWBC weights use the full sampled batch.
Loss/gradient equivalence is tested for BC, DAPG and AWBC.

Training experts must pass independent EV energy/time/capacity validation and
teacher-forced action replay. Each scale runs CPU training/resume checks, then
each method runs full-batch GPU preflight before fresh formal initialization.
Failures block the affected run. No preflight checkpoint initializes training.
Final validation-selected best checkpoints automatically run the independent
1000-instance test, K50, batch8, and accepted results update the EVRPTW table.

Logs: `results/e1/launchers/EVRPTW_2080ti_a_ev_experts_V1/cus{15,50}.log`.
Campaigns: `results/e1/E1_EVRPTW15_S3009_E1500_2080TI_A_EV_EXPERTS_V1`
and `results/e1/E1_EVRPTW50_S3009_E1500_2080TI_A_EV_EXPERTS_V2`.
Cus50 V1 failed hardware preflight before formal training (OOM); it is retained
for audit and never supplies a training/test checkpoint.
Each has `scheduler_2080ti_a/status.json` and `test_watcher/status.json`.
Overrides before preparation: `E1_AWBC_PPO_CHUNK`, `E1_DAPG_PPO_CHUNK`,
`E1_EXPERT_CHUNK`, and the shared interpreter/data/output overrides above.

## Three-card 2080 Ti host: Cus50 PPO comparison

```bash
bash scripts/run_e1_evrptw50_ppo_2080ti3.sh          # preview
bash scripts/run_e1_evrptw50_ppo_2080ti3.sh --execute # nohup + setsid
```

GPU0 runs PPO base, GPU1 PPO + RDI + AGDA, each on one GPU; GPU2 is unused.
Both run from scratch with seed3009, batch64 per GPU/global, n_traj50, four
PPO passes, four minibatches, 1500 online epochs and constant LR1e-4.
Base hides road inputs while preserving physical EV resource state; enhanced
uses the same E1 graph RDI/AGDA configuration as the other scales. The PPO
chunks default to 32/8, respectively, subject to full-batch GPU preflight.
No experts or initialization checkpoint are used for either PPO method.

CPU smoke/resume, GPU preflight, source snapshots and GPU locks precede formal
training. Automatic best-checkpoint test uses 1000 independent instances, K50
and batch8; accepted tests update EVRPTW reports. Existing runs are preserved.

Campaign: `results/e1/E1_EVRPTW50_S3009_E1500_2080TI_3_EV50_V1`.
Status: `scheduler_2080ti_3/status.json` and `test_watcher/status.json` within
that campaign. Preparation log:
`results/e1/launchers/EVRPTW_2080ti_3_ev50_V1/cus50.log`.
Use `E1_PPO_BASE_PPO_CHUNK` / `E1_PPO_RDI_AGDA_PPO_CHUNK` before preparation
to adjust execution chunks; global batch and trajectory budgets stay fixed.
