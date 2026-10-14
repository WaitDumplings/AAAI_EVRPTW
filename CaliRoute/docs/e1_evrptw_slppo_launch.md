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
