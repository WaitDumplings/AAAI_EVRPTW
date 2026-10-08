# CaliRoute

CaliRoute is the training code for real-road CVRP, VRPTW, and EVRPTW
experiments. The repository is organized around one main research line:

```text
PPO backbone -> SL-PPO
```

`SL-PPO` is the proposed method. `PPO`, `DAPG`, and `AWBC` are comparison
methods that share the same backbone and environment interface.

## EVRPTW100 dual-GPU scratch comparison

Use one pair of GPUs for the original `f388343` SL-PPO model/loss with an external
synchronous execution adapter, and another pair for the optimized model:

```bash
bash scripts/run_evrptw100_original_dual.sh --gpus 0,1 --epochs 1500
bash scripts/run_evrptw100_optimized_dual.sh --gpus 2,3 --epochs 1500
```

Both launch in the background, train from random initialization, and use two
synchronized ranks × 32 instances × 50 trajectories, five PPO passes, LR `1e-4`,
and full validation every 50 epochs. Each runs a separate full-allocation
two-epoch preflight before restarting from scratch for formal training. No
initialization checkpoint is needed. Use `--prepare-only` to inspect configs;
on a two-card server pass `--gpus 0,1` to either script.

The [dual-GPU EVRPTW guide](docs/evrptw_dual_scratch.md) documents physical
charging, partial expert coverage, failed-rollout rewards, evaluation,
monitoring and the scope of local CPU/Gloo versus remote GPU validation.

## Current VRPTW100 scratch comparison

The current four-arm experiment starts every model from random initialization:
`legacy`, `physics`, `archive`, and `explore`. `legacy` uses the complete original
source at commit `f388343`, with an external evaluation adapter; it is not the
current model with features disabled. From `CaliRoute/`:

```bash
bash scripts/run_scratch_comparison.sh --seed 3010 --gpus 0,1,2,3
# Prepare configs and source snapshots without starting training:
bash scripts/run_scratch_comparison.sh --prepare-only --seed 3011 --gpus 0,1,2,3
```

The shell starts a background supervisor with one GPU per arm and a separate
preflight. All arms use 300 epochs, 64 instances × 50 trajectories, **five PPO
passes**, four minibatches, constant LR `1e-4`, entropy `0.01`, and SL coefficient
`0.5`. Validation uses all 1,000 instances, best-of-50, at epoch zero and every
50 epochs. No PPO-init checkpoint, previous results directory or bundled weight
asset is required; training expert routes are still required by SL-PPO.

Place `AAAI_Dataset` beside `CaliRoute`, or pass `--data-root /path/to/AAAI_Dataset`.
Legacy uses PPO chunk size 8 and expert encoding chunk 128; current arms use
PPO chunk 15 for memory control. Original
four-pass/priority defaults are explicitly overridden to five passes/uniform
sampling for this experiment. Earlier warm-start launchers below remain as
historical experiment entry points. See [the scratch comparison guide](docs/scratch_comparison.md)
for data files, source provenance, evaluation limits and monitoring commands.

## Data Layout

The code and dataset live under the same `AAAI_EVRPTW` workspace:

```text
AAAI_EVRPTW/
  CaliRoute/
  AAAI_Dataset/               # downloaded separately; excluded from Git
    dataset/                 # train/val splits and reference solutions
    test_release/            # frozen final test splits
```

Dataset download link: **to be added**. Extract the download into
`AAAI_EVRPTW/AAAI_Dataset/`.

By default `train.py` resolves data from `../AAAI_Dataset/dataset` relative to
the CaliRoute code root. You can override this with `--data-root`.
The frozen test set is stored separately at `../AAAI_Dataset/test_release`,
with splits such as `evrptw/test/Cus50/`. It is reserved for final evaluation.

Expected split layout:

```text
AAAI_Dataset/dataset/
  cvrp/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
  vrptw/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
  evrptw/
    train/Cus15/
    val/Cus15/
    train/Cus50/
    val/Cus50/
    train/Cus100/
    val/Cus100/
```

Each split directory should contain:

```text
instances.pkl
metadata.json
expert_solutions.csv
gurobi_summary.csv
gurobi_time_trace.csv
public_metadata.json
```

## Quick Start

From the `AAAI_EVRPTW` repository root, install dependencies and run SL-PPO
on EVRPTW Cus50:

```bash
cd CaliRoute
pip install -r requirements.txt
python train.py \
  --problem evrptw \
  --customers 50 \
  --charging-stations 10 \
  --offline-method slppo \
  --device cuda:0
```

The command automatically uses:

```text
../AAAI_Dataset/dataset/evrptw/train/Cus50
../AAAI_Dataset/dataset/evrptw/val/Cus50
../AAAI_Dataset/dataset/evrptw/train/Cus50/expert_solutions.csv
../AAAI_Dataset/dataset/evrptw/val/Cus50/gurobi_summary.csv
```

Inspect the generated config without launching training:

```bash
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo --dry-run --print-config
```

## Methods

All methods use the same PPO rollout/update backbone:

```bash
python train.py --problem cvrp   --customers 50 --offline-method ppo
python train.py --problem vrptw  --customers 50 --offline-method dapg
python train.py --problem evrptw --customers 50 --charging-stations 10 --offline-method awbc
python train.py --problem evrptw --customers 50 --charging-stations 10 --offline-method slppo
```

Default update counts:

```text
PPO/DAPG/AWBC: ppo_update_epochs = 3
SL-PPO:        ppo_update_epochs = 4
```

SL-PPO uses a solution-level objective with group-relative and reference
advantages. Its candidate pool is configurable:

```bash
--pool weighted   # default priority sampler
--pool best       # always choose the currently highest-priority instances
--pool off        # no priority sampler
```

The bundled multi-method launch scripts use `weighted` by default. Set
`SLPPO_POOL=best` before running a script to switch the SL-PPO branch to
best-pool.

The expert-candidate term is exposed with descriptive public names:

```bash
--sl-expert-candidate-weight 0.60
```

## Paper Launch Scripts

The scripts under `scripts/paper_2080ti_runs/` assume this layout:

```text
AAAI_EVRPTW/
  CaliRoute/
  AAAI_Dataset/
    dataset/
    test_release/
```

Activate the Python environment, then run a full pipeline with `all` from the
CaliRoute root:

```bash
bash scripts/paper_2080ti_runs/run_evrptw_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_vrptw_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_vrptw_cus50_all_methods.sh all
bash scripts/paper_2080ti_runs/run_cvrp_cus15_all_methods.sh all
bash scripts/paper_2080ti_runs/run_cvrp_cus50_all_methods.sh all
```

The scripts require a run group such as `all`, or a GPU id and method name.
For example, run PPO on GPU 0:

```bash
bash scripts/paper_2080ti_runs/run_evrptw_cus15_all_methods.sh 0 ppo
```

`all` starts PPO and then launches DAPG, SL-PPO, and AWBC after the PPO
initialization checkpoint is available (epoch 100 by default). Single-method
DAPG, SL-PPO, and AWBC launches wait for that checkpoint. Scripts detach by
default; append `--foreground` to keep the launcher in the current terminal.

Default GPU mapping is:

```text
PPO   -> GPU0
DAPG  -> GPU1
SL-PPO -> GPU2
AWBC  -> GPU3
```

Override only the GPU ids if needed:

```bash
GPU_PPO=0 GPU_DAPG=1 GPU_SLPPO=2 GPU_AWBC=3 \
  bash scripts/paper_2080ti_runs/run_vrptw_cus50_all_methods.sh all
```

Cus15 and Cus50 use the same fixed 2080Ti-safe rollout/update geometry from
`scripts/paper_2080ti_runs/common_2080ti_config.sh`:

```text
num_envs=64, n_traj=50, ppo_step_chunk_size=16,
num_minibatches=4, eval_batch_size=128
```

SL-PPO uses `ppo_update_epochs=4`; PPO, DAPG, and AWBC use
`ppo_update_epochs=3`. These values are intentionally shared across methods so
the comparison does not change rollout or minibatch geometry.

## Ablations

Backbone and decoder ablations are controlled from the same entry point:

```bash
python train.py --problem evrptw --customers 50 --charging-stations 10 \
  --offline-method slppo \
  --dde on \
  --qkv-delta none \
  --action-key on \
  --action-bias on \
  --distance-injection encoder
```

Supported switches:

```text
--dde on/off
--qkv-delta none/k/v/kv
--action-key on/off
--action-bias on/off
--distance-injection encoder/none
```

Encoder distance injection is implemented as road-distance attention bias.
Embedding-level distance injection is intentionally not exposed until it is
implemented as a separate model path.

## Generated Outputs

Training outputs are written under `results/` and are intentionally ignored by
git. The sibling `AAAI_Dataset/` directory is also excluded from the
`AAAI_EVRPTW` Git repository and is distributed through a separate download.

## Project Structure

```text
train.py, main.py              Public training entry points.
caliroute/                     Method presets, config builder, CLI, ablation flags.
offline2online/                Stable training engine retained for reproducibility.
configs/templates/             Reference generated configs.
scripts/paper_2080ti_runs/     Paper experiment launch scripts.
scripts/validation/            Short regression checks against old runs.
docs/                          Engineering notes and review documents.
```

Generated logs and checkpoints are written under `results/`, which is ignored by
git.

## Gurobi benchmarks

`EVRPTW_Benchmark/Exact/Gurobi_Solver/{CVRP,VRPTW,EVRPTW}/` contains the original
AAAI Gurobi batch solvers. They use the sibling `AAAI_Dataset` and write results
under `results/gurobi/`. The solvers preserve the earlier `gurobi_mul`
formulations; see the [benchmark guide](EVRPTW_Benchmark/Exact/Gurobi_Solver/README.md)
for setup, dry runs, test-set selection, and EVRPTW warm starts.

For a full frozen test batch, choose only the problem and scale:

```bash
bash scripts/run_gurobi_test.sh evrptw 50
```

Problems: `vrptw`, `evrptw`, `cvrp`; scales: `15`, `50`, `100`. This starts all
1,000 test instances in the background with 30 workers, one Gurobi thread per
worker, and a 2-hour optimization limit per instance. Results and logs are under
`results/gurobi/<problem>/test/CusN/`. For EVRPTW, the launcher automatically uses
1 copy per physical charging station for Cus15 and 2 copies for Cus50/Cus100.
EVRPTW's optional vehicle-count tie-break is disabled to avoid a second 2-hour
solve; see the benchmark guide for details.

Resume is automatic: re-run the same command to skip completed instance IDs in
that task/scale's `gurobi_summary.csv`. `TIME_LIMIT` results count as completed;
errors, interrupted runs, and missing results retry. The shell shows the checked
summary path; the background log reports exact skipped/pending counts before
solving. If no instances remain, the runner exits without creating workers.

To split EVRPTW Cus100 across five servers, add that server's shard number:

```bash
bash scripts/run_gurobi_test.sh evrptw 100 1  # Use 1, 2, 3, 4, or 5 on each server.
```

Each shard owns 200 consecutive entries in the same frozen bundle, selected before
resume. Shards cover all 1,000 instances without overlap and keep the 30-worker,
2-hour, 2-CS-copy settings. Outputs and resume state are separate under
`results/gurobi/evrptw/test/Cus100/shard_N_of_5/`; `test_shard.json` records the
assigned IDs. Reuse the same shard number when resuming. Without the third
argument, the command still runs the full bundle. See the benchmark guide for
all five commands.


## Reward and normalization fine-tuning comparison

Run from `CaliRoute/` after pulling the same source branch on each machine.
The launcher uses repository-relative defaults, snapshots source/config/weights,
and starts a detached supervisor. It waits for unused GPUs; each arm first runs
an isolated two-epoch GPU preflight, then starts its full training from the
shared initialization. Preflight epochs are never counted as training progress.

```bash
# First 4 x 2080 Ti server: four single-GPU arms, seed 3009.
bash scripts/run_reward_norm_comparison.sh --seed 3009 --gpus 0,1,2,3

# Second 4 x 2080 Ti server: repeat all four arms with seed 3010.
bash scripts/run_reward_norm_comparison.sh --seed 3010 --gpus 0,1,2,3
```

Use `--prepare-only` to inspect configs without launching. Use `--gpus 0,1,2`
on a three-card server or `--gpus 0,1` on a two-card server; the four arms queue
on those devices. One GPU is used per arm, including on A6000 hardware. Compare
runtime within the same GPU model. `PYTHON_BIN` can select your Python environment;
`--data-root` and `--init-checkpoint` override relocated data/weights.

The experiment branch includes one fixed weights-only initialization (about
19 MB), so `git pull --ff-only` also supplies the required model:

```text
assets/reward_norm/vrptw100_update5_epoch0300.pt
assets/reward_norm/vrptw100_update5_epoch0300.json
```

It contains the exact actor/critic tensors from update5 epoch300, the source
seed, and model/environment units. Optimizer, sampler and replay state are
excluded because this experiment initializes them afresh. This is an
initialization artifact, not a resumable checkpoint. The adjacent JSON records
both the original checkpoint SHA256 and the exported artifact/tensor SHA256.
The launcher checks the bundled hash and epoch before preparing any experiment;
all machines therefore start from the same weights. Generated training
checkpoints and datasets remain excluded from Git.

Place these four dataset/reference files on each server (or set `--data-root`):

```text
../AAAI_Dataset/dataset/vrptw/train/Cus100/instances.pkl
../AAAI_Dataset/dataset/vrptw/train/Cus100/expert_solutions.csv
../AAAI_Dataset/dataset/vrptw/val/Cus100/instances.pkl
../AAAI_Dataset/dataset/vrptw/val/Cus100/gurobi_summary.csv
```

The launcher reports all missing required files together and records their
hashes, including optional per-split metadata when present. No old `results/`
directory is required. An explicit `--init-checkpoint` remains available for the
original full checkpoint; compare `initialization_provenance.model_state_sha256`
in manifests to establish identical weights across differently packaged files.

To reproduce the export from the trusted original checkpoint:

```bash
python scripts/reward_norm_initialization.py \
  --source /path/to/checkpoint_epoch_0300.pt \
  --output /tmp/vrptw100_update5_epoch0300.pt
```

| Arm | gamma | Normalization |
|---|---:|---|
| baseline | 0.99 | existing |
| reward | 1.0 | existing |
| normalization | 0.99 | shared actor RMS, physical-cost LOO SL advantage, PopArt |
| combined | 1.0 | shared actor RMS, physical-cost LOO SL advantage, PopArt |

All arms fine-tune VRPTW100 for 80 additional epochs: 64 instances x 50
trajectories per rollout, 5 PPO passes, 4 minibatches, LR 1e-5, entropy 0.002,
and SL coefficient 0.35. They use fresh optimizers/replay and uniform shuffled
training instances. These common fine-tuning settings differ from the earlier
500-epoch sweep. New runs validate all 1,000 instances, best-of-50, at epoch 0,
every 50 epochs, and the final epoch (even if it is not a multiple of 50).
Use `--eval-interval 20` to reproduce the earlier evaluation cadence. Existing
frozen runs retain their recorded interval. Test is reserved for final selection
and is not run here.

To extend an already running comparison to **300 total fine-tuning epochs**:

```bash
# On this host, select its one active seed-3009 comparison.
bash scripts/extend_reward_norm_comparison.sh --seed 3009 --epochs 300
# On the other host, use --seed 3010 instead.
```

Alternatively pass `--experiment results/optimization/<source-run>` explicitly
(required if more than one run for that seed is active, or the source already
completed). The extension controller detaches and prints a new `E300` directory.
It waits for the original 80-epoch run to finish, copies each final full
checkpoint and all committed history, then trains epochs 81–300 automatically.
The new status reports target 300 and shows source progress while waiting;
`extension.json` in the original run links to it. There is no interruption or
loss of the current training work. The original run and its frozen source are
preserved; only orchestration code changes in the continuation snapshot.

Optimizer moments, AMP scaler, RNG, sampler cursor, replay, actor RMS, PopArt and
historical best are restored. The same seed, batches, five PPO passes, constant
LR/entropy, and the source validation interval continue; no second preflight or
epoch-zero evaluation is run. Final-epoch validation is always retained, including
nonmultiple source boundaries. This extension supports completed single-GPU
reward/norm source horizons with unchanged constant schedules.
A failed/interrupted source stops the extension queue with an error, rather than
silently restarting training from weights. Stop the extension supervisor if you
want to cancel the continuation; the source run is not stopped by cancellation.

To view the four validation curves at any point during training:

```bash
python scripts/plot_reward_norm_eval.py results/optimization/<run-directory>
```

This writes `plots/validation_curves.png`, `.svg`, `.pdf`, raw `.csv`, and a
`.json` with snapshot time and source hashes. Curves show raw completed validation
points without smoothing or extrapolation; the latest common evaluation epoch
is marked to distinguish comparisons at the same epoch from partial newer data.

The printed experiment directory contains `status.json`, `comparison.json`,
`manifest.json`, `hardware.jsonl`, each arm's `preflight/` and `monitoring/`, and
independently validated routes. `comparison.json` aligns completed validation
epochs and checks the common initial evaluation. The new host mode currently
requires a scalar critic, complete feasible rollouts, no reward shaping, and a
single GPU; it fails explicitly for unsupported settings. See
[the design and ablation notes](docs/review_and_ablation_plan.md#reward-and-normalization-screen-2026-10-07)
for objective semantics and scope.


## Physical network-input normalization comparison

This separate screen changes network inputs while holding the reward definition,
SL-PPO losses, gamma 0.99, and `physical_shared_popart` training normalization
fixed. It uses the completed normalization arm's best epoch-300 weights, included
as `assets/input_norm/vrptw100_norm_epoch0300.pt` with checked SHA256 metadata.
Optimizers, replay, actor RMS and PopArt statistics start fresh in every arm.

```bash
# Four independent single-GPU arms, 300 additional epochs, full validation every 50.
bash scripts/run_input_norm_comparison.sh --seed 3009 --gpus 0,1,2,3

# Replication on a second server with the same dataset and source branch.
bash scripts/run_input_norm_comparison.sh --seed 3010 --gpus 0,1,2,3
```

Two- and three-card servers use `--gpus 0,1` or `--gpus 0,1,2`; excess arms queue.
The shell runs in the background; `--prepare-only` only writes frozen configs.
Each arm first passes a separate two-epoch GPU preflight. All four share 64
instances x 50 trajectories, update5, four minibatches, and LR 1e-5. Validation
uses all 1,000 instances, best-of-50, at epoch 0, every 50 epochs, and the end.
The test split is reserved for final evaluation.

| Arm | Coordinate input | Added physical context |
|---|---|---|
| `legacy` | Existing per-instance axis min-max | No |
| `depot` | Depot-relative kilometres / frozen shared distance unit | No |
| `context` | Existing per-instance axis min-max | Directed node relations and vehicle/global resources |
| `combined` | Depot-relative kilometres / frozen shared distance unit | Both node and global context |

The shared distance unit is 43.638668060302734 km, inherited from the source
model and frozen across all instances/cities/customer counts in this screen.
It is a unit choice for this experiment, not a claimed universally optimal value.
Time uses the working horizon, energy the battery capacity, and demand the cargo
capacity; the added global context records the corresponding physical scales.
No physical transition, feasibility mask, route objective or reward is changed.
The two context MLP output layers initialize to zero. Therefore epoch-0 results
must agree within `legacy/context` and within `depot/combined`; changed coordinates
can shift the starting performance between those pairs. This mature-model screen
does not establish from-scratch or cross-size performance.

Training monitors add input-encoder residual magnitude, gradient norm and update
norm alongside KL, clipping, entropy, feasibility, reward-normalization statistics
and runtime. New fixed-unit checkpoints record the input schema and units; full
resume rejects incompatible input settings. Legacy configs without an explicit
observation unit retain their historical fallback behavior. Older weights may
initialize the new adapter using
the explicit whitelist, with the representation migration recorded.

```bash
python scripts/plot_reward_norm_eval.py results/optimization/<input-run-directory>
```

The input configuration remains opt-in until matched validation and repeat-seed
results justify fixing it. Literature rationale and acceptance criteria are in
[the design and ablation notes](docs/review_and_ablation_plan.md#physical-input-normalization-and-representation-screen-2026-10-07).

## Physical model-integration comparison

The second-stage screen fixes **combined physical input in all four arms** and
compares how the embedding, encoder and decoder use that input. Combined is a
controlled assumption here, not a claim that the preceding input screen has
already selected a winner. Reward, gamma, PopArt, actor RMS and SL-PPO stay fixed.

On the other four-card server, from the repository root:

```bash
git fetch origin
git switch opt/model-integration-20261007
git pull --ff-only
cd CaliRoute
bash scripts/run_model_integration_comparison.sh --seed 3010 --gpus 0,1,2,3
```

| Arm | Static fusion and directed relations | Resource-aware decoder |
|---|---|---|
| `baseline` | Current model with combined input | Current decoder |
| `static` | Grouped physical embedding, resource conditioning, cached directed-edge encoding | Current decoder plus the static relation reader |
| `dynamic` | Current model with combined input | Candidate transition features, resource conditioning, separate observation/action masks |
| `combined` | Static changes | Dynamic changes plus the static relation reader |

New output heads start at zero. All arms share weights and the same combined
input, so their epoch-zero policy/critic behavior should match; full-split
validation checks are recorded in `comparison.json`. Unlike the first-stage
input screen, there is no coordinate-mode difference between arms.

The source initialization is the **bundled prior Norm epoch-300 checkpoint**
`assets/input_norm/vrptw100_norm_epoch0300.pt`, not a trained combined-input best
checkpoint. No checkpoint from another server's `results` directory is needed.
Every arm starts with fresh optimizer, replay, actor RMS and PopArt state; each
preflight is discarded before the formal run. Initialization allows only the
explicitly named new modules to be absent from the older weights.

Defaults are 300 additional epochs, one GPU per arm, 64 instances x 50
trajectories, PPO update=5, four minibatches, PPO chunk=12 and LR=1e-5. Each arm
first passes a two-epoch GPU preflight using the **same training batch, trajectory
count, update count and chunk size**. Only preflight duration and validation
size (four instances, best-of-four) are reduced, so it checks actual training
memory before starting the long run. Formal validation covers all 1,000 validation instances,
best-of-50, at epochs 0, 50, 100, 150, 200, 250 and 300. Test data are not used for
selection. The shell detaches the supervisor; it waits for idle GPUs and queues
arms when fewer cards are supplied. `--prepare-only` writes frozen inputs,
source and configs without starting jobs. Existing jobs are not stopped.

The lightweight screen leaves the optional edge-value messages and edge-state
updates off. To run the heavier structure as a separate experiment:

```bash
bash scripts/run_model_integration_comparison.sh --seed 3010 --gpus 0,1,2,3 \
  --edge-messages --edge-updates
```

These flags affect only `static` and `combined` and are recorded in the manifest.
They may increase memory/runtime; compare their own matched baseline and report
same-time as well as same-epoch quality. They do not change the reward or the
physical feasibility mask. For a different data mount, pass
`--data-root /path/to/AAAI_Dataset`. To change PPO memory chunking consistently
across arms, pass `--chunk-size N` and keep that choice in the experiment record.
If full-shape preflight runs out of memory, start a fresh comparison with
`--chunk-size 8` for all arms; do not silently reduce one arm's batch or trajectory
count. A two-epoch pass checks the exercised allocation, not every possible
later replay/evaluation peak.

Results are written beneath
`results/optimization/MODEL_INTEGRATION_VRPTW100_S3010_E300_<UTC>/`.
Inspect `status.json`, `comparison.json`, `supervisor.log` and each arm's monitor
output for feasibility, KL, clipping, entropy, module gradient/update magnitudes,
normalization statistics, runtime and GPU memory. Plot observed validation:

```bash
python scripts/plot_reward_norm_eval.py results/optimization/<model-integration-run>
```

The [design notes](docs/review_and_ablation_plan.md#physical-model-integration-screen-2026-10-07)
distinguish paper mechanisms, our adaptations and validation requirements.


## Track experiment snapshots

The standard-library-only reader prints progress, latest validation, best selected
checkpoint, KL, clipping and LR, followed by validation at the latest complete
common epoch. From `CaliRoute`:

```bash
python scripts/watch_comparison.py --watch 30
python scripts/watch_comparison.py results/optimization/<run>
```

With no path, it selects the most recently modified `comparison.json` once and
keeps watching that experiment. Pass a run directory or a JSON file to select
one explicitly; omit `--watch` for one snapshot and press Ctrl-C to stop watching.
A directory falls back to `status.json` before `comparison.json` exists. It reads
formal metrics only and shows missing values as `--`. Snapshot timestamps expose
stale reports; this command does not check live GPU utilization or process liveness.


### Physical reward and exploration screen

The next incremental screen fixes combined inputs and the full stage-two model,
then compares `legacy`, `physics`, `archive`, and `explore` on four independent
GPUs. It adds consistent road/time/energy inputs, finite-route gamma=1, valid-action
PPO loss, optional fresh-KL stopping, structurally diverse verified replay, and
independent branch search. Search trajectories never enter on-policy PPO.

```bash
git fetch origin
git switch opt/physics-exploration-20261008
git pull --ff-only
# From CaliRoute:
bash scripts/run_physics_exploration_comparison.sh --seed 3010 --gpus 0,1,2,3
```

The default is 300 additional epochs and full validation every 50 epochs. Every
arm runs a full-batch two-epoch preflight first. See
[configuration, controls and monitoring](docs/physics_exploration_comparison.md)
for frozen units, initialization, optional `--target-kl`, and the extra compute
budget in the exploration arm. Accuracy gains require the resulting controlled
experiments; these changes do not establish global optimality.
