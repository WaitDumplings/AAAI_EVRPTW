# CaliRoute

CaliRoute is the training code for real-road CVRP, VRPTW, and EVRPTW
experiments. The repository is organized around one main research line:

```text
PPO backbone -> SL-PPO
```

`SL-PPO` is the proposed method. `PPO`, `DAPG`, and `AWBC` are comparison
methods that share the same backbone and environment interface.

## Consolidated AAAI candidate (2026-10-10)

Use `scripts/run_aaai_candidate.sh` for the versioned **aaai_graph_v1** candidate.
It resolves the recorded successful Graph training implementation at `d436492`;
model, environment and trainer code retain those exact bytes. Research settings
live in [the recipe](configs/recipes/aaai_graph_v1.yaml), while memory chunks and
rank allocation live in [hardware profiles](configs/hardware). The older `train.py`
method presets and historical experiment scripts retain their original defaults.

The candidate currently supports **VRPTW100 and EVRPTW100**, with one 48GB GPU or
one pair of 2080 Ti GPUs per model. CVRP and Cus15/Cus50 need separate validation
before being added to this recipe. This entry point does not define E1.

```bash
# Inspect the complete configuration without creating a run or starting GPUs.
bash scripts/run_aaai_candidate.sh --problem vrptw --hardware 2080ti_dual --print-config

# Freeze code, data hashes and configs; prepare-only is also the default.
bash scripts/run_aaai_candidate.sh --problem vrptw --hardware 2080ti_dual --gpus 0,1 --prepare-only
bash scripts/run_aaai_candidate.sh --problem evrptw --hardware rtx48_single --gpus 0 --prepare-only
```

Use `--encoder current` for the matching modern encoder control. Both arms keep
the same AGDA, reward, normalization, SL-PPO, archive and search settings. The
encoder comparison changes the joint node/edge encoder and its decoder edge
interface, including the effective edge width (32 versus 16). CURRENT is distinct
from the historical original implementation.

A later explicit `--launch` creates a fresh background run, waits for the chosen
GPUs and performs an independent full-allocation two-epoch preflight. The formal
run then starts from scratch, discarding preflight state. Each preflight requires
40 successful optimizer updates per rank, finite gradients and matching rank
checksums. This does not certify every later replay allocation: archive weight
starts after 25 epochs, so memory headroom and long-run monitoring remain relevant.
No running experiment is stopped by this launcher. EVRPTW's 2080 Ti memory profile
still requires its own real GPU preflight; VRPTW's profile has passed one.

| Research setting | VRPTW100 | EVRPTW100 |
| --- | ---: | ---: |
| Global instances per rollout | 40 | 32 |
| Trajectories per instance | 50 | 50 |
| PPO passes / minibatches | 5 / 4 | 5 / 4 |
| Learning rate / gamma | 1e-4 / 1 | 1e-4 / 1 |
| Rollout and evaluation horizon | 201 | 512 |
| Default training | 1500 epochs, SL-PPO from epoch 1 | Same |
| Validation | All 1000 instances, best-of-50, epoch 0 and every 50 | Same |

Hardware profiles preserve these global budgets. Two ranks split the 8 exploration
instances and the 32-route archive intake limit. Per-instance archive capacity
is unchanged. Rank-local samplers, histories and masked-loss means make dual-GPU
execution numerically different from a single-GPU run even at the same global
batch. Explicit batch overrides are recorded research variants; evaluation batch
changes can also alter sampled trajectories. Every run records resolved settings,
component branches, hashes and observed optimizer/AMP counters.

### What the evidence supports

The local seed 3011, batch 40, PPO 5 controlled comparison at epoch 1000 gives
**215.9273 km for Graph versus 219.1131 km for CURRENT** (1.4539% lower); both have
100% independent-route-validation feasibility on 1000 instances. Graph is better
at all 20 common validation checkpoints from epoch 50 to 1000. It wins 757/1000 paired
instances at epoch 1000. Median non-evaluation epoch time over epochs 101–1000 is
84.10 s versus 82.30 s, about 2.19% slower. This is evidence for the complete encoder
and matching edge interface at one paired training seed, not proof of each graph
mechanism or a cross-seed significance result. Actual successful updates through
1000 are 19995/19997 (5/3 AMP skips), despite the common nominal 20 attempts per epoch.

The [evidence record](docs/experiments/aaai_candidate_evidence_20261010.json) separates
these local checks from user-reported remote scores. RTX VRPTW 214.7, EVRPTW 215.4,
and the other server's 223.6/216.6 need their latest epochs, configs and code lineage
before forming one consolidated results table. The third server's reported
seed 2010 versus 3010 is awaiting confirmation.

| Component | Candidate decision | Evidence boundary |
| --- | --- | --- |
| Joint directed node/edge encoder and decoder edge readout | Keep | Controlled whole-block comparison; internal mechanisms and edge width not separately isolated |
| Physical D/T/E context and AGDA candidate transitions | Keep reference implementation | Successful bundle plus physical-feature and gradient tests |
| Strict-distance reward, fixed physical units, gamma 1, actor RMS and PopArt | Keep reference configuration | Unit/reward invariants and bundle results; separate quality ablations pending |
| Online SL, expert candidates, structural archive and branch search | Keep | Successful bundle; individual contributions need matched-budget ablations |
| Finite masks, SL weight/gradient AMP consistency, shared/cached computations | Keep verified repairs | Correctness and equivalence tests; no full-chain AMP-equivalence claim |
| Original-P1 adapters, 100-epoch PPO warmup, PPO 3 | Exclude from this candidate | They change the successful recipe and have no established independent benefit here |

`experiment_protocol.resolved_components` describes the effective code path.
Under `physical_shared_popart`, online SL uses a leave-one-out physical-cost
advantage with the same actor RMS snapshot as PPO. Legacy online group/reference
settings in the source config are bypassed; expert and replay gates still operate
separately. A local `use_rdi_v2=False` or `use_agda_v2=False` switch removes a
residual adapter, not all RDI or AGDA information. Do not label these switches
as complete module ablations.

The release candidate preserves remaining numerical semantics, including mixed
FP32/AMP paths and rank-local loss reduction. More complete resource isolation,
full-chain precision changes and globally weighted masked losses belong to new
controlled versions. Current integration does not silently apply those changes.

Before freezing a final paper release, reconcile the third server's source/config
bundle, confirm the RTX revision and define E1's tasks, scales, methods, seeds,
training budgets and evaluation protocol. Useful attribution checks are additional
paired seeds, matched edge width, and separate expert/archive/search ablations.
To reconcile another server, export its actual run directories after pulling this
branch (run this command from `CaliRoute/`):

```bash
python scripts/export_run_evidence.py /path/to/original_run /path/to/optimized_run \
  --output /path/to/new_evidence_bundle --include-source
```

The exporter verifies recorded source, configuration and input hashes, and copies
configs, captured status/comparison and rank-zero training/validation CSVs.
`--include-source` also copies the verified source subset. Dataset contents and
checkpoints are excluded. The destination must be new; active logs are snapshots,
not a cross-file atomic record. The bundle can be reviewed with the exact source
branch/commit to resolve differences before the final paper release.

The historical sections below document how earlier experiments were launched.

## Joint node-edge graph encoder experiment (VRPTW100 / EVRPTW100)

`--encoder-variant graph` adds a new graph encoder to the existing **explore**
training bundle. It keeps physical input normalization, reward, PPO5, AGDA and
independent search settings fixed. It is an architecture experiment, with no
claim of superior accuracy or speed before paired training results are available.

The encoder adds directed incoming/outgoing road summaries to the typed node
embedding. Every layer uses compact edge states for per-head attention bias,
post-softmax gates and edge-value messages, then updates those edge states from
separate source/target projections, reverse edges, node interactions and global
context. Final edge states feed the existing resource decoder. Both tasks use
256-dimensional nodes, two layers and 32-dimensional edge states by default.

The design draws on [UniteFormer (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/86ddf3543ad437d71c37e510f41b1a53-Abstract-Conference.html)
for joint routing node/edge encoding, [GRIT (ICML 2023)](https://proceedings.mlr.press/v202/ma23c.html)
for evolving pair representations and edge-valued messages, and
[EGT (KDD 2022)](https://arxiv.org/abs/2108.03348) for edge channels and gates.
This implementation combines selected mechanisms; it does not reproduce these
papers, add RRWP/degree encodings, or inherit their benchmark/theoretical claims.
A complete terminal-to-terminal road-cost matrix is not the original street graph.

Physical D/T/E remain authoritative and directed: there is no Euclidean fallback,
no customer-count scaling, and no averaging of forward and reverse roads. Latent
LayerNorm never changes environmental distance/time/energy units. Structural
attention may exchange information between currently infeasible moves; environment
and decoder action masks still enforce feasibility. Invalid costs are excluded
before learned arithmetic; inactive battery/time fields do not affect encoding.

Edge memory is `O(B*N*N*32)`, shared across trajectories. Attention still costs
`O(B*heads*N*N)`; the implementation explicitly forms attention weights and cannot
use the old encoder's fused SDPA path. Edge values aggregate in the small edge
space before node projection, without an `N*N*node_width` value tensor. Zero
dropout and no batch running statistics preserve cached PPO replay. Diagnostics
include graph gradient/update norms, edge gates, edge-update/value magnitudes,
directionality and attention entropy. Old checkpoints cannot initialize this
architecture; use scratch, or a checkpoint with the identical graph profile.

From `CaliRoute/`, run two independent jobs on two GPUs, one task per card:

```bash
bash scripts/run_graph_rdi100_single.sh evrptw --gpus 0 --seed 3011 --epochs 1500
bash scripts/run_graph_rdi100_single.sh vrptw --gpus 1 --seed 3011 --epochs 1500
# Inspect configs without reserving a GPU or starting training:
bash scripts/run_graph_rdi100_single.sh evrptw --prepare-only
```

Single-card defaults are **from scratch, seed3011, batch32, n-traj50, PPO5**,
1500 epochs, LR1e-4, chunk120, expert chunk128, and validation1000/best-of-50 every50
epochs. Each task's global rollout batch is 32 instances (1600 trajectories).
Exploration
uses up to eight instances times eight trajectories every five epochs, retaining
the global 64-trajectory search budget. Each task has its own sampler and archive.
On an RTX 6000 Ada 48GB, an initial full EVRPTW update with batch32/chunk120/
expert chunk128 reached 81.6% peak allocated GPU memory and 82.3% reserved memory,
with finite parameters after the update. VRPTW's first full-route update at
batch32 reached 70.2%; batch40/chunk120/expert chunk128 reached 84.9% allocated
and 88.2% reserved. A batch40 VRPTW run therefore needs `--batch-per-gpu 40`
on **both** graph and current commands; the script default remains batch32.
This calibration covers one update; the separate
two-epoch preflight still checks training, expert updates, search and evaluation.
Peak memory usage differs from sustained memory usage and GPU compute utilization.
Memory usage depends on the task and hardware.
The remote agent can set `--batch-per-gpu`, `--chunk-size`, and
`--expert-chunk-size`. To target over 80% GPU RAM, measure the full preflight's
peak allocation and tune the two chunk sizes first while retaining batch32;
leave headroom for varying routes and expert lengths. A larger batch changes the
training protocol and must be applied to both encoder variants. Match global batch, seed, trajectories,
passes and validation protocol with the current-encoder control; time chunks may
differ to fit memory. For that control use the same shell with
`--encoder-variant current`. Run names and manifests distinguish the variants.
Each background job owns only its selected GPU, performs a full-allocation two-epoch preflight
(with search every preflight epoch and one full batch16/best-of-50 validation), then discards its state and starts formal
training from scratch. Both task commands can run concurrently on their separate
cards; the single-card wrapper rejects a list of two GPUs. No existing local
training job is stopped. The optional `run_graph_rdi100_dual.sh` still supports
one task on a synchronized pair, with batch32 per rank, chunk8 and expert chunk64
by default.

For ordered encoder comparisons, save the printed graph experiment directories
and pass each directory to its current-encoder run:

```bash
bash scripts/run_graph_rdi100_single.sh evrptw --gpus 0 --encoder-variant current \
  --after-run "$EVRPTW_GRAPH_RUN"
bash scripts/run_graph_rdi100_single.sh vrptw --gpus 1 --encoder-variant current \
  --after-run "$VRPTW_GRAPH_RUN"
```

`EVRPTW_GRAPH_RUN` and `VRPTW_GRAPH_RUN` are the paths printed by the first two
commands. `--after-run` may be repeated to wait for both tasks before starting a
second comparison block. It starts only after every prerequisite completes
successfully; a failed or interrupted prerequisite prevents dependent training.
The same option orders dual-card runs on one GPU pair. GPU locks prevent overlap,
while this explicit dependency fixes the order. Use identical seed and global
batch in both variants, including any overrides.

The launcher finds `AAAI_Dataset/dataset` either inside the repository or beside
it; `--data-root /path/to/AAAI_Dataset` overrides
discovery. On this checkout the detected path is `/data/Maojie/AAAI/AAAI_Dataset`.

Local checks include unit/integration contracts, full-size real Cus100 input
forward/backward and actual single-process and two-rank CPU/Gloo training for both tasks, including
20 optimizer updates/rank, expert SL, exploration, normalization, independent
evaluation and checkpoint round trips. The on-server preflight checks the full
configured CUDA allocation; CPU tests do not establish target-GPU capacity or throughput.

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
