# VRPTW100 comparison from random initialization

The current comparison entry point is `scripts/run_scratch_comparison.sh`.
All four arms start from random model initialization. No PPO initialization,
previous best checkpoint, or bundled weight asset is loaded. Training expert
routes remain part of SL-PPO; starting from scratch does not remove supervision.

The `legacy` arm runs the complete original `CaliRoute` source from commit
`f388343dbb1d54bbd3f76dd29ca95208070d31a8`. The other arms run the current source.
The launcher archives the original files separately, snapshots the current
files, and records their hashes. Turning off current feature flags is not used
to construct the original baseline.

## Start a four-card comparison

From `CaliRoute/`, after pulling the experiment code and activating the intended
Python environment:

```bash
# Four independent single-GPU arms; the supervisor runs in the background.
bash scripts/run_scratch_comparison.sh --seed 3010 --gpus 0,1,2,3

# Inspect a separate prepared experiment without starting training.
bash scripts/run_scratch_comparison.sh --prepare-only --seed 3011 --gpus 0,1,2,3
```

The shell prefers `../.venv/bin/python` when present, otherwise `python3`.
Set `PYTHON_BIN=/path/to/python` to select another environment. Python invoked
directly prepares by default; add `--launch` to start its supervisor.

Each arm waits for an idle selected GPU and runs a separate two-epoch preflight.
The preflight uses the formal 64-instance, 50-trajectory, five-pass training
shape; its validation is reduced to four instances and four trajectories. The
exploration arm exercises search every preflight epoch. All resulting weights,
optimizer state, archives and running statistics are discarded before formal
training starts. A preflight failure prevents that arm from starting its formal
run. The launcher does not stop other jobs.

CPU cold-start spot checks covered eight instances with 100 uniform/depot-first
trajectories and a 16-trajectory randomly initialized model sample; the audited
samples were feasible. These checks do not establish full-batch GPU readiness.
The launcher must still complete each arm's two-epoch GPU preflight before its
300-epoch formal run.

One, two or three selected GPUs queue the same four arms. Each arm still uses
one GPU; this launcher does not combine cards into a larger distributed batch.
Compare speed on the same GPU model. Replicate on another server by using the
same source and dataset with another seed.

```bash
# Examples of explicit controls.
bash scripts/run_scratch_comparison.sh --seed 3012 --gpus 0,1 \
  --epochs 300 --eval-interval 50 --chunk-size 12 --legacy-chunk-size 8

# data-root is the AAAI_Dataset directory, not its dataset child.
bash scripts/run_scratch_comparison.sh --prepare-only \
  --data-root /path/to/AAAI_Dataset
```

Other controls include `--learning-rate`, `--arms legacy,physics,archive,explore`
and a fresh `--run-id`. A run ID cannot overwrite an existing experiment. There
is no initialization-checkpoint argument and no target-KL argument in this
scratch launcher. Five PPO passes are fixed for all arms.

## Data and common training budget

Place `AAAI_Dataset` beside `CaliRoute`, or pass `--data-root`. Required files are:

```text
AAAI_Dataset/dataset/vrptw/train/Cus100/instances.pkl
AAAI_Dataset/dataset/vrptw/train/Cus100/expert_solutions.csv
AAAI_Dataset/dataset/vrptw/val/Cus100/instances.pkl
AAAI_Dataset/dataset/vrptw/val/Cus100/gurobi_summary.csv
```

The intended split contains 5,000 training instances and 1,000 validation
instances. Preparation checks the required files and records their hashes,
including `metadata.json` when present. It does not require another machine's
`results/` directory or any checkpoint in `assets/`. The original commit must
be available in local Git history; fetch the full history if a shallow clone
omits it. Test data are reserved for final evaluation.

| Setting | All four formal arms |
| --- | --- |
| Initialization | Random from the run seed; fresh optimizer and policy memory |
| Training duration | 300 epochs |
| Sampling | Uniform `shuffle_cycle` |
| Rollout | 64 instances × 50 trajectories, at most 201 actions |
| PPO updates | 5 passes × 4 minibatches: 20 optimizer-step attempts per epoch |
| Learning rate | Constant `1e-4` |
| Entropy / SL coefficient | `0.01` / `0.5` |
| GAE lambda / PPO clip | `0.95` / `0.2` |
| PBRS | All four shaping switches disabled |
| Validation | All 1,000 instances; sample best-of-50; batch size 32 |
| Validation epochs | 0, 50, 100, 150, 200, 250, 300; final epoch also retained after a custom horizon |
| Periodic checkpoints | Every 50 epochs and at the end |

The legacy PPO time chunk defaults to 8; current arms default to 15. These
control memory use without changing the number of instances, trajectories,
minibatches or PPO passes. Legacy retains its original loss reduction, so the
comparison does not claim identical floating-point calculations or loss
weighting. Reduce the appropriate chunk if the server's preflight runs out of
memory. Legacy also uses `--legacy-expert-chunk-size 128`, recorded as an
explicit change from the original fallback of 4096. The historical SL candidate
path reads `advantage.sl_expert_logprob_chunk_size`; the launcher writes that
key and its `offline` counterpart. All candidate routes, all route steps and
loss weights are retained. This limits each expert re-encoding batch and its
backward workspace. It was added after a 2080 Ti preflight ran out of memory
in the original expert-loss backward, despite PPO time chunk 8.

This document does not assert that every hardware configuration has already
passed the scratch preflight.

## What each arm contains

| Arm | Implementation |
| --- | --- |
| `legacy` | Exact original source: per-instance axis min-max input coordinates, training-set D0 shared by input distance and reward, original RDI/DDE, original SL-PPO advantages and step-mean PPO loss, gamma `0.99` |
| `physics` | Current combined physical inputs and embedding/encoder/decoder; strict road-matrix handling and explicit edge T/E support; physical AGDA candidate states and smooth distance features; shared actor RMS/PopArt and physical-cost SL; gamma `1`, valid-action PPO reduction and truncation bootstrap |
| `archive` | `physics` plus structural archive selection and a separate exploration reservoir |
| `explore` | `archive` plus independently sampled prefix-branch search |

The new input and reward units are fixed at `43.638668060302734` km in this
protocol and are not refit for validation. Keeping this physical unit is a
configuration choice; it does not load a trained normalization checkpoint.
The original arm continues to fit its native D0 from training data. Its old
coordinate representation, reward logic, optimizer implementation and losses
are preserved in the archived code.

Both the original public method and the current experiment use embedding size
256, two encoder layers, DDE heads 4, disabled delta-K/delta-V, and enabled
action-key/action-bias changes. The newer physical interfaces, relational
encoder and resource decoder are additional model changes. Consequently,
`legacy` versus `physics` compares complete implementation bundles, including
normalization and training semantics; it is not a single-factor architecture
ablation.

The original public SL-PPO preset has four PPO passes and weighted-priority
sampling. This comparison explicitly overrides them to **five passes and
uniform sampling** in every arm. Changes to epoch count, rollout horizon,
instance batch, evaluation schedule/batch and chunk size are also recorded in
`legacy/original_provenance.json`. The default LR `1e-4`, entropy `0.01`, and
SL coefficient `0.5` already match the original public preset. This is an
original-source comparison under declared experimental controls, rather than
an exact reproduction of the historical PPO-init-then-SL-PPO experiment.

`explore` adds at most 8 selected instances × 8 branch trajectories every five
epochs, using temperature 1.2. Search can revisit stagnant archive instances
outside the current PPO batch. Those trajectories are not inserted into the
on-policy PPO rollout. Equal PPO budgets therefore do not imply equal compute;
record completed search trajectories, search time and total wall time.

## Original evaluation adapter

`scripts/run_original_scratch.py` imports the archived model, trainer and
environment, checks their module origins, and rejects checkpoint loading.
An external measurement adapter supplies fixed evaluation RNG, epoch-zero
measurement, route exports and independent validation. Archived source files
are unchanged; the adapter's scope and hashes are recorded separately.

Evaluation uses `17000000 + run_seed`, isolated from training RNG. The adapter
cancels the original evaluator's epoch-dependent seed offset and restores
Python, NumPy, Torch and model train/eval state afterward. Thus changing the
validation schedule does not intentionally consume training randomness.

For each legacy instance, the adapter retains the original choice of minimum
route among environment-successful trajectories. It then independently checks
that selected route, including customer coverage, capacity, directed distance,
time windows, service times and return to depot. A raw explicit travel-time
matrix is loaded into a validation-only sidecar when the old adapter dropped
it. This does **not** change the original model input, training transitions,
feasibility masks or decoding environment; legacy still uses its original
D/v dynamics. If the selected solution fails independent validation, its
failure is reported rather than selecting another trajectory silently.

The legacy aggregate feasible rate requires both environment success and
independent validity of the selected route. Legacy trajectory-feasibility and
median-distribution statistics still use the old environment's definition;
the adapter does not independently validate all 50 sampled trajectories.
Check `eval_environment_feasible_rate`, `eval_independent_valid_rate`, the
per-instance route records and `eval_feasibility_source` when interpreting
these metrics. Distance means use feasible instances, so always report their
coverage as well as kilometres.

Epoch-zero results are measurements of random initialization. They are written
for every arm, but **epoch zero is not eligible for best-checkpoint selection**
in either implementation. Different architectures need not produce identical
policies under the same seed. Only
`physics`/`archive`/`explore`, which share an architecture, are expected to agree
at epoch zero; these pair checks are recorded in the comparison.

Native best-checkpoint rules differ. The original trainer accepts any positive
feasible rate and minimizes mean distance over feasible instances; the current
trainer first maximizes feasible rate, then minimizes distance. Compare full
validation at the same epochs first. If every compared checkpoint is not fully
feasible, the two native `checkpoint_best` files are not a fair selected-best
comparison. Select again from the saved 50-epoch checkpoints with one common
feasibility-first rule before reporting a best-versus-best result. The original
trainer's rule remains unchanged.

## Track and interpret the results

The launcher prints a new directory below
`results/optimization/SCRATCH_ORIGINAL_VRPTW100_*`. Use that exact path:

```bash
python scripts/watch_comparison.py results/optimization/<run-directory> --watch 30
python scripts/plot_reward_norm_eval.py results/optimization/<run-directory>
```

`manifest.json` records the source snapshots, dataset/config hashes, commands
and budgets. `status.json`, `comparison.json`, `hardware.jsonl` and
`supervisor.log` record progress. Each arm has its `config.yaml`, isolated
`preflight/` and `evaluations/epoch_XXXX.jsonl`. Original-only records include
`original_provenance.json` and `original_adapter.json`. Training logs and
checkpoints are stored below the usual `results/logs/Cus_100_CS_0/` and
`results/checkpoints/Cus_100_CS_0/` directories.

The current arms also provide monitoring JSONL for reward/physical consistency,
actor and critic scales, structural diversity, exploration, gradient diagnostics
and fresh post-update KL. The original trainer does not have those new
instruments. Missing measurements must remain missing, never filled with zero.
It also has no `distributed_train_wall_time_s`, so derived
`median_training_seconds_per_epoch` can be null for legacy. Compare speed using
common stage wall times and hardware records; do not treat a missing per-epoch
time as zero or compare incompatible timer definitions.
The tracker's `Train KL` is an aggregate measured during PPO updates; `Fresh KL`
is recomputed using the updated policy on sampled rollout actions. Fresh KL is
normally sampled every ten epochs for current arms and is not a hard bound or
a legacy measurement. `--` means unavailable for that row.

Compare the same completed validation epochs, feasible coverage and wall time.
Use repeat seeds before selecting a configuration. Neither a single seed nor a
win between complete bundles isolates the cause of improvement or establishes
cross-size/cross-city generalization. The previous warm-start experiment remains
documented in [the historical comparison guide](physics_exploration_comparison.md).
