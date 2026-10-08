# Physical consistency and exploration comparison

This launcher prepares four independent, single-GPU VRPTW100 runs. Every arm
uses depot-centred fixed-unit inputs, physical input context, and the same full
stage-two embedding/encoder/decoder architecture. `legacy` here means the
previous PPO/AGDA/replay semantics on that architecture; it is not the original
Git baseline.

| Arm | Change from the preceding arm |
| --- | --- |
| `legacy` | Existing gamma 0.99, step-mean PPO reduction, proxy AGDA candidate features and quality archive |
| `physics` | Explicit road matrix contract and optional per-edge T/E, physical candidate states, smooth distance features, distance-only reward, gamma 1, valid-action PPO reduction and truncation bootstrap |
| `archive` | Adds structural archive selection using edge and customer-route partition differences, plus a separate exploration reservoir |
| `explore` | Adds independently sampled prefix branches using the reservoir; search trajectories do not enter PPO as on-policy rollouts |

These are incremental bundles, so a difference between `legacy` and `physics`
does not identify one cause. The comparison holds the combined input and full
stage-two architecture fixed by design; it does not assume either has already
won a controlled experiment.

## Run on a four-GPU server

Place `AAAI_Dataset` beside `CaliRoute`, then from `CaliRoute`:

```bash
bash scripts/run_physics_exploration_comparison.sh --seed 3010 --gpus 0,1,2,3
```

The shell launches a detached supervisor. Each arm waits for an idle GPU, runs
a two-epoch preflight with the formal training batch, and then starts fresh from
the shared initialization. Existing jobs are not stopped. Status, manifests,
configs, hardware samples and supervisor logs are written below
`results/optimization/PHYSICS_EXPLORATION_VRPTW100_*`. Follow a selected run with:

```bash
python scripts/watch_comparison.py results/optimization/PHYSICS_EXPLORATION_VRPTW100_<run-id> --watch 30
```

For preparation without any training:

```bash
bash scripts/run_physics_exploration_comparison.sh --prepare-only --seed 3010 --gpus 0,1,2,3
```

The checkpoint `assets/input_norm/vrptw100_norm_epoch0300.pt` is bundled; no
checkpoint from another server's results directory is needed. Use `--data-root`
for a different dataset directory, and `--init-checkpoint` plus
`--expected-init-epoch` for an explicitly selected initialization. Preparation
freezes code, configs and initialization and records dataset hashes.

The defaults are 300 additional epochs, 64 instances × 50 trajectories per
rollout, 5 PPO passes × 4 minibatches, chunk size 15, constant LR `1e-5`, entropy
coefficient `0.002`, and SL coefficient `0.35`. Full validation uses 1,000
instances, best-of-50 sampling, at epochs 0, 50, 100, 150, 200, 250 and 300.
`--epochs`, `--eval-interval`, `--learning-rate`, `--chunk-size`, and `--seed`
override these controls for all arms. One to four GPUs of the same model may be
selected; fewer GPUs queue arms.

`--target-kl` is omitted by default, retaining five fixed PPO passes. Fresh
post-update KL is monitored every 10 epochs. If an optional threshold is set,
passes may stop early and actual update counts must be compared. This KL uses
sampled replay actions, not a full action-distribution KL; a pass-end check does
not roll back overshoot or provide a hard bound. The preflight evaluates this
path each epoch and enables branch search each epoch in `explore`.

## Physical contract

The input length unit and reward unit are separately configured and stored in
checkpoint metadata. This experiment fixes both at the initialization's
43.638668060302734 km. They are not refit to Cus15, Cus50 or another city during
evaluation. Gamma is fixed to 1 for the physical arms because the objective is
finite-episode total distance. GAE lambda remains an independent estimator
parameter, initially 0.95. Customer count never multiplies a step reward or its
unit: the same physical edge has the same reward at Cus15 and Cus1000. A longer
solution may legitimately have a larger total distance/return; fixed units do
not assert that total returns have identical distributions across sizes.

Strict road mode requires an explicit `distance_matrix_km`. A provided matrix is
not by itself proof of its road provenance: the dataset metadata and generator
remain responsible for that evidence. Missing matrices and declared Euclidean
metrics are rejected. Explicit `travel_time_matrix_s` and `energy_matrix_kwh`
are preserved and used when enabled, with shape, reachability and nonnegativity
checks. If absent, travel and energy use the original D/v and D×consumption
model. New flags are off by default for historical configurations; full resume
rejects changed physical/model feature semantics, while weights-only migration
records them.

Distance-only feasible returns equal minus physical route kilometres divided by
the fixed reward unit. A failed termination receives an explicit 1,000 km guard
in the same unit. This number is an experiment setting, not a proof that every
possible infeasible partial route is worse than every feasible route at every
size. PBRS and terminal heuristics are disabled in all four arms. This particular
full-route PopArt training protocol stops before an update if any on-policy
trajectory is infeasible or incomplete; the environment penalty is not an
implicit authorization to learn from cheap partial solutions. The general
collector correctly bootstraps sampling-budget cutoffs, while true environment
failures remain terminal.

AGDA's physical features use the shared candidate-transition implementation,
including charging and depot resets and service-start time windows. Feature
width remains 30, but its semantics change behind a checkpointed flag. Distance
features use smooth signed log compression rather than clipping all values above
2 to the same value.

## Interpretation and monitoring

The initialization is the prior Norm epoch-300 checkpoint, followed by fresh
optimizer, RMS, PopArt and replay states. Updating existing AGDA features can
change the initial policy even when weights are identical: report epoch-zero
migration cost. Only the `physics`/`archive`/`explore` initial validation pairs
are expected to match. This is a fine-tuning screen, not equal-budget training
from scratch.

Archive staleness is measured from ingested observations. It is not the number
of optimizer epochs or the time since every possible solution was explored.
Three stored routes do not establish diversity by themselves; inspect actual
structural distances, unique solutions and new best discoveries. The exploration
reservoir is separate from quality-constrained imitation targets. A diverse
but poor route can seed search without becoming a positive imitation target.

`explore` adds at most 8 selected instances × 8 trajectories every 5 epochs,
with temperature 1.2 and mixed prefix fractions. Search first revisits stagnant
reservoir instances from the registered training pool, even when they are absent
from the current PPO batch; current instances fill the remaining budget. The
`exploration_prefer_stagnant_archive` flag controls that choice. Report requested/completed
search trajectories, search time, successful candidates, archive improvements,
PPO update counts and validation distance together. Equal PPO epochs are not
equal compute for this arm. Compare matched validation epochs and wall-clock
budgets, repeat across seeds, and do not use test data to choose parameters.


The new PPO reduction averages valid actions within each optimizer minibatch;
`gradient_accumulation_steps=1` is required for this release. On-policy PPO and
solution-level loss share historical actor RMS; the critic uses PopArt. Expert
and replay gates remain bounded, dimensionless auxiliary weights, not members
of the on-policy group or its normalization statistics. The extra search pool
never receives a positive imitation target merely for being diverse.

`monitor_rank_*.jsonl` records reward components and physical return error,
actor/critic scales, valid-action counts, group cost spread, structural diversity,
best-of-10 to best-of-K gain, search discoveries and time. On diagnostic epochs,
expert/replay gradients are compared with the complete first PPO minibatch on
the same action-query head (after configured loss coefficients, before gradient
clipping). These sampled head-level ratios/cosines identify possible auxiliary
conflicts; they do not represent a full-network gradient measurement. The tracker
also distinguishes training-aggregate KL from freshly recomputed pass-end KL.


Local validation on a single RTX 2080 Ti completed two full-batch epochs
(64 × 50 trajectories, five PPO passes) at chunk 15, including search and fresh
KL checks. Peak framework-reserved memory was approximately 9.1 GiB, about 85%
of the usable card capacity. Chunk 18 ran out of memory. Other servers still run
their own preflight; reduce `--chunk-size 12` consistently across all arms if a
server has less free memory. This is a memory/pipeline check, not accuracy evidence.
