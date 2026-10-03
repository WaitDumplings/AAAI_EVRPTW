# CaliRoute Review And Ablation Plan

## Engineering Review

CaliRoute should present a small public surface:

```text
train.py -> caliroute.cli -> caliroute.config/methods -> offline2online.trainer
```

The public layer owns method names, data paths, and ablation switches. The
`offline2online` package is treated as the stable training engine. This keeps the
paper repository easy to run without rewriting the tested rollout/update code.

Current improvements:

- Method presets are centralized in `caliroute.methods`.
- The repository layout is `AAAI_EVRPTW/{CaliRoute,AAAI_Dataset}`; the dataset is
  excluded from Git and will be provided through a separate download link.
- Training and validation resolve from `../AAAI_Dataset/dataset` relative to the
  CaliRoute code root by default. The frozen final test set lives separately in
  `../AAAI_Dataset/test_release`.
- SL-PPO public configuration uses descriptive names such as
  `sl_expert_candidate_weight`.
- The old priority sampler name has been replaced by
  `SolutionPrioritySampler`.
- DDE, dynamic action key/bias, Q/K/V deltas, and encoder distance bias are CLI
  switches.
- Historical one-off YAML, launch, plot, and watcher scripts have been removed
  from the public CaliRoute tree. Keep any old working project outside this
  repository if archival comparison is needed.

Remaining cleanup after validation:

- Split the monolithic trainer into `ppo_loop.py`, `slppo_loss.py`,
  `baseline_losses.py`, and `evaluation.py`.
- Keep only PPO, SL-PPO, DAPG, and AWBC in the public training path.
- Move historical or unused experimental branches behind a legacy module or
  remove them after the 20-epoch regression check passes.

## Academic Review

The paper story should be:

```text
PPO learns online from rollout feedback.
SL-PPO adds solution-level supervision from historical/expert solutions.
DAPG and AWBC are offline demonstration baselines on the same PPO backbone.
```

SL-PPO should be described at the complete-solution level:

- The advantage is solution-level, using group-relative and reference terms.
- The policy ratio is length-normalized over the generated solution actions.
- The incumbent/reference mechanism prevents the model from being bounded by a
  suboptimal expert.
- The optional expert-candidate term keeps high-quality historical solutions in
  the comparison set without turning the method into pure imitation learning.

## Ablation Matrix

Recommended controlled ablations:

```text
Backbone:
  PPO only
  PPO + DDE
  PPO + DDE action key
  PPO + DDE action key + action bias

Distance:
  encoder distance bias on
  encoder distance bias off

SL-PPO:
  group advantage only
  group + reference advantage
  priority pool weighted
  priority pool best
  expert-candidate weight {0.0, 0.3, 0.6}
```

The validation run should be completed before removing more legacy branches so
that the first-20-epoch behavior can be compared against the previous seed-3009
EVRPTW Cus50 runs.

## PPO initialization and optimization study (2026-10-02)

The pre-optimization code is pinned by tag `baseline/slppo-preopt-20261002`
(commit `b401e11518711b920fa78681260e688da014e98f`). Work proceeds on
`opt/slppo-design-20261002`. Check out the tag when cloning the old comparison
version; checkpoints and datasets must be retained separately from Git.

The first shared initialization is CVRP Cus50, seed 3009, trained with PPO for
100 epochs: 64 environments, 50 trajectories, 90 rollout steps, 3 PPO update
passes, step chunks of 16, 4 minibatches, learning rate 1e-4, and mixed precision.
Validation uses 1,000 validation instances with 50 trajectories every 20 epochs.
Both SL-PPO versions should start from the same epoch-100 weights. A three-epoch
speed smoke run is a separate experiment and is not the shared initialization.
Run configurations, source provenance, process status, and console logs are saved
locally under `results/optimization/`; training logs and checkpoints retain the
standard `results/logs/` and `results/checkpoints/` layout.

### Implemented: static encoding during rollout

The offline-to-online graph encoder is deterministic and depends only on static
instance data. Training rollouts and all evaluation entry points now encode once
after each reset and decode the updated dynamic state and action mask at each step.
Caching requires an explicit backbone capability flag. The legacy TERRAN encoder
has training-time dropout and keeps its existing behavior. Cached no-grad outputs
are never passed into PPO gradient updates or reused after another reset.

Equivalence checks cover logits, critic values, complete greedy/sampled trajectories,
DDE switches, reset boundaries, encoder gradients during policy updates, and the
legacy fallback. Compare measured model-forward time and complete epoch time
separately; reducing encoder calls does not imply an equal total-training speedup.

### Next controlled design experiments

- Learn residual per-head edge biases from directed road distance, travel time,
  energy, and static time-window compatibility. Zero-initialize the added output
  layer so an existing PPO checkpoint initially retains its outputs.
- Give EVRPTW's dynamic decoder explicit post-charge departure time and battery
  features, followed by graded margins to a reachable depot or charging station.
  The environment already enforces feasibility; this experiment changes the
  policy's planning features. Validate them against actual environment transitions.
- Retain a small, diverse, train-only pool of verified policy-improved routes per
  instance alongside the original expert. Existing memory stores best objective
  values, not their trajectories. Recompute current-policy reference logprobs
  before replay and limit replay frequency to retain exploration.
- In SL-PPO updates, reuse the PPO forward pass for the solution-level auxiliary
  objective. Verify clipping and chunk normalization with gradient comparisons
  before changing the update path. Profile static observation transfers and
  default DDE projections before optimizing further.

Before SL-PPO fine-tuning, fix the expert-weight configuration wiring, empty-loss
backpropagation, feasibility-aware checkpoint selection, and evaluation gap
filtering identified in the code review. Apply the same evaluator to old and new
models. Tune on train/validation only; frozen test evaluation follows model and
hyperparameter selection. Model-design benefits remain hypotheses until controlled
quality/feasibility experiments are complete.

## Controlled design implementation and two-GPU comparison

The optional designs are now exposed by `--optimization-profile optimized`.
`--optimization-profile baseline` disables all optional architecture, replay,
and speed changes while retaining the shared correctness fixes. Neither profile
changes the PPO/SL-PPO method itself. Individual YAML settings remain available
for later ablation. EVRPTW post-charge features are experimental: synthetic
transition tests do not establish performance on real EVRPTW data.

The optimized profile adds a zero-output directed per-head edge adapter;
post-charge action bias for EVRPTW only; active DDE projection slices and cached
static node projections/observations; shared PPO/SL forward evaluation; static
expert-route encoding; and a bounded policy-route pool for SL-PPO only. A PPO
initialization can use this profile without enabling the SL-only replay loss.
The pool retains at most three diverse, verified routes per training instance,
within 5% of its best retained route; its default replay budget is at most 25%
of current environments (capped at 16 candidates), with a separate weight of
0.2. Original expert weighting and normalization remain unchanged. Pool actions
are independently replayed and current-policy log probabilities are recomputed
before use. No validation or test route enters training memory.

Both comparison arms use the epoch-100 CVRP50 PPO checkpoint trained from the
frozen original code. The planned first comparison is 100 SL-PPO epochs, seed
3009, 64 environments x 50 trajectories, four PPO updates, learning rate 5e-5,
and validation on 1,000 instances with 50 samples before training (epoch 0) and every 20 epochs. This is a
single-seed screening experiment, not a multi-seed improvement claim. The
baseline includes the same expert-weight wiring, empty-loss safety,
feasibility-first checkpoint selection, and independent evaluation RNG fixes as
the optimized arm. It is explicitly not an untouched historical-code run.

From `CaliRoute`, launch both processes in the background with:

```bash
bash scripts/run_slppo_comparison.sh \
  --init-checkpoint /absolute/path/to/checkpoint_epoch_0100.pt \
  --data-root /absolute/path/to/AAAI_Dataset/dataset \
  --problem cvrp --customers 50 --gpus 0,1
```

Commit the source before launch; use a detached checkout of that commit to keep
running experiments isolated from later edits. The launcher rejects an existing
run directory and never overwrites a previous experiment. `--dry-run` prints
both configurations without starting jobs. `--optimization-profile` on
`train.py` is the separate single-run interface. Default methods remain backward
compatible unless a profile or individual flag is supplied.

`results/optimization/<run>/` contains `manifest.json` (source commit and input
SHA256 checksums), `status.json`, per-arm YAML/console logs, and an automatically
updated `comparison.json`. The report pairs the same validation instance IDs at
the same epoch and reports feasibility separately from distance on jointly
feasible instances; failed partial routes cannot masquerade as short solutions.
Matched-epoch training time excludes initial warmup and validation and includes
the cost of the added design and route replay. Exported CVRP solutions undergo
an independent customer-coverage, depot, capacity, and matrix-distance check.
Gurobi TIME_LIMIT values remain incumbent references, not certified optima.

Validation before the full run: all 118 tests passed, including shared-loss and
expert-cache gradient comparisons, zero-init compatibility, replay rejection
cases, evaluation RNG isolation, and configuration checks. A real CVRP50
three-epoch GPU0/GPU1 integration smoke used eight train and eight validation
instances: initial validation distance matched exactly, both arms completed,
and the optimized arm used four historical replay candidates on epoch 3. The
smoke is a functional check, not an estimate of generalization improvement.
