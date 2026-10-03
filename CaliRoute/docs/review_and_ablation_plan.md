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
