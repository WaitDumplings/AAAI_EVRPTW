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
