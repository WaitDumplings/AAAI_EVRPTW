# VRPTW100 original vs compact P0/P1, 2026-10-09

This comparison returns to the original reward and SL-PPO configuration. Both
models start from scratch, train with pure PPO for epochs 1–100, save their own
`checkpoint_epoch_0100.pt`, and continue with SL-PPO at epoch 101. The same
optimizer/scaler and weights continue across the boundary; no external init
checkpoint is used. Total budget is 1500 epochs, including warm-up.

The control executes the complete original `f388343dbb1d54bbd3f76dd29ca95208070d31a8`
archive with external distributed execution, evaluation and warm-up scheduling.
Archived model/environment/loss source files remain byte-identical. The candidate
uses the current runtime, configured with the original algorithm and these changes:

- P0: inactive resource channels do not enter feature arithmetic, LayerNorm
  statistics, resource attention gates or branch biases. VRPTW energy/charging
  placeholders are ignored; CVRP time placeholders are ignored as well.
- P1 nodes: deterministic outgoing/incoming road profiles (five log-distance
  quantiles, mean, standard deviation, reachability per direction) feed a gated
  residual in the original embedding. No random neighbors or randomized SVD.
- P1 attention: directed D/T/E pair features condition a compact nonlinear
  residual on content scores. Forward/reverse costs remain separate. Graph token
  interactions retain the original form. Each adapter starts with zero output.
- Physical context supplies explicit resource-presence flags and unit metadata;
  equivalent static caches and dynamic projection reuse reduce repeated work.

The profiles borrow the idea of directed road summaries from RRNCO, and the
score interaction borrows RADAR's content/edge combination idea. These are our
implementations, not reproductions or a claim of established SOTA. P1 uses the
original light Transformer, not the separate joint node/edge graph encoder.

Heavy edge-state updates, typed resource fusion, branch exploration, diverse
archive, PopArt and strict-distance reward are disabled in this comparison.
Native original reward/penalties, gamma 0.99, GAE lambda 0.95, SL coefficient 0.5
and original expert/incumbent candidate rules are shared. Warm-up disables all
SL/expert/group/reference advantages and auxiliary losses, not just the SL
coefficient. Expert data may be loaded at startup but do not train the warm-up
policy. Pure-PPO equivalence and the transition are tested for both runtimes.

## Budget and speed controls

| Setting | Both arms |
|---|---:|
| GPUs | 2 per model (4 total) |
| Seed | 3011 |
| Epochs | 100 PPO + 1400 SL-PPO |
| PPO passes | 3 fixed, target KL disabled |
| Instances | 16 per GPU, 32 global |
| Trajectories | 50 per instance (1600 global per rollout) |
| Minibatches | 4, gradient accumulation 1 |
| Optimizer steps | 12 per epoch |
| LR | 1e-4 constant |
| Time chunk / expert chunk | 64 / 128 |
| Model | dimension 256, encoder layers 2 |
| Validation | all 1000 instances, best-of-50, every 50 epochs + epoch 0 |

The smaller batch reduces samples per epoch compared with the preceding global
64 experiment. It is fair between these new arms, but a shorter epoch alone is
not evidence of higher instance throughput. Compare same-epoch quality, total
sample count and wall time. Chunking retains all rollout steps and optimizer
boundaries. LR is held common; monitor KL, clipping and feasibility before tuning.

Candidate observations retain original minmax coordinates and a fixed distance
unit of 43.638668060302734 km (audited from the original VRPTW100 training-set
scale). Distance, time and energy features come from the environment matrices,
not coordinate-derived Euclidean distances. No scale is fitted on val/test or
recomputed for a different customer count. Both arms retain original reward
normalization; this is not the final reward/generalization ablation. Both independent
validation scorers use explicit time matrices; these exactly equal D/v on all
1000 current validation instances. The modern runtime also retains numerically
stable FP32/log-domain SL clipping and nonfinite rejection. Its best checkpoint
prioritizes feasibility, while the original minimizes distance among evaluations
with positive feasibility. Compare matched epochs with matched feasibility; the
experiment is not a byte-identical trainer ablation.

## Launch and monitoring

From `CaliRoute`:

```bash
bash scripts/run_vrptw100_p1_original_dual.sh --gpus 0,1
bash scripts/run_vrptw100_p1_optimized_dual.sh --gpus 2,3
```

Both wrappers launch background supervisors. `--prepare-only` writes a reviewable
configuration. `--batch-per-gpu`, `--chunk-size`, `--expert-chunk-size`, `--epochs`
and `--warmup-epochs` are explicit overrides. Keep rollout and update budgets
identical across arms. Each supervisor freezes current code and data hashes,
locks its GPU pair, runs a two-epoch preflight (PPO 1 + SL-PPO 1), and only starts
the formal scratch run after full preflight completion. Preflight weights,
optimizer state, observations and policy memory are discarded.

```bash
python scripts/watch_comparison.py <original_run_dir> --compare-with <optimized_run_dir> --watch 30
```

The watcher shows the actual phase from training CSV, including the epoch within
that phase. Full results also record rank monitoring, KL/clipping, optimizer and
AMP steps, runtime, memory, learning rate, feasibility and SL candidate counts.
Training success cannot be inferred solely from a snapshot's `running` label.

The old local runs stopped at logged epochs 168/170; epoch-150 periodic
checkpoints remain. Stop evidence and preserved snapshots are stored under
`results/optimization/replacement_audit_20261009` (not version controlled).
