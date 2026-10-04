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

## Portable RDI / AGDA / SLPPO and long dual-GPU study (2026-10-04)

The user-confirmed names are RDI for road-distance/edge injection and AGDA for
candidate-side dynamic graph adaptation, previously called DDE in the code.
The portable entry points live in `caliroute/plugins/` and use tensors rather
than routing environments, datasets, or trainer objects:

| Component | Portable contract | Host responsibility |
| --- | --- | --- |
| `RoadDistanceInjection` | Directed `[B,N,N]` road distance and optional travel time, energy, windows, service and capacity tensors produce `[B,H,N,N]` additive attention bias. | Preserve the true road metrics; depot is node zero; apply authoritative hard masks and handle graph tokens. |
| `AdaptiveGraphAttention` | Node embeddings `[B,N,D]`, decision tokens `[B,T,S,D]`, candidate features `[B,T,N,F]` and system features/state tokens produce dynamic key, value, action-key and scalar action-bias residuals. | Construct problem-specific features and summary tokens, then insert the returned residuals into the host decoder. |
| `solution_level_ppo_loss` / `clipped_route_surrogate` | Selected-action log probabilities, valid-step masks, detached route advantages and feasibility produce the clipped length-normalized route surrogate, reusable gradient weights and diagnostics. | Collect trajectories, retain old-policy likelihoods, define meaningful objectives/references, and perform optimizer updates. |

AGDA's complete attention/fusion/projection core is portable; only routing feature
construction remains in `DynamicGraphKVEncoder`, which inherits that core. The
extraction retains all 56 legacy DDE state keys and their same-seed initial
values. `AdaptiveGraphDecisionAdapter` is the optional new candidate-conditioned
gate inside this core, not the entirety of the exported AGDA component.

The `optimized_v2` profile adds a bounded RDI residual using separate physical
scales, directional differences, outgoing-distance ratios and depot detour savings;
it replaces the earlier residual-edge branch rather than stacking both. RDI is
zero initialized. AGDA gates initialize to one and remain between 0.5 and 1.5,
preserving pretrained dynamic residuals initially. Default sizes add 896 RDI and
1,184 gate parameters. Standard encoder attention can use SDPA; the decoder's
learned-scale/tanh attention is not replaced. FP32/AMP SDPA outputs and gradients
are numerically close, not bitwise identical. A controlled 2080 Ti microbenchmark
reduced encoder forward/backward time by approximately 7.5–10.8%; it excludes the
decoder, environment, optimizer and communication, and is not an end-to-end speed
claim.

SLPPO v2 uses a reference-relative advantage standard-deviation floor of 1%,
instead of a fixed distance-unit floor. Verified historical route replay collects
memory during the first 25 epochs, ramps its loss weight over the next 75 epochs,
and reaches 0.10 at epoch 100. A replay route must improve the current sampled
best distance by at least 0.2%; collection, route feasibility checks, bounded pool
size and diversity filtering remain train-only. These are tuning hypotheses;
neither relative normalization nor additional replay guarantees shorter routes.

### Controlled long-run protocol

The planned first long study is CVRP50, seed 3009, 1,000 epochs. Both arms start
from the same completed original PPO epoch-100 checkpoint and share the corrected
loss/evaluation behavior. The baseline uses the original RDI/AGDA/SLPPO designs
with optional optimization branches disabled; the optimized arm uses
`optimized_v2`. Both include the same distributed orchestration and common safety
fixes. The baseline is therefore not a byte-for-byte historical executable.

| Setting | Both arms |
| --- | --- |
| GPUs | Two synchronous ranks per model; baseline physical GPUs 0,2 and optimized 1,3. Both pairs cross the same SYS topology class on this machine. |
| Rollout instance batch | 64 environments per rank, 128 globally; 50 trajectories per instance, 6,400 trajectories per global rollout. |
| PPO update | Four minibatches per rank, four PPO passes, step chunks of 16, clipping coefficient 0.2. Without additional accumulation each optimizer step uses 16 instances per rank / 32 globally. |
| Learning rate | Peak `5e-5 * sqrt(2) = 7.0710678119e-5`; linear warmup for 20 epochs, then cosine decay to `1e-5` at epoch 1000. |
| Entropy coefficient | Cosine decay from 0.01 to 0.002 over the training horizon. |
| Validation | Same 1,000 validation IDs and fixed evaluation seed; 50 sampled routes per instance, best feasible distance; epoch 0 and every 20 epochs, plus the final epoch. |
| Monitoring | Basic records each epoch; more expensive critic/gradient/plugin/synchronization diagnostics at epoch 1 and every 20 epochs, written separately for each rank. |
| Mixed precision | Shared initial gradient scale 4096, then standard dynamic scaling; the real four-rank smoke with the default 65536 initially skipped four updates per arm before recovering. |
| Checkpoints | Every 50 epochs and the final epoch, with rank-local RNG/sampler/replay/scaler state where supported. |

Here an epoch is one global rollout plus its PPO update passes, not a complete
pass through all 5,000 training instances. Independent rank seeds prevent the
ranks from replaying identical stochastic streams; they do not guarantee unique
instance IDs across ranks. Log the observed global unique/duplicate counts rather
than equating global batch size with unique-instance coverage. The synchronized
objective is the mean of rank-local masked objectives; it is not a pooled mean
weighted by the total number of valid tokens across ranks. The larger-batch LR
rule is a conservative experimental setting, not a universal PPO scaling law.

Launch from an immutable, committed detached checkout, with the shared results
location and an explicit absolute data path:

```bash
bash scripts/run_plugin_comparison.sh \
  --init-checkpoint /absolute/path/to/checkpoint_epoch_0100.pt \
  --data-root /absolute/path/to/AAAI_Dataset/dataset \
  --problem cvrp --customers 50 --epochs 1000 \
  --baseline-gpus 0,2 --optimized-gpus 1,3 \
  --num-envs-per-gpu 64 --n-traj 50 --num-minibatches 4 \
  --eval-interval 20 --monitor-interval 20 --checkpoint-interval 50
```

The manifest records source/input/checkpoint/config hashes, device topology and
batch/schedule semantics. `comparison.json` updates paired validation and matched
epoch timings; `hardware.jsonl` records utilization, memory, temperature and
power. Each arm has `monitoring/monitor_rank_0.jsonl` and
`monitoring/monitor_rank_1.jsonl`. `scripts/plot_plugin_comparison.py` exports
`training_curves.png` and `.svg` from those logs automatically as evaluations finish.
The wall-time curve uses `run_elapsed_seconds`, measured from the training entry
point and including initialization and epoch-0 evaluation. This counter resets
on resume; `run_session_id` distinguishes sessions. The separately exported
cumulative epoch-time field excludes startup and must not substitute for it. Matched training timings use the slowest rank and exclude LR warmup
and validation epochs. Report both matched-epoch and equal-wall-time quality,
with observed monitoring/communication overhead included.

### Reading the monitors and selecting the next fine-tuning experiment

- Quality: prioritize feasible coverage, then distance on matching feasible
  instance IDs, paired win/tie/loss counts, and median/tail gap statistics. An
  average over only successful routes is insufficient if coverage changes.
  Gurobi TIME_LIMIT incumbents are references, not certified optima; existing
  distance-gap fields use distance units unless explicitly named as percentages.
- PPO stability: inspect approximate KL, clipping fraction, entropy, effective
  LR, unclipped gradient norms, actual optimizer-step counts, AMP scale and skipped
  steps. A KL threshold of 0.02 is a monitoring alert, not an automatic early-stop
  rule. Persistent large KL/clipping or AMP skips motivates reducing peak LR or
  update passes before increasing adapter capacity. Early entropy collapse
  motivates a higher entropy floor or a slower decay.
- Critic and route signal: compare explained variance/RMSE and step-advantage
  moments with group/reference/candidate/combined route-advantage distributions,
  route likelihood-ratio quantiles, route KL and active/clipped route fractions.
  A repeatedly saturated or nearly inactive SL signal motivates adjusting the
  relative floor, SL coefficient or clipping, using validation rather than loss
  magnitude as the selection criterion.
- RDI/AGDA adaptation: inspect gate mean/std/deviation/saturation, bounded RDI
  residual norm relative to the base bias, feature ranges, and sampled module
  parameter/gradient/update norms. Zero output at initialization is intentional.
  Persistent zero updates, gate saturation or a rapidly dominant residual calls
  for checking gradient flow or reducing adapter scale/LR before adding layers.
- Expert/replay balance: inspect active expert/replay fractions, verified and
  diversity-filtered pool coverage, replay advantage/log-probability support,
  effective scheduled weight and improvement-gate rejection rates. Sparse useful
  replay suggests examining coverage and the 0.2% quality threshold; quality
  stagnation with dominant replay suggests reducing its weight/fraction or
  lengthening warmup. Do not relax feasibility checks to increase pool size.
- Gradient interaction: sampled PPO-versus-SL gradient norms and cosine similarity
  describe the first update's first chunk on a common action-query head subset.
  PPO here includes policy/value/entropy; the sampled SL term excludes experts
  and replay. These diagnostics are directional evidence, not the full-model
  gradient balance. Route diagnostic quantiles are means of minibatch quantiles,
  not global epoch quantiles; inspect both rank logs when diagnosing imbalance.
- Systems: compare model rollout, environment, PPO update, gradient synchronization
  and evaluation times, memory peaks, global throughput and sample coverage.
  Parameter sum/squared-norm checksums, scaler and optimizer counters should
  agree across ranks; checksums are not an elementwise equality proof. Low GPU
  utilization with expensive environment/synchronization time suggests a pipeline
  issue, while high update time points to repeated encoder/decoder work.

The first run is an all-on, single-seed screening experiment. Follow it with
speed-only, RDI-only, AGDA-only and SLPPO-normalization/replay ablations under the
same global batch and optimizer schedule; otherwise architectural gains cannot
be separated from optimization-budget changes. Select schedules and checkpoints
using validation, repeat promising settings across seeds, and only then evaluate
the frozen test split. A tiny unrelated attention backbone validates the portable
tensor contracts and gradient flow; no experiment on a second real routing model
has been completed. EVRPTW post-charge integration remains validated only on
synthetic transition tests, and this launcher intentionally admits CVRP/VRPTW
only. No real-data EVRPTW speed or quality improvement is established here.

The pre-launch four-rank CVRP50 integration check used eight real train and eight
validation instances for three epochs, with a shortened replay warmup solely to
exercise replay. Both arms completed with finite later updates and synchronized
parameter checksums. A real two-rank epoch-2-to-3 resume restored sampler, RNG,
replay, scaler and update counters exactly, and reproduced validation routes.
Parameters differed by at most `3.70e-6`; CUDA execution is not claimed to be
bitwise reproducible. These checks establish functionality, not accuracy gains.
Local evidence is under `results/optimization/PLUGIN_DUAL_SMOKE_20261004/`,
including `resume_check/result.json`.


### Recovery after the first long-run interruption

The first 1,000-epoch job stopped before baseline epoch 20 completed; optimized
reached epoch 25 and was terminated by fail-fast. Its last saved checkpoint and
validation were epoch 20. These are completed-epoch counts, not a 1,000-epoch
result. Baseline had no saved training checkpoint.

The baseline gradient monitor retained an extra SL computation graph from epoch
1 until the next diagnostic at epoch 20. Allocating another graph exceeded the
2080 Ti memory capacity. Diagnostics now extract detached common-head gradients
from the normal PPO and SL forwards, without an extra forward or long-lived
graph. Tests verify identical training gradients/updates with monitoring enabled
and disabled and prove intermediate activations are released. Old memory/timing
measurements involving this retained graph must not be used as the final speed
comparison.

Rolling recovery checkpoints now save every five completed epochs before
validation, independently of the every-50-epoch archive. The atomic latest file
contains each rank's training state and explicitly marks pending evaluation.
The recovery experiment retains the failed run as evidence, restarts baseline
from the original PPO init, and resumes optimized from epoch 20. Imported history
and checkpoint provenance are recorded in its manifest. Progress reports separate
target epochs, completed training epochs, latest validation epoch, and exit state.
Across resumed sessions, plotted time is cumulative recorded active-session
time; it excludes downtime, discarded work, and unrecorded session tails.
