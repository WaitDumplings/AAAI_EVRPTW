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

### CVRP100 / VRPTW100 parameter screening and long runs

`scripts/run_cus100_finetune.sh` runs CVRP100 on GPUs 0,2 and VRPTW100 on
GPUs 1,3, independently and concurrently. Each task first trains a random,
task-specific optimized-v2 PPO initialization for 100 epochs. This PPO phase has
no expert, solution-level, reference, priority-sampling or replay signal. Both
parameter-screening arms then strictly load the same validation-best PPO weights:

| Phase | Epochs | PPO passes | SL coefficient | Schedule |
|---|---:|---:|---:|---|
| Control | 40 | 4 | 0.50 | constant LR 5e-5, entropy 0.01 |
| Candidate | 40 | 3 | 0.35 | constant LR 5e-5, entropy 0.01 |
| Selected long run | 1000 | selected | selected | warmup 20, cosine 5e-5 to 1e-5 |

The final epoch-40 validation exports determine selection: feasible coverage
first, then mean distance on the same jointly feasible instance IDs; distance
ties within 1e-4 km favor fewer updates. The 1000-epoch phase starts again from
the common PPO best weights, with a fresh optimizer and schedule. All phases
use the existing 5000 train / 1000 val instances. Frozen test bundles and the
new test Gurobi references are not inputs. This is a single-seed parameter
screen, with both arms using v2; it is not an original-model comparison or an
isolated attribution of PPO versus SL effects. The constant-schedule 40-epoch
probe does not establish long-horizon optimality. Replay keeps its existing
25-epoch warmup and 75-epoch ramp; this short screen cannot evaluate the full
replay steady state.

Each GPU handles 32 instances with 50 trajectories, giving global batch 64 and
3200 trajectories. Four minibatches mean 16 global instances per optimizer
step, with 16 update attempts per control epoch and 12 per candidate/PPO epoch.
The LR peak 5e-5 corresponds to the reduced global batch, versus 7.071e-5 at
128 instances in the earlier CVRP50 experiment. Chunk size 8 and eval batch 32
limit memory. Train and evaluation allow 201 actions: the former 110/120
limits were shorter than some Cus100 expert routes; completed rollouts still
stop early. RDI/AGDA architecture and replay settings are unchanged from v2.

The launcher writes input/config/source hashes and the full protocol to
`manifest.json`. Top-level and task-level `status.json` separate configured
budgets, currently running phases and actually completed epochs. Each phase
records training/evaluation CSVs, validated route exports, per-rank monitors,
rolling checkpoints every five epochs and validation-best weights. Hardware
samples are stored in the top-level `hardware.jsonl`. Transient reads of
actively written training logs must not terminate training. A failed task is
reported as failed while the other independent task may continue. Launch from
a committed detached snapshot, with shared result storage:

```bash
PYTHON_BIN=/absolute/path/to/.venv/bin/python bash scripts/run_cus100_finetune.sh \
  --data-root /absolute/path/to/AAAI_Dataset/dataset \
  --cvrp-gpus 0,2 --vrptw-gpus 1,3
```

After each selected long run finishes, its validation-best checkpoint is frozen
and evaluated once on all 1000 task-specific test instances, using 50 samples,
batch 32 and 201 steps. The separate `test_best/summary.json` and route exports
compare against the matching `results/gurobi/{task}/test/Cus100` incumbents;
TIME_LIMIT references are not claimed to be certified optima. Test metrics never
feed back into the parameter selector or checkpoint selector. Explicit
`--test-root` and `--gurobi-root` paths are supported for detached checkouts.

#### Increasing useful GPU memory without changing the training batch

The Cus100 launcher accepts `--cvrp-chunk-size` and `--vrptw-chunk-size`.
These control the number of rollout time steps held in each backward graph.
The PPO objective averages masked per-step losses and weights each chunk by its
time span; the shared SL term compensates for that same chunk weight. Enlarging
the chunk preserves that mathematical objective, the global instance batch,
optimizer-update count and LR schedule. Floating-point accumulation and the
unweighted averages of per-chunk diagnostic statistics can still differ.
Memory occupancy alone is not a throughput improvement: profile real updates,
including experts and monitoring, and record both allocated/reserved CUDA
memory and observed device memory use. Do not allocate dummy tensors to satisfy
a memory-utilization target.

For resource changes during PPO initialization, stop the source supervisor at a
completed rolling-checkpoint boundary, then launch a fresh experiment with
`--resume-from-experiment /absolute/path/to/stopped/experiment`. The launcher
copies the latest checkpoint into immutable resume inputs, checks complete
sampler/RNG/scaler state for both ranks, and imports both rank CSV histories,
monitor records and validation exports only through the checkpoint epoch.
Original artifacts are preserved. Pending-validation checkpoints, changed
batch/LR/evaluation protocols, and missing epoch history are rejected. Only the
PPO init resumes; subsequent parameter-screening and long-run phases continue
to initialize from the completed shared PPO best. Progress distinguishes
imported epochs from newly completed epochs. A new source snapshot and the
checkpoint/config/history hashes record the resource-change boundary.


## Reward and normalization screen (2026-10-07)

The current hypothesis is that physical-unit advantages and stable critic target
scaling improve short fine-tuning. This is a controlled screen, not an established
accuracy improvement or a replacement for a full multi-task training comparison.
The fixed starting point is update5 epoch300 from the VRPTW100 sweep, shared by
all four arms and both fine-tuning seeds (3009, 3010). Seeds do not represent
independently pretrained models.

The experiment branch distributes the exact source weights as a compact
`assets/reward_norm/vrptw100_update5_epoch0300.pt` initialization. Its manifest
records the source archive hash separately from the compact artifact and tensor
hashes. Optimizer/sampler/replay state is excluded; this screen creates those
states afresh. New servers only need Git plus the local train/val dataset and
reference files, not a copy of the old sweep's results directory.

The distance reward is `-edge_km / d0`. Freeze `d0` to the source checkpoint's
training unit (43.638668060302734 km here); do not refit it separately for each
customer count. Observation distance scaling is now independent via
`env.observation_distance_scale_km`; its default preserves legacy behavior and
this screen explicitly preserves the checkpoint's input unit. Demand/capacity,
time/horizon and energy/battery features remain unchanged. With gamma1 and a
complete feasible rollout, accumulated reward equals negative total route cost
in this fixed unit. Reward-identity checks execute before each new-mode update.

`training.reward_norm_mode=physical_shared_popart` enables a bundle:

- Scalar PopArt keeps value predictions in original reward units at the rollout
  interface. It updates target moments once per complete rollout and compensates
  the final linear head to preserve its raw predictions. Normalized MSE is used
  for critic learning. Adam head moments are scaled by old_std/new_std (and its
  square for second moments); this is a gradient-unit conversion, not a claim of
  optimizer-trajectory invariance.
- One actor EMA RMS normalizes all PPO step advantages. The first training
  rollout calibrates it before optimization; subsequent rollouts use the previous
  historical snapshot, frozen for every minibatch and PPO pass. No customer count,
  per-instance standard deviation, or validation/test statistics enter the RMS.
- On-policy solution advantages use leave-one-out sampled cost differences,
  converted from km to the same fixed reward unit and divided by the same actor
  snapshot. Experts do not participate in the on-policy group moments. Expert
  and replay losses remain separate, dimensionless auxiliary objectives with
  their existing weights and gates.

The four-arm factorial separates gamma .99/1 from this entire bundle; it does
not isolate the bundle's three components. Original length-normalized SL ratios,
step-loss reduction, expert/replay denominators, and RDI/AGDA architectures remain
in this screen. Changing those requires a further ablation. In particular the
existing SL ratio is a geometric-mean surrogate, not a joint trajectory importance
ratio. The scalar critic/complete-episode/single-GPU restrictions are enforced;
new failure penalties, truncation bootstrapping and multi-GPU host integration
are not claimed implemented. Portable normalizer primitives do have CPU Gloo
checks for global moments and empty ranks.

Checkpoints store inference-ready raw-unit critic weights under the existing
model_state_dict keys, plus the normalized head and normalization statistics for
exact epoch-boundary training resume. Thus existing evaluator loading remains
compatible. Weights-only initialization resets optimizer and normalizers; full
resume rejects changed normalization semantics, reward unit or gamma. Mid-PPO-pass
resume is outside the supported checkpoint protocol.

Monitor every rollout's reward identity, raw advantage moments, actor scale used
and next scale, critic mean/std, SL advantage spread, normalization update count,
plus existing raw critic explained variance, PPO KL/clip, entropy, gradient norms,
AMP skips, plugin diagnostics and epoch runtime. Hardware samples continue while
jobs are active. Existing gradient-component probes cover only the first update's
first chunk at the common action-query head, excluding expert/replay gradients;
they must not be described as whole-model gradient conflict measurements.

Resource plan: two homogeneous 4 x 2080 Ti servers run the same factorial with
different fine-tuning seeds. Keep the third 4-card server, the 3-card server and
the 2 x A6000 server available for follow-up once validation identifies a useful
direction. A next three-card experiment can isolate LR 5e-6 / 1e-5 / 2e-5 from a
common selected checkpoint, rather than mixing that sweep into the normalization
comparison. A6000 can later test larger memory footprints; its speed is a separate
hardware result. Existing 500-epoch jobs and their final test retain their GPUs
until completion. Copied stale status files from another machine do not claim
local devices; explicit wait prerequisites still apply.

Validation includes normalizer output preservation, checkpoint and optimizer
resume, CPU global moments, mask/padding handling, legacy behavior, fixed-edge
rewards across Cus15/50/100/1000 synthetic instances, and real VRPTW100 CPU
end-to-end smoke runs for all four arms. Large-N learning quality and an 80-epoch
validation improvement remain experimental outcomes, not consequences of those
invariance checks.


Evaluation cadence update: new reward/norm launches default to validation every
50 epochs, with epoch-zero and final-epoch evaluation retained. The recorded
seed3009 E80/E300 experiment keeps its frozen 20-epoch schedule. Continuations
inherit their source interval and preserve intermediate source-final validation
points. Raw validation curves can be exported with
`scripts/plot_reward_norm_eval.py`; they are validation observations, not test
results or independent pretraining-seed evidence.


## Physical input normalization and representation screen (2026-10-07)

This screen isolates network inputs from reward and advantage normalization.
The design objective is to preserve directed road relationships, node roles,
and physically meaningful resource constraints while making input magnitudes
usable by the network. Depot centering, fixed distance units, and the new
physical context adapter are our adaptations. Their accuracy and speed benefits
are hypotheses; earlier reward/normalization validation results do not establish
an input-representation improvement.

### Literature basis and limits

- [MatNet: Matrix Encoding Networks for Neural Combinatorial Optimization
  (NeurIPS 2021), Sections 3.1-3.3](https://proceedings.neurips.cc/paper/2021/file/29539ed932d32f1c56324cded92c07c2-Paper.pdf)
  encodes relationship matrices using separate row/column representations and
  learns to mix pairwise costs with query-key scores. This supports exposing
  actual directed road costs rather than asking coordinates to recover them.
  Its original one-hot initialization has a size limit and is not adopted here.
  [Official implementation](https://github.com/yd-kwon/MatNet).
- [RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization
  (ICLR 2026), Sections 4.1.1-4.1.2](https://arxiv.org/html/2503.16159v2)
  combines coordinate and sampled-distance node features through contextual
  gates. Its edge bias separately embeds distance, duration, and direction
  before fusion, retaining asymmetric relationships. We borrow the separation
  of physical feature types and complementary node/edge information; we do not
  reproduce its complete encoder or claim that its normalization preserves
  absolute city scale. [Official implementation](https://github.com/ai4co/real-routing-nco).
- [RADAR: Learning to Route with Asymmetry-aware Distance Representations
  (ICLR 2026), Section 4 and Appendix E](https://arxiv.org/html/2603.03388v1)
  initializes nodes using left/right factors of a truncated distance-matrix
  SVD and uses both edge directions in attention. Appendix E compares z-score,
  min-max, and unnormalized inputs on synthetic ATSP; this evidence does not
  establish the best scaling for constrained real-world VRPTW/EVRPTW. Its
  real-world experiment uses the RRNCO min-max setup. Per-instance z-score
  removes absolute scale unless another input preserves it. SVD and Sinkhorn
  are deferred, separately testable architecture changes.
  [Official implementation](https://github.com/yihang0410/RADAR).
- [RouteFinder: Towards Foundation Models for Vehicle Routing Problems,
  Sections 4.2.2 and Appendix B.2](https://arxiv.org/html/2406.15007v3)
  explicitly distinguishes node and global attributes and feeds global
  conditions into the encoder. Appendix A divides demand by capacity and
  describes normalized coordinate inputs; its example time-window generation
  assumes unit speed. We borrow the explicit global context, not that speed
  assumption or its full normalization recipe. Transformer RMSNorm/pre-norm
  concerns hidden activations and is not changed by this input experiment.
  [Official implementation](https://github.com/ai4co/routefinder).
- [Buckingham, On Physically Similar Systems; Illustrations of the Use of
  Dimensional Equations (1914)](https://doi.org/10.1103/PhysRev.4.345)
  motivates consistent dimensionless groups and unit transformations. It does
  not prescribe a neural architecture, guarantee generalization, or imply that
  cities with distinct constraints should share identical embeddings. The
  particular groups below are our application of dimensional consistency.

### Input contract

The experiment exposes two independent factors. Existing behavior remains the
fallback when these options are absent:

```yaml
env:
  observation_coordinate_mode: legacy_minmax  # or depot_fixed
  observation_distance_scale_km: 43.638668060302734
  observation_input_context: false
model:
  use_physical_input_context: false
  physical_input_context_hidden_dim: 32
```

The environment and model context flags must agree. The context interface is a
static node tensor `[B,N,12]` and a static graph tensor `[B,10]`, with an explicit,
versioned feature order and normalization signature. Node context describes
physical incoming/outgoing road relationships, depot relations, and finite-road
reachability (not dynamic vehicle feasibility). Graph context exposes physical resource scales and enabled-task
flags. No city ID, arbitrary customer index, or customer-count multiplier is
used. Changes to feature meanings or ordering require a new schema; matching
tensor dimensions alone is insufficient checkpoint compatibility.

For `depot_fixed`, let `p_i` be coordinates in km, `p_0` the depot coordinate,
and `D0 = 43.638668060302734 km`. Use `(p_i - p_0) / D0` with the same divisor
for both axes. Negative and greater-than-one coordinates are valid; do not clip
or refit a bounding box. Raw latitude/longitude is not a km coordinate and must
be projected or converted by the data layer before this operation. Coordinates
provide geometry; independently supplied road matrices remain authoritative.
The existing directed `D_ij / D0` channel stays available in every arm, together
with the existing RDI relative features and absolute distance bias.

Resource scaling uses consistent groups, not independent arbitrary min-max
transforms. With time unit `T0` and energy unit `E0`, use `T_ij / T0` and
`E_ij / E0`; time windows, service/wait/charge durations and current time must
share the corresponding time origin/unit, and battery capacity and remaining
energy must share the energy unit. For constant physical speed `v`, consumption
rate `c`, and charging power `P`, the consistent dimensionless coefficients
are `v*T0/D0`, `c*D0/E0`, and `P*T0/E0`. Thus `T'=D'/v'`, `E'=c'*D'`, and
`charge_time'=charged_energy'/P'` remain valid under that physical model.
When measured time/energy matrices exist, retain those matrices instead of
reconstructing them from a constant speed or consumption rate.

For this first screen, existing time/horizon, demand/capacity, and
energy/battery observations remain in their current compatible units. The
optional graph context must preserve the physical scale/constraint information
needed to interpret those ratios. Missing/inactive resources need explicit
flags and finite neutral values, not an unmarked zero that could mean a real
zero budget. Denominators must be validated; padding and self-edges must not
silently alter neighborhood statistics. A reachability feature is an input
hint, not a replacement for the environment's authoritative action mask.

Two cities may legitimately share a representation when their full routing
inputs and constraints are equivalent. The requirement is to retain differences
that affect the decision: road detours, directionality, duration, energy,
resource budgets, and customer attributes. A tenfold physical distance change
with unchanged time windows/battery is not such an equivalence. Merely changing
from km to meters, with all units transformed consistently, is an equivalence.

Only static instance quantities belong in the cached context adapter. Current
position/time/load/battery and visited-node-dependent margins remain in the
dynamic observation/AGDA path. The new context adapter is a residual with only
its final projection initialized to zero; zeroing every layer would obstruct
learning. Adding it must preserve outputs at initialization for the same
coordinate mode, while allowing nonzero gradients into the output projection.

### Four-arm protocol and initialization fairness

| Arm | Coordinate mode | Physical node/global context | Isolated comparison |
| --- | --- | --- | --- |
| `legacy` | `legacy_minmax` | Off | Current Norm input control |
| `depot` | `depot_fixed` | Off | Depot centering and fixed coordinate scale |
| `context` | `legacy_minmax` | On | Added physical context at the legacy coordinates |
| `combined` | `depot_fixed` | On | Both factors and their interaction |

From `CaliRoute`, inspect or launch the portable screen with:

```bash
bash scripts/run_input_norm_comparison.sh --help
bash scripts/run_input_norm_comparison.sh --seed 3009 --gpus 0,1,2,3
```

The shell launches a detached supervisor by default. `--prepare-only` freezes
and hashes the configurations/source/inputs without starting GPU workers.
`--data-root /path/to/AAAI_Dataset` overrides the sibling dataset path; one to
four homogeneous GPUs are supported, with remaining arms queued when fewer
than four cards are provided. GPU locks, idle checks, separate preflight runs,
and completion checks are shared with the reward/norm launcher. Preflight
weights are discarded; formal training reloads the frozen shared archive.

All arms share the completed Norm experiment's validation-best epoch-300
weights, identified by source-file and model-tensor hashes. This is weights-only
fine-tuning: optimizer, sampler, replay, actor RMS and PopArt training state are
initialized afresh under the same policy in every arm. Source inference weights
must represent the raw-unit critic, rather than silently importing a normalized
training head without its moments. Newly introduced adapter keys are the only
allowed missing checkpoint parameters; pre-existing model tensors must match
exactly after loading. Record their provenance separately from the newly
initialized parameters. Re-seed training/evaluation streams after construction
so creating adapter layers does not accidentally shift the sampler/rollout RNG.

The shared configuration is VRPTW100, PPO update passes 5, `n_traj=50`,
300 fine-tuning epochs, validation at epochs 0/50/100/150/200/250/300, and the
same train/validation splits, instance order, batch, learning-rate schedule,
SL/expert/replay coefficients and monitoring cadence. Keep
`training.reward_norm_mode=physical_shared_popart`, gamma 0.99, and the raw
reward `-edge_km / 43.638668060302734` in all four arms. This screen does not
introduce GDPO, independent PPO/SL RMS, a new critic loss, SVD, Sinkhorn, hidden
LayerNorm/RMSNorm changes, or a different RDI/AGDA architecture. The added
context adapter is the explicit architecture factor in the table.

A mature legacy-input checkpoint favors the representation on which it was
trained. Changing coordinates immediately changes its inputs, even though the
weight file is the same. Therefore every arm must run a full epoch-zero
validation and retain its own initial objective/feasibility. Only the
`legacy`/`context` and `depot`/`combined` pairs should match initially after
zero-output adapter loading; equality between those pairs is not expected.
Compare absolute same-epoch and same-wall-time validation outcomes, report the
initial shift and recovery cost, and do not rank solely by improvement from an
arm's own worse epoch zero. This design measures usefulness for checkpoint
migration, not an unbiased ranking of representations trained from scratch.
The promising input design should later receive matched training from scratch
or matched task-specific pretraining, additional seeds and cross-size/city
validation. Several fine-tuning seeds sharing one pretrained checkpoint are not
independent pretraining replications.

### Validation gates and priorities

1. **Representation and physical consistency.** Check depot coordinates are
   exactly zero in fixed mode, translation and unit-conversion invariance,
   preservation of aspect ratio, and equivariance to a joint node/edge
   permutation that preserves the depot convention. Construct two geometrically
   similar instances of different physical extent: the absolute branch must
   distinguish them. Construct identical coordinates with different directed
   road matrices: their road features must differ. Adding customers must not
   rescale existing fixed coordinates or existing physical edge entries; full
   network embeddings and neighborhood summaries may change with the graph.
2. **Environment and consumer parity.** Replay identical fixed actions through
   each mode using real CVRP/VRPTW data and synthetic EVRPTW cases. Raw distance,
   arrival/wait/service/charge times, remaining resources, rewards and action
   masks must agree within documented floating-point tolerances. Check regular
   and fast environments, rollout observations, PPO observation slicing,
   RDI/AGDA inputs and evaluation loaders. This establishes transition/input
   consistency, not real EVRPTW solution quality.
3. **Compatibility and state.** Verify the two within-coordinate epoch-zero
   equivalences on logits and critic outputs; verify adapter gradient flow.
   Disabled-context loading must retain legacy outputs. Save/load and resume
   must preserve the input schema, coordinate mode and scales; fail clearly on
   incompatible full-resume requests. Exercise cached static features and
   renewed reset boundaries, not only direct model-forward calls.
4. **Real-data learning smoke before the long run.** Run short VRPTW100 updates
   for all four arms, including PPO, SL, expert and replay paths. Require finite
   inputs/gradients, completed feasible rollouts, correct evaluation counts and
   readable checkpoints. Log coordinate/context ranges and adapter residual
   norms as well as existing KL, clipping, raw critic error, advantage scales,
   feasibility, optimizer steps, memory and runtime. Initial migration losses
   are reportable evidence; an unexpected numerical/feasibility failure blocks
   the long launch until diagnosed.
5. **Controlled quality assessment.** After the smoke passes, run the four arms
   concurrently on matched GPUs with frozen code/configuration and per-arm
   epoch-zero validation. Keep feasibility visible alongside distance, compare
   paired instance IDs at common epochs and equal training time, and account
   for input/adapter computation in timing. Select using validation; frozen
   test evaluation follows selection. Do not infer city or Cus1000
   generalization from a single VRPTW100 fine-tuning run.

The first priority is lossless unit/feature construction and compatibility, then
the coordinate/context factorial. Broader node/edge fusion, structural SVD
features, attention normalization, and more complex dynamic resource features
remain separate follow-ups. No outcome is asserted until its recorded checks
and corresponding experiments have completed.


Implementation checks completed on 2026-10-07 (before the formal GPU run):

- Broad CPU regression: 548 tests passed, excluding the four unrelated Gurobi
  suites; subsequent targeted input-profile checks also passed.
- Data audit: 54 first/middle/last records across all 18 train/val task-size
  cohorts (CVRP/VRPTW/EVRPTW, 15/50/100). This is a sample, not a full-data audit.
  New context remained finite; physical-unit reconstruction, unchanged non-coordinate
  observations and one-step rewards/dynamics, and fast/reference agreement passed.
- Independent saved travel-time matrices were present for the 18 sampled VRPTW
  records. Maximum reconstruction difference was 0.001776 seconds, within float32
  tolerance. The other records do not establish agreement with independent saved T.
- Full-route synthetic tests include depot returns, charging, capacity, and time
  windows. Translation and physical-scale tests verify the intended invariances.
- Audit details and feature quantiles: repository `results/audit/input_normalization_20261007.json`.
- Input profile guarantees cover the new modes with explicit fixed units. Legacy
  configurations with implicit reward-unit fallback retain historical behavior;
  their effective unit can be resolved after checkpoint loading.

These checks establish implementation consistency, not improved policy quality.
The matched four-arm validation screen and repeat-seed confirmation remain pending.

## Physical model-integration screen (2026-10-07)

This second stage holds the first stage's **combined input** fixed by design.
It does not assert that combined is the winning input normalization. The task is
how to expose physically consistent node, edge and vehicle-state information to
embedding, encoder and decoder, without changing objective or learning loss.
Input units, physical transitions and the environment's final feasible-action
mask remain authoritative. Learned latent vectors need not themselves satisfy a
physical equation such as time=distance/speed.

### Literature and adaptation boundaries

- [FiLM, AAAI 2018](https://arxiv.org/abs/1709.07871) conditions neural features
  through feature-wise affine transformations. Here global vehicle/horizon
  context modulates grouped physical node features with a zero-output residual.
  Its visual-reasoning evidence motivates a mechanism, not a routing improvement
  guarantee.
- [Relational Attention, ICLR 2023](https://arxiv.org/abs/2210.05062) explicitly
  represents and updates edge vectors. We adapt the node/edge distinction to
  directed normalized distance, time and energy, and cache small edge latents.
  A low-dimensional relation head augments the existing encoder bias; a shared
  current-node row can inform decoding. Optional edge-value messages and edge
  updates are separate heavier settings, not a reproduction of the full paper.
- [RRNCO](https://arxiv.org/html/2503.16159v2) motivates retaining real road
  matrices rather than treating coordinates as the whole routing relation. Its
  coordinate/road fusion and the new static integration overlap conceptually;
  implementation, task and controlled comparison here differ.
- [Chain-of-Context Learning, ICLR 2026](https://arxiv.org/abs/2603.01667) builds
  changing constraint context and uses trajectory-shared node re-embedding.
  We borrow resource-conditioned reading, not its cross-trajectory recurrent
  re-embedding. Avoiding dependence on other trajectories keeps stored PPO
  observations sufficient for repeatable policy evaluation.
- [CARM, 2026 preprint](https://arxiv.org/abs/2605.10122) analyzes observation
  restrictions in state embedding and proposes constraint-aware residual
  modulation. We retain the existing feasible-action read and add a separate,
  zero-output resource-conditioned observation read. The environment mask still
  controls the final action distribution. This is an adaptation, not an exact
  CARM reproduction or proof of benefit for our time windows and charging rules.

All five references supply architectural motivation. The exact combination,
initialization strategy and task-specific feature definitions are ours and must
be evaluated rather than attributed to the papers as established conclusions.

### Integration contracts

Static typed fusion separates geometry, temporal/service, demand and physical
road/context information before resource-conditioned residual fusion. Existing
embedding parameters remain shared with the baseline. Directed relation encoding
retains distinct i-to-j and j-to-i costs; scalar attention bias alone need not be
the only consumer. Static relations are cached per instance, and decoder reading
uses the current-node row rather than recomputing all pairs per action. Optional
edge-value messages and edge updates can increase cost and are off in the first
screen; measure actual runtime/memory before claiming a speed improvement.

Dynamic candidate features distinguish arriving at a node from leaving it after
service, charging or depot handling. Customer time-window feasibility is based on
start-of-service, matching the environment; completion slack is a separate
quantity. The candidate representation must distinguish customer, depot and
charging-station semantics, including the charging mode. Depot resource resets
and charging-station post-charge state must not be misrepresented as ordinary
customer transitions. Inactive constraints must not introduce dummy capacities
or battery scales as active decision signals.

The decoder has separate observation and action roles. Currently unavailable
customers can still matter for the decision to return to the depot, so the new
read may observe them. However, already served customers, padding, depot and
revisitable stations require explicit semantics; a broad read must not silently
make infeasible actions selectable. The original feasible read remains the
initial behavior because the new readout head is zero initialized.

Output heads of the new residual branches initialize to zero, and constructing
new branches must not perturb initialization of shared parameters. This gives a
meaningful epoch-zero equivalence check for the second-stage four arms. It does
not imply that all internal features are zero or that all module layers receive
nonzero gradient on the very first step. New parameters and input/model profiles
are recorded with checkpoints; weights-only migration allows only named new
modules, while full training resume must retain compatible architecture and input
semantics.

### Frozen four-arm protocol

| Arm | Static integration | Dynamic integration |
|---|---|---|
| `baseline` | No new static modules; combined input remains enabled | Existing decoder |
| `static` | Typed fusion plus directed relation encoder and current-edge reader | No resource decoder |
| `dynamic` | No new static modules | Resource decoder with dual observation |
| `combined` | Static integration | Dynamic integration |

The static current-edge reader is included whenever the relation encoder is
active, including the `static` arm. Thus this arm evaluates useful shared static
relations across encoder and decoder, not only an encoder modification. The
optional `--edge-messages` and `--edge-updates` switches affect the static-enabled
arms only and are recorded separately. To isolate individual static changes,
subsequent fine-grained ablations should split typed fusion from relations.

```bash
bash scripts/run_model_integration_comparison.sh --seed 3010 --gpus 0,1,2,3
```

The portable bundled initialization remains
`assets/input_norm/vrptw100_norm_epoch0300.pt`: the completed prior Norm model's
epoch-300 raw-unit inference weights, **not** a trained combined-input winner.
It removes any dependency on another server's private results. All arms apply
the same coordinate/context migration and start fresh optimizer, replay,
actor-RMS and PopArt state. The new physical-input adapter is enabled in all arms
and learns during this screen. Epoch-zero policy/critic equivalence therefore
compares second-stage structures under the same input shift.

Protocol: VRPTW100, seed 3010 by default, 300 additional epochs, four independent
single-GPU arms, 64 instances x 50 trajectories, update5, four minibatches,
chunk12, LR 1e-5, gamma .99, GAE .95, entropy .002, SL coefficient .35 and expert
weight .6. Replay schedule, `physical_shared_popart` and all reward/SL-PPO
semantics stay unchanged. Full 1,000-instance validation uses best-of-50 at epoch
0, every 50 epochs and final epoch. Each formal run starts after its separate
two-epoch GPU preflight with the same 64-instance x 50-trajectory training
shape, PPO update count, minibatches and chunk size. Preflight validation alone
is reduced to four instances and best-of-four; learned weights are discarded.
This checks actual training memory rather than only a tiny rollout. It cannot
prove every later replay/evaluation peak will fit. If it fails with OOM, rerun
a fresh whole comparison using `--chunk-size 8`, keeping batch and trajectories
unchanged across arms. Tests are
reserved for final selection; no running jobs on the preparation machine need
to be interrupted.

Source, initialization, data references and generated configs are hashed and
frozen by the shared supervisor. Fewer than four matching GPUs queue the arms;
idle checking and cooperative GPU locks avoid taking active cards. Compare
quality at matching epochs and matching elapsed training time, reporting
feasibility and GPU model. The optional heavier arms should be treated as a new
experiment block, not silently added to one arm mid-run.

### Checks before interpreting quality

- Check default-disabled compatibility, shared-parameter/RNG preservation and
  zero-initial policy/critic equivalence across all four arms. At epoch zero the
  measured raw distance, feasibility and coverage must agree; missing or partial
  validation is pending, not a successful equivalence check.
- Verify candidate arrivals, waiting, start-of-service, departure, capacity and
  charging/depot handling against physical environment transitions. Cover
  capacity-only, time-window and battery-active cases and inactive constraints.
- Check directed-edge/node permutation behavior, static-cache versus direct
  execution, PPO replay and expert paths, old-weight migration and full-resume
  signature checks. All-padded or no-newly-observable cases must remain finite.
- Check gradient flow and finite optimizer steps, recording module output,
  gradient and update norms together with KL, clipping, entropy, raw critic
  error, normalization statistics, feasibility, timing and GPU memory. A
  zero-output module that stays disconnected is not a valid ablation.
- Interpret this as a single-scale migration screen. Repeat seeds, matched
  pretraining/from-scratch tests and cross-size/city validation are still needed
  before claiming broad plugin improvements. Preserve reward until this stage
  is assessed to avoid mixing architectural and reward effects.


### Implementation validation (2026-10-07)

The non-Gurobi test suite passed **628 tests**. This covers the new typed/static
and dynamic modules, physical transitions, mixed-precision tensor paths,
permutation/direction behavior, PPO chunk/trajectory equivalence, old-weight
migration, resume contracts and launcher behavior. The four unrelated Gurobi
integration/resume/sharding/launcher test files were excluded from this run.
After the comparison-report metadata adjustment, both affected launcher test
files passed again (54 tests).

A real-data CPU pipeline check used four training and four validation VRPTW100
instances, the actual 256-wide two-layer model and bundled initialization. Each
of the four arms completed two epochs with PPO, expert/replay paths, evaluation
and checkpoint saving. All four produced identical epoch-zero selected routes
and distances; every selected validation route at epochs zero and two passed
independent physical validation. Added module groups had nonzero gradients and
parameter updates, with no nonfinite gradients or skipped optimizer steps.
The optional combined edge-value/edge-update configuration also completed two
CPU epochs. A separate combined run loaded its epoch-two model and optimizer
with no missing/unexpected keys and completed epoch three.

These CPU checks used a reduced batch/trajectory/update count and are pipeline
verification, not quality, speed or CUDA-memory evidence. Their local artifacts
are in `results/optimization/MODEL_INTEGRATION_CPU_SMOKE_20261007T220246Z/`
and are deliberately not committed. A prepare-only four-card check verified
frozen source/config hashes and the full-shape GPU preflight settings without
launching or interrupting GPU work. Actual CUDA allocation and long-run results
remain for the destination server to verify.
