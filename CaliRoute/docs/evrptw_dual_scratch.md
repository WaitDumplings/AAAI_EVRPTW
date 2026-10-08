# EVRPTW100 original versus optimized training on two GPUs

These launchers train EVRPTW100 from random initialization with two GPUs per
model. They compare the original model and SL-PPO losses at commit
`f388343dbb1d54bbd3f76dd29ca95208070d31a8` against the current optimized `explore`
bundle. No PPO-init, previous best, initialization asset or resume checkpoint is
used. Training expert routes remain part of SL-PPO.

The original model/loss source is archived separately. An external adapter
provides gradient synchronization across two ranks and the shared evaluation
measurements; it does not substitute the modern model or environment for the
original ones. Distributed execution and the declared budget overrides are part
of this experiment. This is not a bitwise reproduction of the historical
single-GPU run.

The original EVRPTW expert builder copies static edge matrices at every expert
step. With this release's 494,521 expert steps and 121 nodes, just the three
float32 edge matrices would occupy about 81 GiB per rank before other arrays.
The dual-GPU original launcher therefore enables
`offline.original_share_static_expert_observations`: identical static arrays
within an instance are stored once, read-only, while constructing the expert
buffer. Dynamic observations and all expert steps are retained. This external
storage adaptation does not introduce a modern encoder-forward cache or alter
the original loss; its storage counters are recorded by the adapter. Original
single-card launchers retain their previous storage behavior by default. A CPU
two-rank EVRPTW SL-PPO comparison with five passes and four minibatches confirmed
that enabling this storage adaptation preserved every final model tensor and
all logged losses, entropy, KL, reward and evaluation values exactly in that
fixture. It does not claim that the whole new distributed run matches the
historical single-GPU run bit for bit.

## Launch on the target server

From `CaliRoute/`, with the dataset installed and two GPUs available per run:

```bash
# Original model, using physical GPUs 0 and 1.
bash scripts/run_evrptw100_original_dual.sh --gpus 0,1

# Optimized explore bundle, using physical GPUs 2 and 3.
bash scripts/run_evrptw100_optimized_dual.sh --gpus 2,3
```

The shells start background supervisors. Each run reserves its GPU pair
atomically and uses both cards for one distributed model. These are two
independent model runs, not four single-GPU arms. A supervisor waits for its
pair; it does not interrupt existing training. Keep the currently running
four-card experiment separate from these new launches.

To prepare and inspect configurations without starting workers:

```bash
bash scripts/run_evrptw100_original_dual.sh --prepare-only --gpus 0,1
bash scripts/run_evrptw100_optimized_dual.sh --prepare-only --gpus 2,3
```

The default seed is `3010`; use `--seed` to replicate. The original wrapper
defaults to GPUs `0,1` and the optimized wrapper to `2,3`. The main controls are:

```bash
bash scripts/run_evrptw100_optimized_dual.sh --gpus 0,1 \
  --epochs 1500 --batch-per-gpu 32 --chunk-size 8 --expert-chunk-size 64 \
  --data-root /path/to/AAAI_Dataset
```

`--data-root` points to `AAAI_Dataset`, not to its `dataset` child. The original
commit must be available in local Git history for source archiving; fetch full
history if a shallow clone omits it. There are no initialization or resume
arguments in this comparison. Use the same seed and dataset release when
comparing the two launchers; repeat with additional seeds before selecting a
configuration.

Each run first performs a separate two-epoch distributed preflight with its
formal training batch and reduced validation. Only a passing preflight starts
the formal scratch run. All preflight weights, optimizer state, replay and
normalization statistics are discarded. Local CPU/Gloo and toy-instance checks
are the verification scope before this remote GPU preflight; they do not prove
that a full EVRPTW100 batch fits or trains successfully on the target GPUs.
The existing local four-card jobs are not used or stopped for this check.

`tests/test_evrptw_dual_training.py` also runs the actual optimized trainer with
two CPU/Gloo ranks, a small synthetic dataset and a 16-dimensional network.
One rank receives only failed episodes and the other only successful episodes;
the test checks synchronized parameters and reward statistics after four
optimizer steps, physical reward identity, and shared independent evaluation
at epochs 0 and 1, including an instance without a reference solution. This
covers distributed control flow and failure handling, not target-GPU memory
capacity or EVRPTW100 convergence.

## Common training and data protocol

| Setting | Original and optimized runs |
| --- | --- |
| Task | EVRPTW100, 20 physical charging stations |
| Initialization | Random weights and fresh training state |
| Duration | 1,500 epochs |
| Distributed batch | 2 ranks × 32 instances = 64 instances per rollout |
| Trajectories | 50 per instance; 3,200 per global rollout |
| PPO | 5 passes, 4 minibatches per pass |
| Learning rate | Constant `1e-4` |
| Entropy / SL coefficient | `0.01` / `0.5` |
| Training sampler | Uniform `shuffle_cycle` |
| PPO / expert chunk | 8 / 64 by default |
| Charging | `fixed_full`: every station visit takes the fixed full-charge time and restores a full battery |
| Collector horizon | 512 actions |
| Validation | All 1,000 instances, sample best-of-50, batch size 16, fixed isolated evaluation seed |
| Validation schedule | Epoch 0, every 50 epochs, and the final epoch |

The original public SL-PPO preset uses four PPO passes and weighted-priority
sampling. This protocol explicitly overrides them to five passes and uniform
sampling, along with its batch, horizon and validation settings. Original
architecture, reward/input normalization and loss semantics remain native to
the archived source. The current model uses the complete physical-input,
model-integration, reward/PPO, structural-archive and independent-search bundle.
It also fixes charging-return action masks: a final customer may require a CS
before returning, and a return path cannot reuse a CS already visited in the
current route. The archived original environment remains unchanged; this is
an additional reason to interpret the result as a complete-system comparison,
not a pure model-architecture ablation.

The implementation averages rank-local masked objectives at synchronized
optimizer boundaries; it does not claim an action-count-weighted loss over
one concatenated global rollout. Samplers and policy memories are rank-local.

Four minibatches divide the distributed rollout; they are not four additional
independent training processes. Chunk sizes bound time-step/expert evaluation
memory and do not increase the intended global batch or PPO pass count. Twenty
synchronized optimizer-step attempts are made per ordinary training epoch;
record actual execution and any skipped AMP steps when comparing compute.
Changing batch size or rank count is a protocol change, not just a memory knob.

There are 121 terminals: depot + 100 customers + 20 physical CS. The environment
has a `4 × 121 = 484` step cap; the 512-action collector budget lets an episode
reach its own terminal success or failure before collection ends. It does not
extend the environment cap or make a failed episode feasible. Early failures
remain failures. Gurobi's virtual CS copies are a solver modeling choice and do
not change the physical CS count fed to the network.

Required split files are:

```text
AAAI_Dataset/dataset/evrptw/train/Cus100/instances.pkl
AAAI_Dataset/dataset/evrptw/train/Cus100/metadata.json
AAAI_Dataset/dataset/evrptw/train/Cus100/expert_solutions.csv
AAAI_Dataset/dataset/evrptw/val/Cus100/instances.pkl
AAAI_Dataset/dataset/evrptw/val/Cus100/metadata.json
AAAI_Dataset/dataset/evrptw/val/Cus100/gurobi_summary.csv
```

Preparation checks the metadata for the required instance counts and Cus100/CS20
shape, and hashes the inputs before freezing the run.

The audited release has **5,000 training instances but 4,832 verified expert
solutions**, and **1,000 validation instances with 968 verified references**.
Missing expert/reference solutions do not remove instances from the task.
Train on the intended training pool; use an expert term only where the required
expert route exists. Evaluate all 1,000 validation instances and retain their
feasibility and raw-distance measurements even when no usable Gurobi reference
exists. Reference-gap metrics must disclose their smaller matched subset.

The CPU audit of all expert CSVs found:

| Expert-solution statistic | Train | Validation |
| --- | ---: | ---: |
| Recorded expert solutions | 4,832 | 968 |
| Total solution actions, median / 95th percentile / maximum | 102 / 104 / 137 | 102 / 104 / 111 |
| Solutions visiting a CS | 234 | 46 |
| Vehicle routes repeating the same physical CS | 0 | 0 |
| Selected expert routes passing independent physical validation | 24 / 24 | 22 / 22 |

Actions are counted as `sum(len(route) - 1)` over vehicle routes. The selected
validation samples include the longest solutions, the most CS visits and
regularly spaced rows; this is not a full physical replay of every reference.
Two solutions in each split reuse a CS across different vehicles, which is
allowed. Eleven sequence fields omit their initial depot; their `routes_json`
representations are complete, and the expert loader prefers that field. The
audit is recorded locally in
`results/audit/evrptw100_original_eval_audit_20261008.json` when available; generated
results are excluded from Git. These expert-length statistics do not predict
random-policy success or prove that a shorter training horizon would be safe.

## Failed episodes and reward consistency

A random EVRPTW policy can fail because of energy, time windows or the step cap.
The optimized strict-distance protocol permits completed failed episodes as PPO
samples and applies the explicit 1,000 km failure guard in the same fixed reward
unit as route distance. That guard is an experimental penalty, **not a proven
upper bound ensuring every feasible solution beats every failed partial
solution**. It does not convert a failed route into a valid imitation target or
a feasible validation result.

A sampling-budget cutoff is distinct from a true environment failure and must
not be silently relabeled as one. Track feasible rate, termination causes,
served-customer count and reward/physical-cost consistency alongside distance.
The original arm retains its original reward design; the strict penalty is a
change in the optimized bundle, not an unrecorded change to the historical model.

## Shared independent physical evaluation

Both runs use the same independent EVRPTW route validator on the selected
solution. Final feasibility requires both environment success and independent
validity. Evaluation RNG is fixed and isolated from training RNG. Epoch-zero
results measure random initialization and are excluded from native
best-checkpoint selection.

The verifier operates directly in kilometres, seconds and kWh. It checks
customer coverage, directed route distance, cargo load, battery use on every
edge, arrival at a CS before recharge, fixed charging duration, service-start
time windows, waiting, service/charge completion and return to the depot.
Each vehicle starts with a fresh clock, load and battery. A complete route's
actual suffix provides its return-to-depot witness; the independent check does
not copy the environment's conservative lookahead mask.

Explicit `travel_time_matrix_s` and `energy_matrix_kwh` are preserved for
validation when supplied, including through the original model's separate raw
payload sidecar. If absent, travel time uses D/v and energy uses D×consumption.
The audited EVRPTW100 release has no populated explicit T/E matrices or other
stored time matrices, so these fallback formulas describe its current physics.
The validation sidecar does not change the original network inputs, decoding
mask or training transitions. The supported charging mode is `fixed_full`;
other modes require their own verified implementation.

Physical station IDs are not virtual Gurobi copies. The validator charges on
every physical visit without imposing an artificial copy budget. The original
and current environments' restrictions, including visiting a given CS at most
once per vehicle route, remain covered separately by environment success.
Independent validation covers the selected minimum solution; legacy
trajectory-feasibility/median statistics still have their native environment
interpretation rather than an independent replay of all 50 trajectories.

Native checkpoint-best selection differs: the original trainer minimizes
feasible-subset distance whenever its feasible rate is positive, while the
current trainer prioritizes feasible rate before distance. Prefer comparisons
at the same completed validation epoch. If any compared checkpoints are not
fully feasible, reselect from the saved periodic checkpoints using one common
feasibility-first rule before claiming a best-versus-best result. Always show
coverage beside average feasible distance; a smaller feasible subset can make
its average look deceptively good.

## Results and interpretation

Use the experiment directory printed by each launcher. From `CaliRoute/`:

```bash
python scripts/watch_comparison.py results/optimization/<run-directory> --watch 30
```

Inspect the frozen manifest/configs, source hashes, per-rank training records,
preflight results, hardware samples and independently checked route exports.
The original adapter records its distributed/evaluation scope separately from
the archived model source. Missing original-only monitoring fields are missing
measurements, not zeros. In particular, do not invent fresh post-update KL or
new plugin diagnostics for the old trainer. Compare speed using common stage
wall times and hardware; a missing legacy per-epoch timer is not zero runtime.

Rank-one native log directories have different layouts: original uses
`<run_name>/rank_1/seed_3010`, while current training uses
`<run_name>/seed_3010/rank_1` for the default seed. Prefer the common supervisor
status and rank-monitor records instead of assuming identical native paths.

The optimized bundle includes independent branch search and extra replay work,
with at most four instances × eight branch trajectories per rank every five
epochs (64 trajectories across both ranks). Equal global PPO batches and pass
counts do not imply equal compute. Report
search trajectories, time and discoveries with validation results. The two
models have different architectures and may have different initial outputs
under the same seed. This experiment tests complete systems from scratch; it
does not attribute any gain to a single component or establish generalization
from one scale or one seed.
