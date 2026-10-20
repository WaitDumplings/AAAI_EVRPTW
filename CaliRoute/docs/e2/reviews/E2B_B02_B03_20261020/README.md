# E2-B B02 AWBC / B03 DAPG — native RRNCO VRPTW100

Launch code: `5905f9b` on `experiment/e2b-rrnco-ppo-slppo-20261018`.
GPU0 runs AWBC and GPU1 runs DAPG. B01/B04 are already test-accepted; their frozen
runs and checkpoints are not changed or used to initialize the new experiments.

From the review checkout's CaliRoute directory:

```bash
bash scripts/run_e2b_rrnco_second_batch.sh --execute
```

This uses nohup/setsid and records its PID in the existing campaign scheduler.
The launcher refuses occupied GPUs and existing B02/B03 paths. It requires both
full-protocol GPU preflights to pass before starting either formal run. Formal
runs each select their best validation checkpoint, test 1,000 independent
instances with K=50, check physical costs and inventory, and update E2B_results.csv.
Failures remain explicit; no automatic resume or overwrite of previous runs.

## Common online protocol

The protocol hash is unchanged from B01/B04: seed 3009, scratch initialization,
1,500 online rollout iterations, batch 64, 50 trajectories, four PPO passes,
four minibatches, FP32, actor LR=1e-6, critic LR=1e-4, gamma=1, GAE lambda=.95,
clip=.2, value coefficient=.5, entropy coefficient=.01, gradient norm limit=1,
KL target=.02 with the same pre-minibatch stopping rule. Each method has 96,000
online instance exposures and 4.8 million complete online trajectories.
Native road inputs, actor, critic, physical environment and initialization hashes
remain matched. No additional RDI or AGDA. Validation every 50 iterations uses
1,000 instances and batch 8; final best-test uses batch 1 and inference seed 17003009.

## E1 methods retained

`compute_awbc_loss` and `compute_bc_loss` are compiled unchanged from frozen
`offline2online/offline_data.py`. Only their host policy-statistics callback is
replaced with native RRNCO forward passes. E1 `ablation/dapg.py` delegates its
loss directly to `compute_bc_loss`; the E1 trainer defines the coefficient and
warmup used below.

| Method | Expert loss and schedule |
|---|---|
| AWBC | coefficient .10; positive improvement over current rollout's mean successful cost; weights clipped to [0,1] after division by `.10 * abs(expert_cost)`; global weighted sampled-step mean |
| DAPG | three BC warmup epochs × 16 updates, coefficient 1; then PPO plus BC with coefficient `max(.10 * .995**online_iteration_zero_based, 0) * max(normalized_valid_advantage)`; demo gate 1 |

Each expert update samples 512 decisions uniformly with replacement from the same
5,000 verified train experts (sampler seed 20009). Teacher forcing excludes the
external first POMO customer and post-completion padding, following the common
native-host action contract; the explicit final depot is retained when needed
for native completion. There is no model-generated route replay or incumbent
memory in AWBC/DAPG. The 48 DAPG warmup updates expose 24,576 expert decisions and
are additional to all 1,500 online iterations, never deducted from that budget.
Actual online optimizer updates, total updates including warmup, expert decisions,
warmup time and active wall-clock including validation are recorded separately.
The formal training time includes warmup; preflight work and failed preflights
are excluded and retained separately.

Native prefix states are parameter-independent and cached on CPU; native encoder
embeddings are recomputed with current parameters on every update. Activation
checkpointing with chunk 32 bounds expert memory. Full-batch loss normalization
is preserved; there is no per-chunk weight renormalization. Neighborhood sampling
uses the native distribution and checkpoint recomputation preserves its RNG.
Online rollout/replay retains the original fixed discrete neighborhood samples.

## Evidence and acceptance

The new adapter verifies all 5,000 experts through native masks/completion and the
independent physical cost validator. A native full-route replay is compared with
cached prefix states using identical neighbors and encoder batch shape. The
first diagnostic compared batch 1 against batch 32 and measured a FP32 logprob
maximum difference of 0.000154495; that preflight was rejected before training.
The strict comparison now uses matched encoder batch shapes, with a 2e-5 maximum
logprob threshold. The cross-batch numerical difference remains reported.
A separate full/chunked gradient comparison checks memory chunking.

The complete preflight also checks old/replay ratios, nonzero finite expert
gradients, policy probability change, all 3,200 sampled physical rollout routes,
online and expert sampler recovery, parameter recovery, validation and best
checkpoint reload. DAPG additionally saves/restores midway through BC warmup.
Formal training starts from a fresh initialization, not preflight weights.

The E1 reference files match B04's frozen harness byte-for-byte:

| File | SHA256 |
|---|---|
| offline2online/offline_data.py | 3881f22f7905b98f4ccc67c0db02be2f63b4b453afea87ba2efb46dc62a11cac |
| offline2online/trainer.py | 1d1f83685ccc37ca2ea84f29c1dcd25be45c124abe0dfb79a1313aec2856d263 |
| caliroute/methods.py | 2db1a87643a2284b4b50e9af8b92405c4b7ab6a693e00671103f3157b18932f3 |

Progress is in `results/e2/E2B_RRNCO_VRPTW100_S3009/scheduler_a6000/status.json`.
Method logs are `runs/awbc/console.log` and `runs/dapg/console.log`; DAPG also has
`warmup.jsonl`. Configs, native source, harness and manifests are frozen per run.
The authoritative measured preflights live in `preflight/{awbc,dapg}/preflight.json`.

## Expert memory and throughput check

A separate GPU0 diagnostic used the same 512 sampled native decisions and fixed
neighbor samples to compare checkpointed chunk 32/128/512. It did not load or
modify any formal checkpoint. Chunk 32 took 10.256 s (4.25 GiB allocated peak);
chunk 128 took 9.858 s (16.75 GiB allocated peak), only about 4% faster. The maximum
compared decoder gradient difference was 2.97e-5, consistent with the additional
batch-shape numerical differences. Chunk 512 exceeded this card's memory in the
diagnostic. The actual experiment retains the validated chunk 32, with no change
to effective expert batch 512 or common online batch 64. Exact measurements and
the diagnostic source are saved alongside this report.

## Measured acceptance and launch

Both full-protocol preflights passed and formal jobs started on 2026-10-20 UTC.
Five unit tests passed. Matched-shape cached/full-route logprob error was zero
for both methods; the shared online old/replay check also passed.

| Method | Expert gradient L2 | Expert probability change | Resume parameter max error | Warmup updates | Peak reserved GiB | Preflight seconds |
|---|---:|---:|---:|---:|---:|---:|
| awbc | 30.539171 | 0.215325 | 3.8801809e-06 | 0 | 7.336 | 240.910 |
| dapg | 18.712879 | 0.249507 | 3.9935112e-06 | 48 | 8.299 | 758.706 |

DAPG warmup recovery passed as well. Native source and all three data manifests
were compared against B01 and matched exactly. The first rejected diagnostic
remains under `failed_preflights/second_batch_20261020T033035Z/`.
Formal runs are independent scratch runs with automatic best-checkpoint test.
