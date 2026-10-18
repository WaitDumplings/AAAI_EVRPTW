# RADAR CVRP100: additive RDI and AGDA experiments

User-authorized allocation: GPU0 RADAR+RDI; GPU1 RADAR+AGDA. RDI retains
native SVD and adds a learned directed-road bias before the existing Sinkhorn
operation in each encoder layer. AGDA retains native SVD, encoder, Sinkhorn
and glimpse, and adds candidate-conditioned action-key and post-clip bias.
These are separate arms; AGDA does not also enable RDI.

Both arms start from scratch with seed3009, native REINFORCE/POMO, batch64,
50 trajectories, LR0.0004 and 96,000 instance exposures (1,517 updates,
including data/evaluation boundary batches). This matches the audited E1
RADAR baseline; it is not 1,500 complete data passes or PPO training.
Native sampling, optimizer, scheduler and E1 training pool are retained.
Validation uses 1,000 instances every3,200 exposures, K50 and batch8;
best selection maximizes feasibility then minimizes mean feasible km.
Best checkpoint test uses the independent 1,000-instance split, K50,
batch1, native greedy starts, inference seed17003009 and physical validation.

Hardware queried on this server: two RTX 6000 Ada Generation 48GB GPUs;
`a6000` remains the server alias, not the measured GPU model.
AGDA uses non-reentrant activation checkpointing and trajectory chunk50.
Native environment load/mask inputs are cloned before checkpoint replay to
avoid in-place environment mutations corrupting backward computation.
Batch size is not raised to fill memory, preserving the matched protocol.

## Acceptance evidence

- Four adapter tests pass: disabled/zero equivalence, gradients, masks,
  RDI learning, chunk/checkpoint probability and gradient equivalence,
  including native-style in-place load/mask mutation after forward.
- Both actual batch64×50 preflights complete two optimizer updates and
  physical validation, then restore full checkpoint state and reproduce
  validation cost exactly. See the two adjacent preflight JSON files.
- Shared native initial parameter SHA256 in both arms:
  `237f292a787d19593e2a34a0bea0dbc03b512762630c042b8bfdd056cc77bc8f`.
- Both first batches exactly reproduce the audited E1 loss
  `-331.2273864746094`, cost `1570.586669921875`, and354,119 decisions.
- Peak reserved memory: RDI12,492,734,464 bytes; AGDA13,048,479,744 bytes.
  These are short-preflight peaks, not guaranteed full-run peaks.
- No formal experimental results are claimed by these short checks.

## Launch and monitoring

From the experiment checkout's `CaliRoute` directory:

```bash
bash scripts/run_e2_radar_variants_a6000.sh --execute
```

The shell detaches with nohup/setsid. The supervisor refuses occupied GPUs,
locks the campaign, verifies source/data/configuration hashes and requires
a matching full-batch preflight. Each formal run starts fresh; it never
continues the smoke checkpoint. Every worker runs its frozen harness/source.
Formal test output is checked for 1,000 distinct IDs, K50 and cost mismatches.

Campaign (in the original checkout, visible in the user's IDE):
`results/e2/E2_RADAR_CVRP100_S3009_A6000_V1/`.

- `scheduler_a6000/status.json`: supervisor, worker PIDs and completion.
- `runs/{rdi,agda}/console.log`: live logs.
- `runs/{rdi,agda}/native_train.jsonl`: updates, exposure budget and timing.
- `runs/{rdi,agda}/native_validation.jsonl`: validation and best selection.
- `runs/{rdi,agda}/native_manifest.json`: actual frozen hashes.
- `runs/{rdi,agda}/test/`: accepted best-checkpoint test manifest and instances.

Original baseline: `results/e1_cvrp_cus100/E1_CVRP100_S3009_V1_a6000_cuda_fix1/runs/radar`.
The shell supports `E1_RADAR_PYTHON`, `E2_RADAR_BASELINE`,
`E2_RADAR_CAMPAIGN`, and `--gpus 0 1` overrides for another machine.
