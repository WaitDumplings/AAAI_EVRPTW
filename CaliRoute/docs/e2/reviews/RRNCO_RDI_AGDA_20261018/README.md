# RRNCO VRPTW100: additive RDI and AGDA

Authorized allocation: GPU0 RRNCO+RDI; GPU1 RRNCO+AGDA. These are separate
E2-A module experiments, not the E2-B PPO/AWBC/DAPG/SL-PPO quartet.

Both start from scratch with seed3009, native REINFORCE/POMO, batch64,
50 trajectories and 96,000 instance exposures: 1,517 updates including the
original data/evaluation boundary batches. Adam LR0.0004, weight decay1e-6,
native normalization, sampling with replacement, scheduler and encoder
checkpointing match the audited E1 RRNCO baseline. Validation is 1,000
instances every3,200 exposures, K50 and batch8. Best selection maximizes
feasibility then minimizes physically validated mean km. Best-checkpoint
test is 1,000 independent instances, K50, batch1, native multistart greedy,
inference seed17003009 and the original decoding cap. Test never selects
the checkpoint. The baseline test mean is239.38207336870022 km.

## Integration

RDI retains native AAFM/NAB, distance-angle-time fusion and decoder alpha/beta.
One shared zero-initialized RoadDistanceInjection(heads=1) adds a bounded
directed residual to the native AFT bias **after** the original NAB alpha
weighting, before the original AFT softmax. Column blocks receive its
transpose. The core receives the native normalized directed distance/time,
time windows and service times; energy is absent. It runs within the existing
encoder checkpointed layers and retains no mutable rollout state.

AGDA retains the encoder and native glimpse. The candidate-conditioned
action-key delta augments the native key score before sqrt(D) scaling.
Native decoder alpha*D+beta*T and log(exp(score-bias)+1e-6) remain intact.
The action bias is carried on the decision's logits tensor and applied after
native tanh clipping, before masking and temperature. Untagged native logits
follow the unchanged native processing path. There is no global dynamic
bias cache. Candidate and trajectory axes retain native start-major order.

VRPTW-specific AGDA inputs are19 candidate features: directed current/reverse/
depot distance and time (8), demand/capacity, remaining capacity, visited,
feasible and depot flags (5), waiting time, TW slack, return-to-depot slack,
service time, window width and arrival time (6). System inputs are remaining
capacity, remaining demand, unserved fraction, current clock and return time.
Four tokens are current/depot/graph-mean/unvisited-mean embeddings. Authoritative
time fields use the same horizon normalization as the native environment;
there are no synthetic battery or time signals. Non-reentrant decoder
activation checkpointing uses trajectory chunk50 and explicit decision inputs.

## Acceptance evidence

All evidence is adjacent JSON, from the existing e1-rrnco environment:

- Two focused tensor tests pass (executed directly because the native venv
  has no pytest): post-clip bias/mask/temperature; nonzero AGDA chunk and
  checkpoint probability/gradient equivalence with environment-state mutation.
- For each arm, real VRPTW100 batch2/start3: three native updates, disabled
  and zero-init first-batch equality, positive plugin gradients, physical
  validation and saved-model recovery equality pass.
- Host gradient relative L2 error at zero is2.61e-6 (RDI) and2.59e-6 (AGDA).
  Disabled-native repeat errors are2.61e-6 and2.75e-6 respectively. CUDA
  backward is not claimed bitwise identical; absolute maxima are0.005859375
  on these large native gradients. Relative acceptance tolerance is1e-5.
- Both full batch64/start50 preflights pass two native updates, physical
  validation and full optimizer/scheduler/RNG/sampler checkpoint recovery.
- Both shared host initial parameter hashes equal the audited reconstruction:
  `5da833c4b2d595a918166f79937d9798bb947c56130988e0aa0c47f5dac43acf`.
- Both full first batches exactly match E1: loss -1.7209655046463013,
  native cost2445.19921875,488,882 decisions,3,200 completed rollouts.
- Peak reserved GPU memory: RDI13.074 GiB; AGDA17.646 GiB, including reload.
  The hardware actually reports RTX 6000 Ada Generation48GB; a6000 is the
  server alias. Short-preflight peaks are not full-run peak guarantees.

## Execution and provenance

```bash
bash scripts/run_e2_rrnco_variants_a6000.sh --execute
```

The nohup/setsid supervisor launches independent workers and automatically
runs each best checkpoint on test. Campaign lock, per-run lock, occupied-GPU
check and implicit-restart rejection prevent duplicate training. Source,
configuration and dataset hashes are checked; formal training requires the
same frozen harness and chunk settings as its passed preflight. Training uses
new models, never the preflight checkpoint. Full source/harness snapshots,
configuration, logs and checkpoint backups remain under the campaign.

Campaign in the original checkout:
`results/e2/E2_RRNCO_VRPTW100_S3009_A6000_V1/`.

- `scheduler_a6000/status.json`: worker PIDs and overall status.
- `runs/{rdi,agda}/console.log`, `native_train.jsonl`, `native_validation.jsonl`.
- `runs/{rdi,agda}/native_manifest.json` and `native_resolved.json`.
- `runs/{rdi,agda}/native_checkpoints/best.pt`.
- `runs/{rdi,agda}/test/{manifest.json,instances.summary.json,instances.jsonl}`.

The E1 baseline is `results/e1/E1_VRPTW100_NATIVE_S3009_A6000_V2/runs/rrnco`.
Its124 native source and7 frozen harness files verify exactly; train5000,
val1000 and test1000 NPZ hashes verify. The historical train manifest has no
test entry; the independent test sidecar was verified against its accepted
test manifest and is included explicitly in the new run manifest.

Use E1_RRNCO_PYTHON, E2_RRNCO_BASELINE, E2_RRNCO_CAMPAIGN and --gpus overrides
on other machines. Short-check metrics above are engineering evidence, not
formal performance claims; formal results are recorded only after completion.
