# E2-B first queue items: B01 PPO and B04 SL-PPO

User clarified GPU0=B01 and GPU1=B04. B02 AWBC and B03 DAPG are neither
started nor queued by this launcher. Their E1 method definitions are retained
in the shared protocol for the subsequent implementation/acceptance work.

## Readiness and frozen common protocol

B01 and B04 passed tensor checks, real-data small training and full
batch64×50 preflights, including old/replay equality, physical validation,
optimizer/RNG/sampler/method-state recovery and best-checkpoint reload.
The native E1 RRNCO result is not used as the B01 PPO baseline.

Both use the native frozen RRNCO VRPTW100 actor and its AAFM/NAB and
distance/time bias. No RDI or AGDA is installed. A shared StepCritic consumes
current128 + graph-mean128 + current time + remaining capacity; its value
gradient reaches the actor encoder. Native encoder checkpointing remains.

| Field | Frozen value |
|---|---|
| Initialization / seed | scratch /3009 |
| Online budget |1500 rollout iterations,96000 instance exposures,4.8M trajectories |
| Global batch / starts |64 /50, single GPU per arm |
| PPO |4 passes,4 instance minibatches of16; one optimizer step per minibatch |
| Optimizer steps |maximum24000; actual count reduced by the common KL guard |
| Adam actor / critic LR |1e-6 /1e-4, constant; weight decay0 |
| GAE |gamma1, lambda0.95; fixed100km reward scale |
| PPO/value/entropy |clip0.2, value coefficient0.5, entropy coefficient0.01 |
| Gradient clipping / precision |combined norm1.0 /FP32; no AMP or accumulation |
| KL guard |before each update; skip that minibatch and remaining passes if mean exp(logratio)-1-logratio >0.02 |
| Reduction |mean valid policy decisions; mean alive value decisions |
| Advantage normalization |all valid policy decisions of the current global rollout |
| Data sampling |independent NumPy default_rng(3009), replacement,64 instances every iteration |
| Validation |1000 instances every50 iterations, K50, batch8 |
| Best |maximum feasible count, then minimum independent physical mean km |
| Test |independent1000, K50, batch1, native multistart greedy, seed17003009 |

The LR/gamma/KL settings follow the audited RRNCO short-PPO proposal and
the current full-batch acceptance, not the old native REINFORCE optimizer.
Legacy `native_*` configuration fields remain only for construction of the
audited host; its optimizer is replaced before the first PPO rollout. The
`ppo_protocol` and `protocol.json` specify the actual E2-B optimizer/budget.

POMO forced first customers have zero learnable log-prob and no actor/entropy
loss, but their physical edge reward/value enter GAE. Done padding has no
loss; shortened minibatches accept only terminal zero-depot padding. Illegal
actions or cap201 truncation fail. Rewards include the implicit final return.
Every replay uses the rollout's discrete sampled neighborhood indices and
recomputes current encoder/cache, preserving native start-major order.

## Complete E1 SL-PPO migration

`e2/e1_solution_bridge.py` compiles the unchanged selected function bodies
and their pure helpers from the frozen `offline2online/trainer.py`. It avoids
importing the CaliRoute actor/environment dependencies into the native RRNCO
venv. The only host replacement is `_expert_route_mean_logprobs`, which uses
RRNCO teacher forcing with fixed discrete neighborhoods and native masks.
The compiled function list and trainer SHA are in `slppo_preflight.json`.

Retained E1 pieces include group/reference/candidate advantages, absolute km
scale and5km std floors, the minimum of external expert cost and OLD scalar
policy incumbent, expert quality/current/memory/support gates, route-mean
clipped surrogate, external expert route branch and the original coefficients.
The defaults are group1.0, reference0.5, expert candidate0.6, total solution
loss0.5 and route clip0.2; full effective dictionaries are in each run config.
Aliases are applied by the unchanged E1 alias function. B04 has no BC warmup.

All reference calculations use one old-incumbent snapshot, including duplicate
draws. Only after preparing both on-policy and expert advantages are current
complete route costs admitted to memory. That memory contains scalars only;
no policy history routes or self-replay. The full method state, sampled-data
RNG, shuffle RNG, optimizer, critic and Torch/NumPy/Python RNG are checkpointed.

The expert CSV is the VRPTW100 train pool with SHA256
`c87e32e45dbe1639ffc75cde39b01df4879ce6822861698face19d006171e583`.
All5000 identities, supplied routes and actual teacher-forcing representations
are independently physically validated. Expert backward steps/routes and
candidate diagnostics are logged. B01's training worker does not open this
pool. The supervisor hashes it only to record shared experiment provenance.

## Acceptance evidence

- Three targeted tests pass: GAE/forced/padding behavior; cost-only immutable
  reference snapshots; E1 group moments and expert gates.
- Both full preflights have old/replay log-prob min/median/p95/max error0,
  ratio max error0; all3200 sampled routes pass the independent validator.
- B01 first iteration completes16 updates, max pre-update KL0.004162.
- B04 first iteration completes6 updates, then KL0.024614 triggers the
  declared guard. Expert gradient norm20.1129 is finite/nonzero; expert route
  mean-log-prob changes by0.107075 after the update.
- Save/restore repeats the same sampled identities and scalar incumbent state.
  Native CUDA backward reductions are not bitwise deterministic: resumed
  parameter max difference is7.96e-6 (PPO),4.65e-6 (SL-PPO), within the2e-5
  short-check tolerance. Reloaded fixed-seed physical validation means match
  exactly. This is not a claim of bitwise-identical long resumed training.
- Peak reserved memory about5.01GiB /5.00GiB. Preflight batches are the full
  effective64×50; low memory results from the16-instance PPO minibatches.
- First two iteration times: PPO17.4/15.2s; SL-PPO22.1/21.1s. Longer-run KL
  stops and expert gates can change speed; these are not final runtime claims.
- Both actor initial hashes:
  `0fb94687886a89e11860fc4cc577328b42ba0f2949bc57828bd4ea095715249b`.
- Both critic initial hashes:
  `31fe2372fe73e5687dbb86d2cfe641668ca0d3a80a409991583dca8aaa57853b`.

## Run and recovery

```bash
bash scripts/run_e2b_rrnco_a6000.sh --execute
```

This uses nohup/setsid, checks both GPUs are free, verifies full preflight
evidence and launches only B01/B04. Source/config/data hashes and initialization
must match. Each worker uses frozen source. Best-test failure is recorded;
exit code0 alone does not establish acceptance. Formal test validates all1000
unique IDs, K50 and reported/recomputed costs, retaining failed rows if any.

Campaign in original checkout:
`results/e2/E2B_RRNCO_VRPTW100_S3009/`.

- `protocol.json`, `smoke_results.json`, `scheduler_a6000/status.json`.
- `runs/{ppo,slppo}/{console.log,train.jsonl,validation.jsonl,status.json}`.
- `runs/{ppo,slppo}/native_resolved.json`, `native_manifest.json`, frozen source.
- `runs/{ppo,slppo}/native_checkpoints/{best.pt,last.pt}` and verified backups.
- `runs/{ppo,slppo}/test/{manifest.json,instances.summary.json,instances.jsonl}`.
- `E2B_results.csv`: B01/B04 results after acceptance, B02/B03 left TBD.

Paired improvement is the mean of per-instance percentage improvements on
each pair's common feasible subset, not the ratio of means. Training time
includes validation and checkpoint overhead during the active training loop;
resumed downtime is excluded. Test timing is batch1 candidate generation plus
physical validation/selection. Results cover one training seed only.

Explicit worker recovery (after confirming the old worker has exited): use
the run's frozen `native_harness/scripts/run_e2b_rrnco_worker.py train
--run-dir <run> --resume` with the same native interpreter and GPU. Implicit
restart is refused. The supervisor itself does not automatically restart
failed work or schedule B02/B03.
