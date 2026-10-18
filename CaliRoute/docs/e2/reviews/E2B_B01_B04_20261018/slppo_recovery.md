# B04 missing-depot recovery (2026-10-18 UTC)

Runtime fix: `81336d5`, branch `experiment/e2b-rrnco-ppo-slppo-20261018`.

The first B04 attempt completed 16 iterations and failed during iteration 17
with `Teacher-forced route ends before completion`. Native RRNCO completion
requires every visited bit, including the depot. Removing the expert's final
depot made single-tour expert replay depend on another route's batch padding.
The pool has 553 such single-tour solutions among 5,000 train experts.

The adapter now retains the explicit final depot, while still omitting the
external initial depot. No route is repaired or reordered. Native model,
physical masks, distances, E1 solution-level formulas and experiment protocol
are unchanged. Already terminated routes exclude padded depot actions from
their log probability, value and physical-cost reductions.

All 5,000 experts passed native mask, completion and physical-cost checks.
Maximum absolute cost discrepancy: 0.000184476376 km.
A singleton single-tour test reproduced the old missing-depot failure and passed
with the explicit final depot through the actual native encoder and decoder.
Full B64/S50 preflight passed, including expert gradients, update effect,
checkpoint restore and independent physical validation. Three unit tests passed.
See `slppo_recovery_preflight.json` for measured checks and hashes.

B04 restarts **from scratch** with seed 3009, 1,500 online iterations, batch 64,
50 trajectories and four PPO passes. No failed-attempt weights, optimizer state,
incumbent state or iteration counts are reused. Native source and runtime harness
are frozen again; checkpoints bind their actual manifests. B01 PPO continues
uninterrupted on GPU0 with its original frozen runtime and process 2745804.
The replacement supervisor adopts B01 and runs B04 on GPU1, with automatic best
checkpoint test on completion. B02 and B03 are neither started nor queued.

Archive mappings preserve the original failed run, checkpoints and preflight.
Archived configs retain original paths as historical evidence; these archived
attempts are not runnable in place without explicit relocation.

```json
{
  "adopted_ppo_pid": 2745804,
  "archived_paths": {
    "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/checkpoint_backups/formal/slppo": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/failed_attempts/slppo_missing_final_depot_20261018T224924Z/checkpoint_backups/formal/slppo",
    "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/checkpoint_backups/preflight/slppo": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/failed_attempts/slppo_missing_final_depot_20261018T224924Z/checkpoint_backups/preflight/slppo",
    "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/preflight/slppo": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/failed_attempts/slppo_missing_final_depot_20261018T224924Z/preflight/slppo",
    "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/runs/slppo": "/data/Maojie/AAAI/AAAI_EVRPTW/CaliRoute/results/e2/E2B_RRNCO_VRPTW100_S3009/failed_attempts/slppo_missing_final_depot_20261018T224924Z/runs/slppo"
  },
  "expert_audit": {
    "instances": 5000,
    "maximum_cost_error_km": 0.00018447637557983398,
    "missing_depot_regression_reproduced": true,
    "passed": true,
    "single_tour_instances": 553,
    "singleton_actor_replay": true
  },
  "git_head": "81336d5bc4d02efda32d2f1908080fb2288ddb8b",
  "initialization": "scratch; no failed-attempt weights or optimizer state reused",
  "original_supervisor": 2745712,
  "preflight_sha256": "d079e5acefec2051be1116d21f4db89cb65efe4059773f6ada337d20ef42f4d2",
  "reason": "Retain explicit final depot in native expert replay",
  "started_at": "2026-10-18T22:49:25.087934+00:00"
}
```

The old `slppo_preflight.json` remains evidence of the original two-iteration
preflight; it did not expose the single-tour terminal edge case. The new audit
is the additional required acceptance gate for the repaired B04 run. Failed
attempt wall time is excluded from the fresh formal run's training time and
is retained separately in the archive.
