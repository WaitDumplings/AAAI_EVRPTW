# TERRAN

TERRAN is the reinforcement-learning baseline package for EVRPTW-DB. It uses the
shared `EVRPTW_Env` Gymnasium-style environment and keeps POMO-style parallel
rollouts through the environment's `n_traj` dimension.

## Components

- `models/`: migrated TERRAN attention backbone, actor, and critic.
- `env_factory.py`: creates the shared EVRPTW environment with optional TERRAN
  reward shaping.
- `data_pool.py`: online service-territory pool for training-time instance sampling.
- `pbrs.py`: optional potential-based reward shaping switches.
- `train.py`: PPO-style TERRAN training entry point.
- `eval.py`: fixed-dataset best-of-`n_traj` sample evaluation.
- `prepare_eval_data.py`: fixed Cus15 eval-set generation.
- `smoke_test_terran.py`: verifies the model and environment interface on a
  pickle instance.


## Data Layout

Run the module commands below from `AAAI_EVRPTW/CaliRoute`. Fixed training and
validation paths in the YAML configs resolve relative to `CaliRoute`, so the
released data should be extracted alongside the code:

```text
AAAI_EVRPTW/
├── CaliRoute/
└── AAAI_Dataset/                 # Download separately; excluded from Git.
    ├── dataset/evrptw/
    │   ├── train/{Cus15,Cus50,Cus100}/
    │   └── val/{Cus15,Cus50,Cus100}/
    └── test_release/evrptw/test/{Cus15,Cus50,Cus100}/
```

The fixed-data configs are `cus15_terran.yaml`, `cus15_terran_pbrs.yaml`, and
`cus50_terran*_val_ppo*.yaml` (including `cus50_terran_val_ppo.yaml`). Use
`--train-dataset-path` and `--eval-path` to override their locations. Training uses
`dataset/evrptw/train`; periodic model selection uses `dataset/evrptw/val`.
Reserve `test_release` for final evaluation.

Cus5 configurations are legacy examples: the release contains no Cus5 split.
They require separately prepared Cus5 data at the configured locations, or
explicit path overrides.

## Optional Online Generation

The Cus5, Cus50, and Cus100 configs with `online_training: true` retain their
original online-generation behavior. They require external generator resources,
including `CaliRoute/EVRPTW_Dataset_Generator/configs/amazon_hierarchy.yaml` and any
raw assets referenced by that configuration. Those resources and the generator
preparation CLI are not bundled with this repository; downloading `AAAI_Dataset`
alone does not make these legacy online examples runnable. Fixed-data training
does not require that external configuration.

After providing those resources, a compatible precomputed territory pool can be
selected with an absolute path:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus50_terran.yaml \
  --seed 1515 \
  --territory-pool-path /path/to/generated/service_territory_pool
```

`mother_board_pool_size` remains the backward-compatible config key for the number
of active service territories held by one run. `territory_pool_path` defaults to
`null`: the generator prepares territories online. If an explicitly supplied pool
cannot be loaded or has too few territories, training also falls back to online
generation. The `cycle` replacement policy reuses a successfully loaded pool.

`prepare_eval_data.py` also requires the external generator configuration. Its
default output is `CaliRoute/results/generated_data/terran_eval`; generated samples
are separate from the released validation and frozen test data. Override
`--save-path` to choose a different scratch output directory.

## PBRS Switches

`PotentialRewardConfig` exposes the reward-shaping controls without modifying
the shared environment:

- `use_customer_pbrs`: served-customer progress potential using
  `gamma * Phi(s_next) - Phi(s)`.
- `use_repair_distance_pbrs`: single-customer depot-customer-depot repair
  workload potential using the same gamma potential-difference form.
- `use_feasible_ratio_pbrs`: feasible-unserved-customer ratio potential from
  the action mask. This is optional and disabled in the default PBRS configs.
- `use_terminal_heuristic`: terminal success bonus and failure penalty. This is
  an auxiliary shaping term, not strict PBRS, and is disabled by default.
- `customer_pbrs_mode`: default configs use `progress`, the strict gamma
  potential-difference form.

Evaluation should usually disable PBRS and use the base objective reward. PBRS is
intended for training only.

## Cus15 Baselines

The default Cus15 setup trains two baselines with identical architecture and
hyperparameters:

- `configs/cus15_terran.yaml`: base TERRAN, PBRS disabled.
- `configs/cus15_terran_pbrs.yaml`: TERRAN+PBRS with customer-progress,
  repair-distance progress, and terminal heuristic enabled.

Training samples from the fixed Cus15/CS3 training split. Evaluation uses the
fixed validation split and sample decoding: each instance runs `n_traj=50`
trajectories and keeps the best feasible trajectory by objective distance.


## Normalization And Training Metrics

The shared RL environment keeps physical dynamics in seconds, kilometers, kWh,
and cm3, but model-facing observations are normalized: locations are mapped to
`[0, 1]`, demand and current load are fractions of vehicle capacity, time
windows/service/current time are fractions of the operating horizon, battery
state is a fraction of battery capacity, and the model-facing capacity scalars
are `1.0`. Training rewards are distance-normalized for value-function stability;
`objective_distance_km` in `info` and eval CSVs remains the physical kilometer
objective used for benchmark comparison.

## Periodic Evaluation

`configs/cus15_terran.yaml` and `configs/cus15_terran_pbrs.yaml` run fixed-set
evaluation every `eval_interval` epochs. The default uses the fixed Cus15/CS3
validation set with 50 sampled trajectories (`eval_n_traj: 50`). Evaluation metrics are
written into both `train_log.csv` and `eval_log.csv`:

- `eval_avg_objective_distance_km`
- `eval_avg_vehicle_count`
- `eval_feasible_rate`
- `eval_avg_runtime_s`

## Example

From `AAAI_EVRPTW/CaliRoute`, train on the downloaded fixed dataset:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran.yaml --seed 1515
```

Evaluate a saved checkpoint on the final test split:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.eval \
  --checkpoint-path /path/to/checkpoint.pt \
  --eval-path ../AAAI_Dataset/test_release/evrptw/test/Cus15 \
  --num-customers 15 --num-charging-stations 3 \
  --n-traj 50 --output-dir results/terran_test/Cus15
```
