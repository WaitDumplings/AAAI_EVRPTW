# EVRPTW RL Environment

`EVRPTWVectorEnv` is the shared Gymnasium-style environment for reinforcement
learning baselines in this benchmark. It loads the canonical pickle
`EVRPTWInstance` schema and keeps the vectorized `n_traj` rollout dimension used
by POMO-style methods.

## API

```python
from evrptw_core.io import load_instance
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv

instance = load_instance("EVRPTW_Dataset/.../instance_000000.pkl")
env = EVRPTWVectorEnv(instance, n_traj=8)

obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(actions)
```

The environment follows the Gymnasium return convention:

- `reset(...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

`reward`, `terminated`, and `truncated` are arrays with shape `(n_traj,)`.

## Action Space

Node convention is shared with the benchmark solvers:

- `0`: depot
- `1..N`: customers
- `N+1..N+M`: charging stations

The action passed to `step` is an integer array with shape `(n_traj,)`. The
current feasibility mask is available in `obs["action_mask"]` and
`info["action_mask"]`.

## Battery And Charging

The environment keeps legacy DRL compatibility by exposing
`obs["current_battery"]` as the consumed battery fraction since the last full
charge. `obs["remaining_battery"]` is also provided for models that prefer
remaining capacity.

Charging station actions perform full charging. The default benchmark mode is
`charging_mode="fixed_full"`, so every station visit pays the same full-charge
duration. This matches the generator and exact-solver semantics. For ablations,
`charging_mode="proportional_full"` is also available:

```text
charge_time = consumed_kWh / battery_capacity_kWh * full_charge_time_s
```

## Route Export

`info["routes"]` stores per-vehicle routes. `info["route_sequence"]` stores the
benchmark-wide merged route sequence, for example:

```text
[[0, 3, 2, 1, 0], [0, 7, 5, 0]] -> [0, 3, 2, 1, 0, 7, 5, 0]
```
