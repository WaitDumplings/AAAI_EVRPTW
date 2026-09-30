from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import load_instance
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv


def _choose_actions(obs: dict[str, np.ndarray]) -> np.ndarray:
    mask = obs["action_mask"]
    last = obs["last_node_idx"]
    actions = np.zeros(mask.shape[0], dtype=np.int64)
    for traj_idx in range(mask.shape[0]):
        feasible = np.flatnonzero(mask[traj_idx])
        if feasible.size == 0:
            actions[traj_idx] = 0
            continue
        customer_feasible = feasible[feasible > 0]
        customer_feasible = customer_feasible[customer_feasible < obs["demand"].shape[0]]
        customer_feasible = customer_feasible[obs["demand"][customer_feasible] > 0]
        if customer_feasible.size:
            actions[traj_idx] = int(customer_feasible[0])
        elif 0 in feasible and last[traj_idx] != 0:
            actions[traj_idx] = 0
        else:
            actions[traj_idx] = int(feasible[0])
    return actions


def run_smoke_test(instance_path: Path, n_traj: int, seed: int, max_steps: int) -> None:
    instance = load_instance(instance_path)
    env = EVRPTWVectorEnv(instance=instance, n_traj=n_traj)
    obs, info = env.reset(seed=seed)
    assert obs["action_mask"].shape == (n_traj, instance.num_terminals)
    assert info["action_mask"].shape == (n_traj, instance.num_terminals)

    for _ in range(max_steps):
        actions = _choose_actions(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert reward.shape == (n_traj,)
        assert terminated.shape == (n_traj,)
        assert truncated.shape == (n_traj,)
        if np.all(terminated | truncated):
            break

    print(
        {
            "instance_id": instance.instance_id,
            "n_traj": n_traj,
            "served_customers": info["served_customers"].tolist(),
            "success": info["success"].tolist(),
            "vehicle_count": info["vehicle_count"].tolist(),
            "objective_distance_km": np.round(info["objective_distance_km"], 6).tolist(),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the shared EVRPTW RL env.")
    parser.add_argument("--instance_path", type=Path, required=True)
    parser.add_argument("--n_traj", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--max_steps", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_smoke_test(args.instance_path, args.n_traj, args.seed, args.max_steps)


if __name__ == "__main__":
    main()
