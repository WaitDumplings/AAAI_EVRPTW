from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from .env_factory import make_terran_env
from .models import Agent
from .pbrs import PotentialRewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test TERRAN on the shared EVRPTW env.")
    parser.add_argument("--instance-path", "--instance_path", dest="instance_path", type=Path, required=True)
    parser.add_argument("--n-traj", "--n_traj", dest="n_traj", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=64)
    parser.add_argument("--n-encode-layers", "--n_encode_layers", dest="n_encode_layers", type=int, default=1)
    parser.add_argument("--no-customer-pbrs", "--no_customer_pbrs", dest="no_customer_pbrs", action="store_true")
    parser.add_argument("--use-repair-distance-pbrs", "--use_repair_distance_pbrs", dest="use_repair_distance_pbrs", action="store_true")
    parser.add_argument("--use-terminal-heuristic", "--use_terminal_heuristic", dest="use_terminal_heuristic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbrs_config = PotentialRewardConfig(
        use_customer_pbrs=not args.no_customer_pbrs,
        use_repair_distance_pbrs=args.use_repair_distance_pbrs,
        use_feasible_ratio_pbrs=False,
        use_terminal_heuristic=args.use_terminal_heuristic,
    )
    env = make_terran_env(
        instance_path=args.instance_path,
        n_traj=args.n_traj,
        pbrs_config=pbrs_config,
    )
    obs, info = env.reset(seed=args.seed)
    agent = Agent(
        embedding_dim=args.embedding_dim,
        n_encode_layers=args.n_encode_layers,
        device="cpu",
    )
    action, logits = agent(obs)
    action = action.squeeze(0)
    obs, reward, terminated, truncated, info = env.step(action.detach().cpu().tolist())
    print(
        {
            "logits_shape": list(logits.shape),
            "action_shape": list(action.shape),
            "reward": reward.tolist(),
            "terminated": terminated.tolist(),
            "truncated": truncated.tolist(),
            "reward_components": {
                key: value.tolist() for key, value in info["reward_components"].items()
            },
        }
    )


if __name__ == "__main__":
    main()
