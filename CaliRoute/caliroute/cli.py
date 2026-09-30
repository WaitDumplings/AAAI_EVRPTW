from __future__ import annotations

import argparse
from pathlib import Path

from offline2online.trainer import train_from_config

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_N_TRAJ,
    DEFAULT_NUM_ENVS,
    DEFAULT_NUM_MINIBATCHES,
    DEFAULT_PPO_STEP_CHUNK,
    build_training_config,
    config_as_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CaliRoute PPO / SL-PPO / AWBC / DAPG with a paper-facing interface."
    )
    parser.add_argument("--problem", choices=["cvrp", "vrptw", "cvrptw", "evrptw"], default="evrptw")
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--charging-stations", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--expert-data", type=str, default=None)
    parser.add_argument("--expert-solution", "--expert-solution-path", dest="expert_solution", type=str, default=None)
    parser.add_argument("--gurobi-summary", "--gurobi-summary-path", dest="gurobi_summary", type=str, default=None)

    parser.add_argument("--offline-method", "--method", dest="offline_method", default="slppo")
    parser.add_argument("--pool", choices=["weighted", "best", "off"], default="weighted")
    parser.add_argument("--init-checkpoint", "--init-checkpoint-path", dest="init_checkpoint", type=str, default=None)
    parser.add_argument("--resume-checkpoint", "--resume-checkpoint-path", dest="resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume-start-epoch", type=int, default=None)

    parser.add_argument("--seed", type=int, default=3009)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-envs", "--num-envs-per-gpu", dest="num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--n-traj", type=int, default=DEFAULT_N_TRAJ)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-step-chunk-size", type=int, default=DEFAULT_PPO_STEP_CHUNK)
    parser.add_argument("--ppo-update-epochs", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=DEFAULT_NUM_MINIBATCHES)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-n-traj", type=int, default=DEFAULT_N_TRAJ)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--eval-save-routes", action="store_true")

    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--dde", choices=["on", "off"], default="on")
    parser.add_argument("--dde-heads", type=int, default=4)
    parser.add_argument("--qkv-delta", choices=["none", "k", "v", "kv"], default="none")
    parser.add_argument("--action-key", choices=["on", "off"], default="on")
    parser.add_argument("--action-bias", choices=["on", "off"], default="on")
    parser.add_argument("--distance-injection", choices=["encoder", "none"], default="encoder")

    parser.add_argument("--sl-coef", type=float, default=None)
    parser.add_argument("--sl-expert-candidate-weight", type=float, default=None)
    parser.add_argument("--bc-coef", type=float, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument("--bc-updates-per-epoch", type=int, default=None)
    parser.add_argument("--awbc-coef", type=float, default=None)

    parser.add_argument("--async-instance-prefetch", action="store_true")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-log-every", type=int, default=1)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--write-config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_training_config(args)
    rendered = config_as_yaml(cfg)
    if args.print_config or args.dry_run:
        print(rendered)
    if args.write_config:
        Path(args.write_config).write_text(rendered, encoding="utf-8")
    if args.dry_run:
        return
    ckpt = train_from_config(cfg, seed=args.seed, device=args.device, overrides=None)
    print(f"Saved final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
