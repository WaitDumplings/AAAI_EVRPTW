from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import iter_instances
from .env_factory import make_terran_env
from .models import Agent
from .rollout import rollout_eval_batch


def _eval_instance_batches(
    eval_path: Path,
    num_customers: int,
    num_charging_stations: int,
    batch_size: int,
    limit: int | None = None,
    num_batches_limit: int | None = None,
):
    max_count = None if limit is None else int(limit)
    if num_batches_limit is not None:
        by_batches = max(1, int(batch_size)) * int(num_batches_limit)
        max_count = by_batches if max_count is None else min(max_count, by_batches)
    batch = []
    seen = 0
    for instance in iter_instances(eval_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        if max_count is not None and seen >= max_count:
            break
        batch.append(instance)
        seen += 1
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TERRAN checkpoint with sample best-of-n_traj decoding.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-customers", type=int, default=15)
    parser.add_argument("--num-charging-stations", type=int, default=3)
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--decode-mode", type=str, default="sample", choices=["sample", "greedy"])
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-batches", type=int, default=None)
    parser.add_argument("--info-level", type=str, choices=["light", "full"], default="full")
    parser.add_argument("--save-routes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    solver_name = args.solver_name or str(cfg.get("run_name", "TERRAN"))
    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 3)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", False)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    rows = []
    batch_size = max(1, int(args.eval_batch_size))
    env_info_level = "full" if args.save_routes else args.info_level
    seen_before_batch = 0
    for instances in _eval_instance_batches(
        args.eval_path,
        args.num_customers,
        args.num_charging_stations,
        batch_size,
        args.limit,
        args.eval_num_batches,
    ):
        envs = [make_terran_env(instance=instance, n_traj=args.n_traj, info_level=env_info_level) for instance in instances]
        batch_rows = rollout_eval_batch(
            agent,
            envs,
            decode_mode=args.decode_mode,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed + seen_before_batch,
            include_routes=args.save_routes,
        )
        for instance, row in zip(instances, batch_rows):
            row.update(
                {
                    "instance_id": instance.instance_id,
                    "solver_name": solver_name,
                    "seed": args.seed,
                    "checkpoint": str(args.checkpoint_path),
                    "decode_mode": args.decode_mode,
                    "n_traj": args.n_traj,
                    "eval_batch_size": batch_size,
                    "eval_info_level": env_info_level,
                    "save_routes": args.save_routes,
                }
            )
        rows.extend(batch_rows)
        seen_before_batch += len(instances)

    if not rows:
        raise FileNotFoundError(f"No EVRPTW instances found under {args.eval_path}")

    feasible_rows = [row for row in rows if row["feasible"]]
    summary = {
        "solver_name": solver_name,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint_path),
        "decode_mode": args.decode_mode,
        "n_traj": args.n_traj,
        "eval_batch_size": max(1, int(args.eval_batch_size)),
        "eval_num_batches": int(np.ceil(len(rows) / max(1, int(args.eval_batch_size)))),
        "eval_info_level": env_info_level,
        "save_routes": args.save_routes,
        "num_instances": len(rows),
        "feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "avg_objective_distance_km": float(np.mean([row["objective_distance_km"] for row in feasible_rows])) if feasible_rows else float("nan"),
        "avg_vehicle_count": float(np.mean([row["vehicle_count"] for row in feasible_rows])) if feasible_rows else float("nan"),
        "avg_runtime_s": float(np.mean([row["runtime_s"] for row in rows])),
    }
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "EVRPTW_Benchmark/results/AC_v1" / f"Cus_{args.num_customers}" / f"CS_{args.num_charging_stations}" / solver_name
    _write_csv(output_dir / "terran_routes.csv", rows)
    _write_csv(output_dir / "terran_summary.csv", [summary])
    print(summary)


if __name__ == "__main__":
    main()
