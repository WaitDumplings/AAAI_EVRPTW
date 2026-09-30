from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "EVRPTW_Benchmark.Exact.Gurobi_Solver.CVRP"

from ..paths import resolve_run_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multiprocessing Gurobi shard for AAAI_Dataset CVRP.")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "eval", "test"])
    parser.add_argument("--scale", default="Cus15")
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit_s", type=float, default=7200.0)
    parser.add_argument("--checkpoints_s", default="60,300,900,3600,7200")
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--no_skip_completed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and runner arguments without loading data or Gurobi.")
    args = parser.parse_args()

    dataset_path, output_path = resolve_run_paths(
        problem="cvrp",
        split=args.split,
        scale=args.scale,
        dataset_path=args.dataset_path,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
    )
    cmd = [
        "--dataset_path", str(dataset_path),
        "--save_path", str(output_path),
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--workers", str(args.workers),
        "--threads", str(args.threads),
        "--time_limit_s", str(args.time_limit_s),
        "--checkpoints_s", args.checkpoints_s,
        "--mip_gap", str(args.mip_gap),
    ]
    if not args.no_skip_completed:
        cmd.append("--skip_completed")
    if args.verbose:
        cmd.append("--verbose")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Shard: split={args.split} scale={args.scale} index=[{args.start_index}, {args.end_index}) workers={args.workers} threads={args.threads}")
    print(f"Runner arguments: {shlex.join(cmd)}")
    if args.dry_run:
        return
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    from .run_gurobi import main as run_gurobi_main

    run_gurobi_main(cmd)


if __name__ == "__main__":
    main()
