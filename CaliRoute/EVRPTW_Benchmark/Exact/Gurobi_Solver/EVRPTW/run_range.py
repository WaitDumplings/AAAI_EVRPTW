from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "EVRPTW_Benchmark.Exact.Gurobi_Solver.EVRPTW"


from ..paths import resolve_run_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multiprocessing Gurobi shard for EVRPTW instances. "
            "Example: Cus15 train instances with numeric suffixes [100, 200)."
        )
    )
    parser.add_argument("--dataset_path", default="", help="Split dataset directory or a single pickle file. Overrides --dataset_root/--split.")
    parser.add_argument("--dataset_root", default=None, help="Problem-specific root containing split/CusN directories; defaults to AAAI_Dataset/dataset/evrptw, or AAAI_Dataset/test_release/evrptw for test.")
    parser.add_argument("--split", default="val", choices=["train", "val", "eval", "test"], help="Dataset split when --dataset_path is not provided.")
    parser.add_argument("--scale", default="Cus15", help="Scale to run, e.g. Cus5, Cus15, Cus50.")
    parser.add_argument("--start_index", type=int, required=True, help="Inclusive numeric instance suffix start.")
    parser.add_argument("--end_index", type=int, required=True, help="Exclusive numeric instance suffix end.")
    parser.add_argument("--output_path", default="", help="Output directory for gurobi_summary.csv, time trace, and solution pickles.")
    parser.add_argument("--reference_output_path", default="", help="Optional reference_solutions root for split/solutions.csv and routes/*.json.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker processes.")
    parser.add_argument("--threads", type=int, default=1, help="Gurobi threads per worker.")
    parser.add_argument("--cs_copies", type=int, default=2, help="Charging-station dummy copies per station.")
    parser.add_argument("--time_limit_s", type=float, default=7200.0, help="Gurobi optimize-call time limit in seconds; hard-capped at 7200.")
    parser.add_argument("--checkpoints_s", default="60,300,900,3600,7200", help="Comma-separated incumbent checkpoint seconds.")
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--no_skip_completed", action="store_true", help="Re-solve completed rows instead of resuming.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and runner arguments without loading data or Gurobi.")
    args = parser.parse_args()

    dataset_path, output_path = resolve_run_paths(
        problem="evrptw",
        split=args.split,
        scale=args.scale,
        dataset_path=args.dataset_path,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
    )

    gurobi_args = [
        "--dataset_path", str(dataset_path),
        "--save_path", str(output_path),
        "--reference_split", args.split,
        "--scales", args.scale,
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--workers", str(args.workers),
        "--threads", str(args.threads),
        "--cs_copies", str(args.cs_copies),
        "--time_limit_s", str(args.time_limit_s),
        "--checkpoints_s", args.checkpoints_s,
        "--mip_gap", str(args.mip_gap),
    ]
    if args.reference_output_path:
        gurobi_args.extend(["--reference_save_path", str(Path(args.reference_output_path).resolve())])
    if not args.no_skip_completed:
        gurobi_args.append("--skip_completed")
    if args.verbose:
        gurobi_args.append("--verbose")

    print("EVRPTW core: bundled legacy evrptw_core")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Shard: split={args.split} scale={args.scale} index=[{args.start_index}, {args.end_index}) workers={args.workers}")
    if args.dry_run:
        print(f"Runner arguments: {shlex.join(gurobi_args)}")
        return
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    from .run_gurobi import main as run_gurobi_main

    run_gurobi_main(gurobi_args)


if __name__ == "__main__":
    main()
