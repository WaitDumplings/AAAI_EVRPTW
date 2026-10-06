#!/usr/bin/env python
"""Launch a full frozen test bundle or one EVRPTW Cus100 server shard."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.resume import read_completed_ids


EVRPTW_CS_COPIES = {15: 1, 50: 2, 100: 2}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen test bundle (or a 200-instance EVRPTW Cus100 shard) with 30 workers, "
            "1 Gurobi thread per worker, and a 7200-second solve limit per instance."
        ),
    )
    parser.add_argument("problem", type=str.lower, choices=("vrptw", "evrptw", "cvrp"))
    parser.add_argument("scale", type=int, choices=(15, 50, 100))
    parser.add_argument(
        "shard", nargs="?", type=int, choices=range(1, 6),
        help="Optional server shard 1-5 for EVRPTW Cus100; each receives 200 fixed instances.",
    )
    args = parser.parse_args(argv)
    if args.shard is not None and (args.problem != "evrptw" or args.scale != 100):
        parser.error("Server sharding is supported only for evrptw 100.")

    bundle_dir = (
        CODE_ROOT.parent / "AAAI_Dataset" / "test_release"
        / args.problem / "test" / f"Cus{args.scale}"
    )
    bundle = bundle_dir / "instances.pkl"
    metadata_path = bundle_dir / "metadata.json"
    runner = (
        CODE_ROOT / "EVRPTW_Benchmark" / "Exact" / "Gurobi_Solver"
        / args.problem.upper() / "run_gurobi.py"
    )
    for path in (bundle, metadata_path, runner):
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read test metadata: {exc}")
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != expected
        for key, expected in (
            ("split", "test"), ("num_customers", args.scale), ("num_instances", 1000),
        )
    ):
        parser.error(f"Expected metadata for test/Cus{args.scale} with 1000 instances: {metadata_path}")
    if importlib.util.find_spec("gurobipy") is None:
        parser.error(f"gurobipy is not installed in the active Python environment: {sys.executable}")

    output = CODE_ROOT / "results" / "gurobi" / args.problem / "test" / f"Cus{args.scale}"
    run_label = f"{args.problem}/Cus{args.scale}"
    selection_message = "Selection: full test bundle (1000 instances)"
    if args.shard is not None:
        output = output / f"shard_{args.shard}_of_5"
        start_offset = (args.shard - 1) * 200
        end_offset = args.shard * 200
        run_label += f"/shard_{args.shard}_of_5"
        selection_message = (
            f"Selection: shard={args.shard}/5; assigned=200; "
            f"bundle positions=[{start_offset},{end_offset}) before resume"
        )
    command = [
        sys.executable, "-u", str(runner),
        "--dataset_path", str(bundle), "--save_path", str(output),
        "--workers", "30", "--threads", "1", "--time_limit_s", "7200",
        "--mip_gap", "0.0", "--checkpoints_s", "60,300,900,3600,7200",
        "--skip_completed", "--save_traceback", "--verbose",
    ]
    if args.problem == "evrptw":
        # The legacy tie-break starts a second optimize call with a fresh time limit.
        command.extend([
            "--reference_split", "test", "--cs_copies", str(EVRPTW_CS_COPIES[args.scale]),
            "--no-tie_break_vehicle_count",
        ])

    if args.shard is not None:
        command.extend(["--test_shard", str(args.shard)])

    output.mkdir(parents=True, exist_ok=True)
    pid_path = output / "launcher.pid"
    with (output / ".test.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"This run is already active: {run_label}. PID file: {pid_path}")
        summary_path = output / "gurobi_summary.csv"
        try:
            completed = read_completed_ids(summary_path)
        except (OSError, ValueError, csv.Error) as exc:
            parser.error(f"Cannot check existing results: {exc}")
        resume_message = (
            f"Resume: enabled; summary={summary_path}; recorded_completed={len(completed)}\n"
            "Matching instance IDs will be skipped; exact skipped/pending counts appear in the log before solving."
        )
        print(selection_message)
        print(resume_message, flush=True)
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = logs / f"test_{stamp}.log"
        environment = os.environ.copy()
        # Avoid 30 workers each starting a separate numerical-library thread pool.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"Test: {run_label}; metadata instances=1000\n"
                f"{selection_message}\n"
                "Workers=30; threads per worker=1; Gurobi time limit per instance=7200s\n"
                f"{resume_message}\n"
                f"Command: {shlex.join(command)}\n\n"
            )
            log.flush()
            try:
                child = subprocess.Popen(
                    command, cwd=CODE_ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock.fileno(),),
                )
            except OSError as exc:
                parser.error(f"Could not start Gurobi: {exc}; log: {log_path}")
        # The child keeps the lock until it exits, including after this launcher exits.
        pid_path.write_text(f"{child.pid}\n", encoding="utf-8")
        log_path.with_suffix(".pid").write_text(f"{child.pid}\n", encoding="utf-8")

    print(f"Started {run_label}: PID={child.pid} (30 workers, 7200s/instance)")
    print(f"Input: {bundle} (completed results in this selection are skipped)")
    print(f"Results: {output}")
    print(f"Log: {log_path}")
    print(f"Follow: tail -f {shlex.quote(str(log_path))}")
    print(f"Stop this batch: kill -TERM -- -{child.pid}")


if __name__ == "__main__":
    main()
