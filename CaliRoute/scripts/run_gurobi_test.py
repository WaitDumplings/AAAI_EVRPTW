#!/usr/bin/env python
"""Launch one complete frozen test bundle in a detached Gurobi process pool."""
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all 1000 frozen test instances in the background, with 30 workers, "
            "1 Gurobi thread per worker, and a 7200-second solve limit per instance."
        ),
    )
    parser.add_argument("problem", type=str.lower, choices=("vrptw", "evrptw", "cvrp"))
    parser.add_argument("scale", type=int, choices=(15, 50, 100))
    args = parser.parse_args(argv)

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
            "--reference_split", "test", "--cs_copies", "4",
            "--no-tie_break_vehicle_count",
        ])

    output.mkdir(parents=True, exist_ok=True)
    pid_path = output / "launcher.pid"
    with (output / ".test.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"This task/scale is already running. PID file: {pid_path}")
        summary_path = output / "gurobi_summary.csv"
        try:
            completed = read_completed_ids(summary_path)
        except (OSError, ValueError, csv.Error) as exc:
            parser.error(f"Cannot check existing results: {exc}")
        resume_message = (
            f"Resume: enabled; summary={summary_path}; recorded_completed={len(completed)}\n"
            "Matching instance IDs will be skipped; exact skipped/pending counts appear in the log before solving."
        )
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
                f"Test: {args.problem}/Cus{args.scale}; metadata instances=1000\n"
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

    print(f"Started {args.problem}/Cus{args.scale}: PID={child.pid} (30 workers, 7200s/instance)")
    print(f"Input: {bundle} (1000 instances; completed results are skipped)")
    print(f"Results: {output}")
    print(f"Log: {log_path}")
    print(f"Follow: tail -f {shlex.quote(str(log_path))}")
    print(f"Stop this batch: kill -TERM -- -{child.pid}")


if __name__ == "__main__":
    main()
