from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "EVRPTW_Benchmark.Exact.Gurobi_Solver.CVRP"

from .classical_core.io import iter_instances, save_solution
from .classical_core.schema import ClassicalVRPSolution, solution_route_sequence
from .gurobi_solver import GurobiCVRPSolver, GurobiSolverConfig

SUMMARY_FIELDNAMES = [
    "instance_id", "file", "status_name", "feasible", "objective_distance_km",
    "vehicle_count", "runtime_s", "mip_gap", "best_bound", "routes_json",
    "route_sequence_json", "solution_path", "time_trace_path", "errors", "traceback",
]
TIME_TRACE_FIELDNAMES = [
    "instance_id", "file", "checkpoint_s", "elapsed_s", "reached_checkpoint", "status",
    "has_incumbent", "objective_distance_km", "best_bound", "mip_gap", "vehicle_count",
    "routes_json", "route_sequence_json", "checkpoint_solution_path", "source", "errors",
]


def parse_checkpoints(raw: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return tuple(sorted(set(values)))


def instance_index(instance_id: str) -> int | None:
    try:
        return int(str(instance_id).rsplit("_", 1)[1])
    except Exception:
        return None


def checkpoint_label(checkpoint_s: float | int | None) -> str:
    if checkpoint_s is None:
        return "final"
    value = float(checkpoint_s)
    return f"{int(value)}s" if value.is_integer() else f"{value:g}s".replace(".", "p")


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], sort_key) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert(rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], instance_id: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("instance_id")) != instance_id] + new_rows


def write_checkpoint_solution(instance_id: str, snapshot: dict[str, Any], checkpoint_dir: Path) -> str:
    if not snapshot.get("has_incumbent"):
        return ""
    solution = ClassicalVRPSolution(
        instance_id=instance_id,
        solver_name="gurobi_cvrp_arcflow",
        routes=snapshot.get("routes", []),
        objective_distance_km=snapshot.get("objective_distance_km"),
        vehicle_count=snapshot.get("vehicle_count"),
        runtime_s=snapshot.get("elapsed_s"),
        feasible=True,
        metadata={"checkpoint_s": snapshot.get("checkpoint_s"), "best_bound": snapshot.get("best_bound"), "mip_gap": snapshot.get("mip_gap")},
    )
    path = checkpoint_dir / f"{instance_id}_{checkpoint_label(snapshot.get('checkpoint_s'))}_solution.pkl"
    save_solution(path, solution)
    return str(path)


def append_time_rows(rows: list[dict[str, Any]], instance_file: Path, solution: ClassicalVRPSolution, checkpoint_dir: Path) -> None:
    for snap in solution.metadata.get("checkpoint_snapshots", []):
        rows.append({
            "instance_id": solution.instance_id,
            "file": str(instance_file),
            "checkpoint_s": snap.get("checkpoint_s"),
            "elapsed_s": snap.get("elapsed_s"),
            "reached_checkpoint": snap.get("reached_checkpoint"),
            "status": snap.get("solver_status"),
            "has_incumbent": snap.get("has_incumbent"),
            "objective_distance_km": snap.get("objective_distance_km"),
            "best_bound": snap.get("best_bound"),
            "mip_gap": snap.get("mip_gap"),
            "vehicle_count": snap.get("vehicle_count"),
            "routes_json": json.dumps(snap.get("routes", [])),
            "route_sequence_json": json.dumps(snap.get("route_sequence", [])),
            "checkpoint_solution_path": write_checkpoint_solution(solution.instance_id, snap, checkpoint_dir),
            "source": snap.get("source"),
            "errors": "",
        })


def solve_task(instance, instance_file: str, cfg: GurobiSolverConfig, checkpoint_dir: str, save_traceback: bool) -> dict[str, Any]:
    try:
        solver = GurobiCVRPSolver(cfg)
        solution = solver.solve(instance)
        return {"instance_id": instance.instance_id, "solution": solution, "instance_file": instance_file, "error": "", "traceback": ""}
    except Exception as exc:
        return {
            "instance_id": instance.instance_id,
            "solution": None,
            "instance_file": instance_file,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc() if save_traceback else "",
        }


def summary_row(instance_file: Path, solution: ClassicalVRPSolution, solution_path: Path, trace_path: Path) -> dict[str, Any]:
    meta = solution.metadata
    return {
        "instance_id": solution.instance_id,
        "file": str(instance_file),
        "status_name": meta.get("status"),
        "feasible": solution.feasible,
        "objective_distance_km": solution.objective_distance_km,
        "vehicle_count": solution.vehicle_count,
        "runtime_s": solution.runtime_s,
        "mip_gap": meta.get("mip_gap"),
        "best_bound": meta.get("best_bound"),
        "routes_json": json.dumps(solution.routes),
        "route_sequence_json": json.dumps(solution_route_sequence(solution.routes)),
        "solution_path": str(solution_path),
        "time_trace_path": str(trace_path),
        "errors": "",
        "traceback": "",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run exact Gurobi CVRP solver on AAAI_Dataset pickle bundles.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--time_limit_s", type=float, default=7200.0)
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--output_flag", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--checkpoints_s", default="60,300,900,3600,7200")
    parser.add_argument("--save_traceback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    summary_path = save_path / "gurobi_summary.csv"
    trace_path = save_path / "gurobi_time_trace.csv"
    solution_dir = save_path / "solutions"
    checkpoint_dir = solution_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = {r["instance_id"] for r in read_rows(summary_path)} if args.skip_completed else set()
    records = []
    for inst in iter_instances(dataset_path):
        idx = instance_index(inst.instance_id)
        if args.start_index is not None and (idx is None or idx < args.start_index):
            continue
        if args.end_index is not None and (idx is None or idx >= args.end_index):
            continue
        if inst.instance_id in completed:
            continue
        records.append((dataset_path, inst))
        if args.limit is not None and len(records) >= int(args.limit):
            break

    cfg = GurobiSolverConfig(
        time_limit_s=args.time_limit_s,
        mip_gap=args.mip_gap,
        output_flag=args.output_flag,
        checkpoints_s=parse_checkpoints(args.checkpoints_s),
        threads=args.threads,
    )
    print(f"Loaded {len(records)} CVRP instances. workers={args.workers} threads={args.threads} output={save_path}")
    summary_rows = read_rows(summary_path)
    time_rows = read_rows(trace_path)

    def handle(result: dict[str, Any]) -> None:
        nonlocal summary_rows, time_rows
        iid = result["instance_id"]
        if result["solution"] is None:
            row = {"instance_id": iid, "file": result["instance_file"], "status_name": "ERROR", "feasible": False, "errors": result["error"], "traceback": result["traceback"]}
            summary_rows = upsert(summary_rows, [row], iid)
            write_csv_atomic(summary_path, summary_rows, SUMMARY_FIELDNAMES, sort_key=lambda r: str(r.get("instance_id", "")))
            return
        sol: ClassicalVRPSolution = result["solution"]
        sol_path = solution_dir / f"{iid}_solution.pkl"
        save_solution(sol_path, sol)
        row = summary_row(Path(result["instance_file"]), sol, sol_path, trace_path)
        new_time: list[dict[str, Any]] = []
        append_time_rows(new_time, Path(result["instance_file"]), sol, checkpoint_dir)
        summary_rows = upsert(summary_rows, [row], iid)
        time_rows = upsert(time_rows, new_time, iid)
        write_csv_atomic(summary_path, summary_rows, SUMMARY_FIELDNAMES, sort_key=lambda r: str(r.get("instance_id", "")))
        write_csv_atomic(trace_path, time_rows, TIME_TRACE_FIELDNAMES, sort_key=lambda r: (str(r.get("instance_id", "")), float(r.get("checkpoint_s") or 1e30)))
        if args.verbose:
            print(f"{iid}: {row.get('status_name')} obj={row.get('objective_distance_km')} gap={row.get('mip_gap')}")

    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(solve_task, inst, str(path), cfg, str(checkpoint_dir), args.save_traceback) for path, inst in records]
            for fut in as_completed(futures):
                handle(fut.result())
    else:
        for path, inst in records:
            handle(solve_task(inst, str(path), cfg, str(checkpoint_dir), args.save_traceback))

    print(f"Saved summary: {summary_path}")
    print(f"Saved time trace: {trace_path}")


if __name__ == "__main__":
    main()
