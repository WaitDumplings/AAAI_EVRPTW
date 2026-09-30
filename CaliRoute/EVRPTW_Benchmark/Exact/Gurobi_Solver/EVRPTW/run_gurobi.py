from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "EVRPTW_Benchmark.Exact.Gurobi_Solver.EVRPTW"


from ..resume import completed_instance_ids
from .evrptw_core.io import iter_instances, save_solution
from .evrptw_core.schema import EVRPTWSolution, solution_route_sequence
from .evrptw_core.validation import validate_instance_structure
from .gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig, MAX_GUROBI_TIME_LIMIT_S, capped_time_limit_s


DEFAULT_EXACT_TIME_LIMIT_S = 7200.0

SUMMARY_FIELDNAMES = [
    "instance_id", "file", "status", "status_name", "feasible", "objective_distance_km",
    "vehicle_count", "runtime_s", "first_feasible_time_s", "mip_gap", "best_bound",
    "routes_json", "route_sequence_json", "solution_path", "time_trace_path",
    "tie_break_applied", "stage1_best_distance_km", "distance_tolerance", "errors", "traceback",
]

TIME_TRACE_FIELDNAMES = [
    "instance_id", "file", "checkpoint_s", "elapsed_s", "reached_checkpoint", "status",
    "has_incumbent", "first_feasible_time_s", "objective_distance_km", "best_bound", "mip_gap",
    "vehicle_count", "routes_json", "route_sequence_json", "checkpoint_solution_path", "source", "errors",
]

REFERENCE_FIELDNAMES = [
    "instance_key", "split", "scale", "instance_id", "region_id", "status", "objective",
    "is_certified_optimal", "lower_bound", "optimality_gap", "runtime_sec", "solver_name",
    "solver_version", "time_limit_sec", "solution_path", "notes",
]


def discover_instance_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]

    direct = dataset_path / "instances.pkl"
    if direct.exists():
        return [direct]

    search_root = dataset_path / "instances" if (dataset_path / "instances").exists() else dataset_path
    paths = set(search_root.glob("**/instances.pkl"))
    paths.update(search_root.glob("**/instance_*.pkl"))
    return sorted(paths)


def parse_checkpoints(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return tuple()
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"Checkpoint must be non-negative, got {value}")
        values.append(value)
    return tuple(sorted(set(values)))


def parse_scales(raw: str) -> set[str]:
    scales: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower().startswith("cus"):
            suffix = item[3:]
        else:
            suffix = item
        scales.add(f"Cus{int(suffix)}")
    return scales


def resolve_time_schedule(
    requested_checkpoints_s: tuple[float, ...],
    requested_time_limit_s: float | None,
) -> tuple[tuple[float, ...], float]:
    if not requested_checkpoints_s:
        requested = float(requested_time_limit_s) if requested_time_limit_s is not None else DEFAULT_EXACT_TIME_LIMIT_S
        time_limit_s = capped_time_limit_s(requested)
        return (time_limit_s,), time_limit_s

    if requested_time_limit_s is None:
        requested = max(requested_checkpoints_s)
    else:
        requested = max(float(requested_time_limit_s), max(requested_checkpoints_s))
    time_limit_s = capped_time_limit_s(requested)
    checkpoints_s = tuple(t for t in requested_checkpoints_s if t <= time_limit_s)
    if not checkpoints_s:
        checkpoints_s = (time_limit_s,)
    return checkpoints_s, time_limit_s


def checkpoint_label(checkpoint_s: float | int | None) -> str:
    if checkpoint_s is None:
        return "final"
    value = float(checkpoint_s)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}s".replace(".", "p")


def scale_for_instance(instance: Any) -> str:
    return f"Cus{int(instance.num_customers)}"


def instance_index(instance_id: str) -> int | None:
    try:
        return int(str(instance_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def infer_split(dataset_path: Path) -> str:
    for part in reversed(dataset_path.parts):
        if part in {"train", "val", "eval", "test"}:
            return part
    return "eval"


def instance_info(instance: Any, split: str) -> dict[str, str]:
    scale = scale_for_instance(instance)
    metadata = instance.metadata or {}
    instance_key = metadata.get("reference_solution_key") or metadata.get("instance_key")
    if not instance_key:
        instance_key = f"{split}/{scale}/{instance.instance_id}"
    return {
        "instance_key": str(instance_key),
        "split": split,
        "scale": scale,
        "instance_id": str(instance.instance_id),
        "region_id": str(instance.region_id),
    }


def gurobi_version_string() -> str:
    try:
        import gurobipy as gp
    except Exception:
        return ""
    return ".".join(str(part) for part in gp.gurobi.version())


def write_checkpoint_solution(
    instance_id: str,
    snapshot: dict[str, Any],
    solver_name: str,
    checkpoint_dir: Path,
) -> str:
    if not snapshot.get("has_incumbent"):
        return ""
    label = checkpoint_label(snapshot.get("checkpoint_s"))
    solution = EVRPTWSolution(
        instance_id=instance_id,
        solver_name=solver_name,
        routes=snapshot.get("routes", []),
        objective_distance_km=snapshot.get("objective_distance_km"),
        vehicle_count=snapshot.get("vehicle_count"),
        runtime_s=snapshot.get("elapsed_s"),
        feasible=True,
        metadata={
            "checkpoint_s": snapshot.get("checkpoint_s"),
            "reached_checkpoint": snapshot.get("reached_checkpoint"),
            "best_bound": snapshot.get("best_bound"),
            "mip_gap": snapshot.get("mip_gap"),
            "solver_status": snapshot.get("solver_status"),
            "source": snapshot.get("source"),
        },
    )
    path = checkpoint_dir / f"{instance_id}_{label}_solution.pkl"
    save_solution(path, solution)
    return str(path)


def append_time_rows(
    rows: list[dict[str, Any]],
    instance_file: Path,
    instance_id: str,
    solution: EVRPTWSolution,
    checkpoint_dir: Path,
) -> None:
    first_feasible_time_s = solution.metadata.get("first_feasible_time_s")
    for snapshot in solution.metadata.get("checkpoint_snapshots", []):
        checkpoint_solution_path = write_checkpoint_solution(
            instance_id=instance_id,
            snapshot=snapshot,
            solver_name=solution.solver_name,
            checkpoint_dir=checkpoint_dir,
        )
        rows.append({
            "instance_id": instance_id,
            "file": str(instance_file),
            "checkpoint_s": snapshot.get("checkpoint_s"),
            "elapsed_s": snapshot.get("elapsed_s"),
            "reached_checkpoint": snapshot.get("reached_checkpoint"),
            "status": snapshot.get("solver_status"),
            "has_incumbent": snapshot.get("has_incumbent"),
            "first_feasible_time_s": first_feasible_time_s,
            "objective_distance_km": snapshot.get("objective_distance_km"),
            "best_bound": snapshot.get("best_bound"),
            "mip_gap": snapshot.get("mip_gap"),
            "vehicle_count": snapshot.get("vehicle_count"),
            "routes_json": json.dumps(snapshot.get("routes", [])),
            "route_sequence_json": json.dumps(snapshot.get("route_sequence", [])),
            "checkpoint_solution_path": checkpoint_solution_path,
            "source": snapshot.get("source"),
            "errors": "",
        })


def append_error_time_rows(
    rows: list[dict[str, Any]],
    checkpoints_s: tuple[float, ...],
    instance_file: Path,
    instance_id: str,
    status: str,
    error: str,
) -> None:
    for checkpoint_s in checkpoints_s:
        rows.append({
            "instance_id": instance_id,
            "file": str(instance_file),
            "checkpoint_s": checkpoint_s,
            "elapsed_s": "",
            "reached_checkpoint": False,
            "status": status,
            "has_incumbent": False,
            "first_feasible_time_s": "",
            "objective_distance_km": "",
            "best_bound": "",
            "mip_gap": "",
            "vehicle_count": "",
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "checkpoint_solution_path": "",
            "source": "error",
            "errors": error,
        })


def invalid_summary_row(instance: Any, instance_file: Path, errors: str) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "file": str(instance_file),
        "status": "INVALID_INSTANCE",
        "status_name": "INVALID_INSTANCE",
        "feasible": False,
        "objective_distance_km": "",
        "vehicle_count": "",
        "runtime_s": "",
        "first_feasible_time_s": "",
        "mip_gap": "",
        "best_bound": "",
        "routes_json": "",
        "route_sequence_json": "",
        "solution_path": "",
        "time_trace_path": "",
        "tie_break_applied": "",
        "stage1_best_distance_km": "",
        "distance_tolerance": "",
        "errors": errors,
        "traceback": "",
    }


def error_summary_row(
    instance_id: str,
    instance_file: Path,
    error: str,
    traceback_text: str,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "file": str(instance_file),
        "status": "ERROR",
        "status_name": "ERROR",
        "feasible": False,
        "objective_distance_km": "",
        "vehicle_count": "",
        "runtime_s": "",
        "first_feasible_time_s": "",
        "mip_gap": "",
        "best_bound": "",
        "routes_json": "",
        "route_sequence_json": "",
        "solution_path": "",
        "time_trace_path": "",
        "tie_break_applied": "",
        "stage1_best_distance_km": "",
        "distance_tolerance": "",
        "errors": error,
        "traceback": traceback_text,
    }


def solved_summary_row(instance: Any, instance_file: Path, solution: EVRPTWSolution) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "file": str(instance_file),
        "status": solution.metadata.get("gurobi_status"),
        "status_name": solution.metadata.get("gurobi_status_name"),
        "feasible": solution.feasible,
        "objective_distance_km": solution.objective_distance_km,
        "vehicle_count": solution.vehicle_count,
        "runtime_s": solution.runtime_s,
        "first_feasible_time_s": solution.metadata.get("first_feasible_time_s"),
        "mip_gap": solution.metadata.get("mip_gap"),
        "best_bound": solution.metadata.get("best_bound"),
        "routes_json": json.dumps(solution.routes),
        "route_sequence_json": json.dumps(solution_route_sequence(solution)),
        "solution_path": "",
        "time_trace_path": "",
        "tie_break_applied": solution.metadata.get("tie_break_applied"),
        "stage1_best_distance_km": solution.metadata.get("stage1_best_distance_km"),
        "distance_tolerance": solution.metadata.get("distance_tolerance"),
        "errors": json.dumps(solution.violations),
        "traceback": "",
    }


def solve_instance_task(
    instance: Any,
    solver_config: GurobiSolverConfig,
    instance_file_raw: str,
    checkpoints_s: tuple[float, ...],
    save_traceback: bool,
    warm_start_routes: list[list[int]] | None = None,
) -> dict[str, Any]:
    instance_file = Path(instance_file_raw)
    instance_id = instance.instance_id
    try:
        validation = validate_instance_structure(instance)
        if not validation.success:
            errors = json.dumps(validation.errors)
            time_rows: list[dict[str, Any]] = []
            append_error_time_rows(time_rows, checkpoints_s, instance_file, instance_id, "INVALID_INSTANCE", errors)
            return {
                "instance_id": instance_id,
                "summary_row": invalid_summary_row(instance, instance_file, errors),
                "solution": None,
                "time_rows": time_rows,
            }

        solver = GurobiEVRPTWSolver(solver_config)
        solution = solver.solve(instance, warm_start_routes=warm_start_routes)
        return {
            "instance_id": instance_id,
            "summary_row": solved_summary_row(instance, instance_file, solution),
            "solution": solution.to_dict(),
            "time_rows": [],
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        time_rows = []
        append_error_time_rows(time_rows, checkpoints_s, instance_file, instance_id, "ERROR", error)
        return {
            "instance_id": instance_id,
            "summary_row": error_summary_row(
                instance_id,
                instance_file,
                error,
                traceback.format_exc() if save_traceback else "",
            ),
            "solution": None,
            "time_rows": time_rows,
        }


def reference_status(summary_row: dict[str, Any]) -> str:
    status = summary_row.get("status_name") or summary_row.get("status") or ""
    return str(status).lower()


def write_reference_route(
    reference_root: Path,
    info: dict[str, str],
    solution: EVRPTWSolution,
    solver_version: str,
    time_limit_s: float,
) -> Path:
    path = reference_root / info["split"] / "routes" / info["scale"] / f"{info['instance_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    detail = {
        "instance_key": info["instance_key"],
        "split": info["split"],
        "scale": info["scale"],
        "instance_id": info["instance_id"],
        "region_id": info["region_id"],
        "objective": solution.objective_distance_km,
        "objective_unit": "km",
        "status": str(solution.metadata.get("gurobi_status_name") or "").lower(),
        "is_certified_optimal": solution.metadata.get("gurobi_status_name") == "OPTIMAL",
        "lower_bound": solution.metadata.get("best_bound"),
        "optimality_gap": solution.metadata.get("mip_gap"),
        "runtime_sec": solution.runtime_s,
        "solver_name": solution.solver_name,
        "solver_version": solver_version,
        "time_limit_sec": time_limit_s,
        "vehicle_count": solution.vehicle_count,
        "routes": [
            {
                "vehicle_id": vehicle_id,
                "terminal_sequence": [int(node) for node in route],
            }
            for vehicle_id, route in enumerate(solution.routes)
        ],
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)
    return path


def make_reference_row(
    info: dict[str, str],
    summary_row: dict[str, Any],
    solution: EVRPTWSolution | None,
    reference_root: Path,
    route_path: Path | None,
    solver_version: str,
    time_limit_s: float,
) -> dict[str, Any]:
    if route_path is not None:
        solution_path = str(route_path.relative_to(reference_root / info["split"]))
    else:
        solution_path = ""

    status_name = str(summary_row.get("status_name") or "")
    is_optimal = status_name == "OPTIMAL"
    solver_name = solution.solver_name if solution is not None else GurobiEVRPTWSolver.name
    notes = ""
    if summary_row.get("vehicle_count") not in ("", None):
        notes = f"vehicle_count={summary_row.get('vehicle_count')}"
    errors = str(summary_row.get("errors") or "")
    if errors and errors != "{}":
        notes = errors

    return {
        "instance_key": info["instance_key"],
        "split": info["split"],
        "scale": info["scale"],
        "instance_id": info["instance_id"],
        "region_id": info["region_id"],
        "status": reference_status(summary_row),
        "objective": summary_row.get("objective_distance_km", ""),
        "is_certified_optimal": str(is_optimal).lower(),
        "lower_bound": summary_row.get("best_bound", ""),
        "optimality_gap": summary_row.get("mip_gap", ""),
        "runtime_sec": summary_row.get("runtime_s", ""),
        "solver_name": solver_name,
        "solver_version": solver_version,
        "time_limit_sec": time_limit_s,
        "solution_path": solution_path,
        "notes": notes,
    }


def write_reference_csvs(reference_root: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)

    for split, split_rows in sorted(grouped.items()):
        path = reference_root / split / "solutions.csv"
        write_csv_atomic(
            path,
            split_rows,
            REFERENCE_FIELDNAMES,
            sort_key=lambda row: str(row.get("instance_id", "")),
        )


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_status_name(row: dict[str, Any]) -> str:
    return str(row.get("status_name") or row.get("status") or "").strip().upper()


def load_expert_routes(summary_row: dict[str, Any]) -> list[list[int]]:
    routes_json = str(summary_row.get("routes_json") or "").strip()
    if routes_json:
        try:
            routes = json.loads(routes_json)
            if isinstance(routes, list) and routes:
                return [[int(node) for node in route] for route in routes if isinstance(route, list)]
        except Exception:
            return []
    return []


def row_has_objective(summary_row: dict[str, Any]) -> bool:
    return str(summary_row.get("objective_distance_km") or "").strip() != ""


def build_expert_index(expert_summary_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(expert_summary_path)
    return {
        str(row.get("instance_id", "")): row
        for row in rows
        if str(row.get("instance_id", "")).strip()
    }


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    sort_key: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def upsert_instance_rows(
    rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    instance_id: str,
) -> list[dict[str, Any]]:
    kept = [row for row in rows if str(row.get("instance_id", "")) != instance_id]
    kept.extend(new_rows)
    return kept


def load_reference_rows(reference_root: Path | None, split: str) -> list[dict[str, Any]]:
    if reference_root is None:
        return []
    return read_csv_rows(reference_root / split / "solutions.csv")


def checkpoint_sort_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("checkpoint_s"))
    except (TypeError, ValueError):
        return float("inf")


def preflight_gurobi_license() -> None:
    try:
        import gurobipy as gp

        model = gp.Model("gurobi_license_preflight")
        model.Params.OutputFlag = 0
        x = model.addVar(lb=0.0, ub=1.0, name="x")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.optimize()
        model.dispose()
    except Exception as exc:
        license_file = os.environ.get("GRB_LICENSE_FILE", "")
        suffix = f" GRB_LICENSE_FILE={license_file!r}." if license_file else ""
        raise RuntimeError(
            f"Gurobi license preflight failed before starting the batch.{suffix} "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the exact Gurobi EVRP-TW-D solver on pickle instances.")
    parser.add_argument("--dataset_path", required=True, help="Dataset root or one instance pickle file.")
    parser.add_argument("--save_path", required=True, help="Directory for benchmark summaries and solution pickles.")
    parser.add_argument("--time_limit_s", type=float, default=None, help="Max solve time in seconds for each Gurobi optimize call. Hard-capped at 7200 seconds.")
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--cs_copies", type=int, default=3, help="Number of dummy copies per active charging station. Default: 3.")
    parser.add_argument("--output_flag", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi thread limit per model. Defaults to 1 when --workers > 1.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes. Default: 1.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total instance limit after scale filtering.")
    parser.add_argument("--start_index", type=int, default=None, help="Optional inclusive lower bound for the numeric instance id suffix.")
    parser.add_argument("--end_index", type=int, default=None, help="Optional exclusive upper bound for the numeric instance id suffix.")
    parser.add_argument("--scales", default="", help="Optional comma-separated scale filter, e.g. Cus5,Cus15.")
    parser.add_argument("--skip_completed", action="store_true", help="Skip finished summary results, including TIME_LIMIT; retry errors and interrupted runs.")
    parser.add_argument("--expert_summary_path", default="", help="Optional existing gurobi_summary.csv used as warm-start experts for refine runs.")
    parser.add_argument("--reference_save_path", default="", help="Optional reference_solutions root for split/Cus*/solutions.csv and routes/*.json.")
    parser.add_argument("--reference_split", default="", help="Reference split name. Defaults to train/val/eval/test inferred from dataset_path.")
    parser.add_argument("--checkpoints_s", default="", help="Comma-separated seconds for incumbent snapshots, e.g. 60,300,900.")
    parser.add_argument("--tie_break_vehicle_count", action=argparse.BooleanOptionalAction, default=True, help="Within optimal distance tolerance, minimize vehicle count. Default: true.")
    parser.add_argument("--distance_tolerance_abs", type=float, default=1e-6)
    parser.add_argument("--distance_tolerance_rel", type=float, default=1e-8)
    parser.add_argument("--save_traceback", action="store_true", help="Store Python tracebacks in the summary CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print per-instance progress.")
    args = parser.parse_args(argv)

    requested_checkpoints_s = parse_checkpoints(args.checkpoints_s)
    checkpoints_s, time_limit_s = resolve_time_schedule(requested_checkpoints_s, args.time_limit_s)
    workers = max(1, int(args.workers))
    threads = args.threads if args.threads is not None else (1 if workers > 1 else None)
    scale_filter = parse_scales(args.scales)
    limit = int(args.limit) if args.limit is not None else None
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")
    if args.start_index is not None and args.start_index < 0:
        raise ValueError(f"--start_index must be non-negative, got {args.start_index}")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError(f"--end_index must be non-negative, got {args.end_index}")
    if args.start_index is not None and args.end_index is not None and args.start_index >= args.end_index:
        raise ValueError(f"--start_index must be less than --end_index, got {args.start_index} >= {args.end_index}")

    print(
        f"Exact benchmark schedule: time_limit_s={time_limit_s:g}, max_optimize_call_s={MAX_GUROBI_TIME_LIMIT_S:g}, "
        f"checkpoints_s={list(checkpoints_s)}, cs_copies={args.cs_copies}, "
        f"workers={workers}, threads_per_worker={threads or 'gurobi-default'}"
    )

    expert_summary_path = Path(args.expert_summary_path) if args.expert_summary_path else None
    expert_index: dict[str, dict[str, Any]] | None = None
    if expert_summary_path is not None:
        if not expert_summary_path.exists():
            raise FileNotFoundError(f"--expert_summary_path does not exist: {expert_summary_path}")
        expert_index = build_expert_index(expert_summary_path)
        print(f"Expert summary: {expert_summary_path} rows={len(expert_index)}")

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    trace_path = save_path / "gurobi_time_trace.csv"
    summary_path = save_path / "gurobi_summary.csv"
    solutions_dir = save_path / "solutions"
    checkpoint_dir = solutions_dir / "checkpoints"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reference_root = Path(args.reference_save_path) if args.reference_save_path else None
    reference_split = args.reference_split.strip() or infer_split(dataset_path)
    solver_version = gurobi_version_string()

    solver_config = GurobiSolverConfig(
        time_limit_s=time_limit_s,
        mip_gap=args.mip_gap,
        cs_copies=args.cs_copies,
        output_flag=args.output_flag,
        checkpoints_s=checkpoints_s,
        tie_break_vehicle_count=args.tie_break_vehicle_count,
        distance_tolerance_abs=args.distance_tolerance_abs,
        distance_tolerance_rel=args.distance_tolerance_rel,
        threads=threads,
    )

    instance_files = discover_instance_files(dataset_path)
    existing_summary_rows = read_csv_rows(summary_path)
    completed_ids = completed_instance_ids(existing_summary_rows)
    records: list[tuple[Path, Any, list[list[int]] | None]] = []
    skipped_completed_count = 0
    skipped_range_count = 0
    skipped_refine_no_expert = 0
    skipped_refine_optimal = 0
    skipped_refine_no_incumbent = 0
    skipped_refine_status = 0
    for instance_file in instance_files:
        for instance in iter_instances(instance_file):
            if limit is not None and len(records) >= limit:
                break
            if scale_filter and scale_for_instance(instance) not in scale_filter:
                continue
            idx = instance_index(instance.instance_id)
            if args.start_index is not None and (idx is None or idx < args.start_index):
                skipped_range_count += 1
                continue
            if args.end_index is not None and (idx is None or idx >= args.end_index):
                skipped_range_count += 1
                continue
            if args.skip_completed and instance.instance_id in completed_ids:
                skipped_completed_count += 1
                continue

            warm_start_routes = None
            if expert_index is not None and expert_summary_path is not None:
                expert_row = expert_index.get(str(instance.instance_id))
                if expert_row is None:
                    skipped_refine_no_expert += 1
                    continue

                expert_status = normalize_status_name(expert_row)
                if expert_status == "OPTIMAL":
                    skipped_refine_optimal += 1
                    continue
                if expert_status != "TIME_LIMIT":
                    skipped_refine_status += 1
                    continue

                warm_start_routes = load_expert_routes(expert_row)
                if not warm_start_routes or not row_has_objective(expert_row):
                    skipped_refine_no_incumbent += 1
                    continue

            records.append((instance_file, instance, warm_start_routes))
        if limit is not None and len(records) >= limit:
            break

    instance_infos = {
        instance.instance_id: instance_info(instance, reference_split)
        for _, instance, _ in records
    }
    range_label = "all"
    if args.start_index is not None or args.end_index is not None:
        start_label = "" if args.start_index is None else str(args.start_index)
        end_label = "" if args.end_index is None else str(args.end_index)
        range_label = f"[{start_label}, {end_label})"
    print(
        f"Loaded {len(records)} instances from {len(instance_files)} bundle/file(s). "
        f"index_range={range_label} skipped_range={skipped_range_count} "
        f"skipped_completed={skipped_completed_count}"
    )
    if expert_index is not None:
        print(
            "Refine filter: "
            f"candidates={len(records)} "
            f"skipped_no_expert={skipped_refine_no_expert} "
            f"skipped_optimal={skipped_refine_optimal} "
            f"skipped_time_limit_no_incumbent={skipped_refine_no_incumbent} "
            f"skipped_other_status={skipped_refine_status}"
        )

    print(
        f"Resume: {'enabled' if args.skip_completed else 'disabled'}; "
        f"skipped={skipped_completed_count} pending={len(records)}; summary={summary_path}"
    )
    if not records:
        print("No pending instances; nothing to solve.")
        return
    preflight_gurobi_license()

    summary_rows: list[dict[str, Any]] = existing_summary_rows
    time_rows: list[dict[str, Any]] = read_csv_rows(trace_path)
    reference_rows: list[dict[str, Any]] = load_reference_rows(reference_root, reference_split)

    def consume_result(result: dict[str, Any]) -> None:
        nonlocal summary_rows, time_rows, reference_rows
        instance_id = str(result["instance_id"])
        summary_row = dict(result["summary_row"])
        summary_row["time_trace_path"] = str(trace_path)
        solution_dict = result.get("solution")
        solution = EVRPTWSolution.from_dict(solution_dict) if solution_dict is not None else None
        new_time_rows: list[dict[str, Any]] = []

        if solution is not None:
            solution_path = solutions_dir / f"{instance_id}_solution.pkl"
            save_solution(solution_path, solution)
            summary_row["solution_path"] = str(solution_path)
            append_time_rows(new_time_rows, Path(summary_row.get("file") or dataset_path), instance_id, solution, checkpoint_dir)
        else:
            new_time_rows.extend(result.get("time_rows", []))

        summary_rows = upsert_instance_rows(summary_rows, [summary_row], instance_id)
        time_rows = upsert_instance_rows(time_rows, new_time_rows, instance_id)

        if reference_root is not None:
            info = instance_infos[instance_id]
            route_path = None
            if solution is not None and solution.routes:
                route_path = write_reference_route(reference_root, info, solution, solver_version, time_limit_s)
            reference_row = make_reference_row(info, summary_row, solution, reference_root, route_path, solver_version, time_limit_s)
            reference_rows = upsert_instance_rows(reference_rows, [reference_row], instance_id)

        write_csv_atomic(
            summary_path,
            summary_rows,
            SUMMARY_FIELDNAMES,
            sort_key=lambda row: str(row.get("instance_id", "")),
        )
        write_csv_atomic(
            trace_path,
            time_rows,
            TIME_TRACE_FIELDNAMES,
            sort_key=lambda row: (str(row.get("instance_id", "")), checkpoint_sort_value(row)),
        )
        if reference_root is not None:
            write_reference_csvs(reference_root, reference_rows)

    if records and workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(records))) as executor:
            futures = {
                executor.submit(
                    solve_instance_task,
                    instance,
                    solver_config,
                    str(instance_file),
                    checkpoints_s,
                    args.save_traceback,
                    warm_start_routes,
                ): instance.instance_id
                for instance_file, instance, warm_start_routes in records
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                consume_result(result)
                if args.verbose:
                    row = result["summary_row"]
                    print(f"[{done_count}/{len(records)}] {result['instance_id']}: {row.get('status_name')} obj={row.get('objective_distance_km')}")
    else:
        for done_count, (instance_file, instance, warm_start_routes) in enumerate(records, start=1):
            result = solve_instance_task(
                instance,
                solver_config,
                str(instance_file),
                checkpoints_s,
                args.save_traceback,
                warm_start_routes,
            )
            consume_result(result)
            if args.verbose:
                row = result["summary_row"]
                print(f"[{done_count}/{len(records)}] {result['instance_id']}: {row.get('status_name')} obj={row.get('objective_distance_km')}")

    if reference_root is not None:
        print(f"Saved reference solutions under: {reference_root / reference_split}")

    print(f"Saved summary: {summary_path}")
    print(f"Saved time trace: {trace_path}")


if __name__ == "__main__":
    main()
