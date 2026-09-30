from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gurobipy import GRB, Model, quicksum

from .classical_core.schema import ClassicalVRPInstance, ClassicalVRPSolution, solution_route_sequence

MAX_GUROBI_TIME_LIMIT_S = 7200.0


def capped_time_limit_s(time_limit_s: float | int | None) -> float:
    if time_limit_s is None:
        return MAX_GUROBI_TIME_LIMIT_S
    requested = float(time_limit_s)
    if requested < 0.0:
        raise ValueError(f"time_limit_s must be non-negative, got {requested}")
    return min(requested, MAX_GUROBI_TIME_LIMIT_S)


@dataclass(frozen=True)
class GurobiSolverConfig:
    time_limit_s: float = 7200.0
    mip_gap: float = 0.0
    output_flag: int = 0
    checkpoints_s: tuple[float, ...] = field(default_factory=tuple)
    threads: int | None = 1


class GurobiCVRPSolver:
    name = "gurobi_cvrp_arcflow"

    def __init__(self, config: GurobiSolverConfig | None = None):
        self.config = config or GurobiSolverConfig()
        self.model: Model | None = None
        self.x: dict[tuple[int, int], Any] = {}

    def solve(self, instance: ClassicalVRPInstance) -> ClassicalVRPSolution:
        start = time.perf_counter()
        model, x = self._build_model(instance)
        self.model = model
        self.x = x
        trace = self._new_trace()
        callback = self._make_callback(trace, instance, x)
        model.optimize(callback)
        runtime = time.perf_counter() - start

        status_name = self._status_name(model)
        status_code = int(model.Status)
        has_solution = model.SolCount > 0
        routes = self._extract_routes(x, instance.num_customers) if has_solution else []
        objective = self._route_distance_km(routes, instance) if has_solution else None
        vehicle_count = len(routes) if routes else None
        best_bound = float(model.ObjBound) if has_solution or model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else None
        mip_gap = float(model.MIPGap) if has_solution and np.isfinite(model.MIPGap) else None
        final = self._snapshot(
            checkpoint_s=None,
            elapsed_s=runtime,
            reached_checkpoint=True,
            status=status_name,
            routes=routes,
            objective=objective,
            vehicle_count=vehicle_count,
            best_bound=best_bound,
            mip_gap=mip_gap,
            source="final",
        )
        self._finalize_checkpoints(trace, final, runtime, status_name)
        model.dispose()
        return ClassicalVRPSolution(
            instance_id=instance.instance_id,
            solver_name=self.name,
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=vehicle_count,
            runtime_s=runtime,
            feasible=has_solution,
            metadata={
                "status": status_name,
                "status_code": status_code,
                "best_bound": best_bound,
                "mip_gap": mip_gap,
                "checkpoint_snapshots": trace["checkpoint_snapshots"],
            },
        )

    def _build_model(self, instance: ClassicalVRPInstance) -> tuple[Model, dict[tuple[int, int], Any]]:
        n = instance.num_customers
        nodes = range(n + 1)
        customers = range(1, n + 1)
        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        demand = np.zeros(n + 1, dtype=float)
        demand[1:] = np.asarray(instance.demands_cm3, dtype=float)
        capacity = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        if not np.isfinite(capacity) or capacity <= 0.0:
            capacity = max(float(demand.sum()), 1.0)

        model = Model(f"CVRP_{instance.instance_id}")
        model.Params.TimeLimit = capped_time_limit_s(self.config.time_limit_s)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.OutputFlag = int(self.config.output_flag)
        if self.config.threads is not None:
            model.Params.Threads = int(self.config.threads)

        arcs = [(i, j) for i in nodes for j in nodes if i != j]
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        load = model.addVars(customers, lb=0.0, ub=capacity, vtype=GRB.CONTINUOUS, name="load")
        model.setObjective(quicksum(float(distance[i, j]) * x[i, j] for i, j in arcs), GRB.MINIMIZE)

        for j in customers:
            model.addConstr(quicksum(x[i, j] for i in nodes if i != j) == 1, name=f"in_{j}")
            model.addConstr(quicksum(x[j, k] for k in nodes if k != j) == 1, name=f"out_{j}")
            model.addConstr(load[j] >= float(demand[j]), name=f"load_lb_{j}")

        model.addConstr(
            quicksum(x[0, j] for j in customers) == quicksum(x[j, 0] for j in customers),
            name="depot_balance",
        )
        model.addConstr(quicksum(x[0, j] for j in customers) >= 1, name="at_least_one_vehicle")

        for i in customers:
            for j in customers:
                if i == j:
                    continue
                model.addConstr(
                    load[j] >= load[i] + float(demand[j]) - capacity * (1 - x[i, j]),
                    name=f"capacity_mtz_{i}_{j}",
                )
        return model, x

    @staticmethod
    def _route_distance_km(routes: list[list[int]], instance: ClassicalVRPInstance) -> float:
        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        total = 0.0
        for route in routes:
            total += sum(float(distance[int(a), int(b)]) for a, b in zip(route[:-1], route[1:]))
        return total

    @staticmethod
    def _status_name(model: Model) -> str:
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        return status_map.get(model.Status, str(model.Status))

    def _extract_routes(self, x: dict[tuple[int, int], Any], n: int) -> list[list[int]]:
        outgoing: dict[int, list[int]] = {}
        for (i, j), var in x.items():
            if var.X > 0.5:
                outgoing.setdefault(int(i), []).append(int(j))
        routes: list[list[int]] = []
        for first in sorted(outgoing.get(0, [])):
            route = [0, first]
            current = first
            seen = {0}
            while current != 0:
                if current in seen:
                    break
                seen.add(current)
                nxts = sorted(outgoing.get(current, []))
                if not nxts:
                    break
                current = int(nxts[0])
                route.append(current)
            if len(route) >= 2 and route[-1] == 0:
                routes.append(route)
        return routes

    def _new_trace(self) -> dict[str, Any]:
        return {
            "checkpoints_s": tuple(sorted({float(t) for t in self.config.checkpoints_s if float(t) >= 0.0})),
            "next_checkpoint_index": 0,
            "last_snapshot": None,
            "checkpoint_snapshots": [],
        }

    def _make_callback(self, trace: dict[str, Any], instance: ClassicalVRPInstance, x: dict[tuple[int, int], Any]):
        start = time.perf_counter()

        def callback(model: Model, where: int) -> None:
            elapsed = time.perf_counter() - start
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(x)
                routes = self._routes_from_values(vals, instance.num_customers)
                obj = self._route_distance_km(routes, instance) if routes else None
                bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                incumbent = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                gap = abs(incumbent - bound) / max(abs(incumbent), 1e-10) if incumbent is not None else None
                trace["last_snapshot"] = self._snapshot(None, elapsed, True, "RUNNING", routes, obj, len(routes), bound, gap, "incumbent")
            self._record_due_checkpoints(trace, elapsed, "RUNNING")

        return callback

    @staticmethod
    def _routes_from_values(vals: dict[tuple[int, int], float], n: int) -> list[list[int]]:
        outgoing: dict[int, list[int]] = {}
        for (i, j), value in vals.items():
            if value > 0.5:
                outgoing.setdefault(int(i), []).append(int(j))
        routes: list[list[int]] = []
        for first in sorted(outgoing.get(0, [])):
            route = [0, first]
            current = first
            seen = {0}
            while current != 0:
                if current in seen:
                    break
                seen.add(current)
                nxts = sorted(outgoing.get(current, []))
                if not nxts:
                    break
                current = int(nxts[0])
                route.append(current)
            if route[-1] == 0:
                routes.append(route)
        return routes

    @staticmethod
    def _snapshot(
        checkpoint_s: float | None,
        elapsed_s: float,
        reached_checkpoint: bool,
        status: str,
        routes: list[list[int]],
        objective: float | None,
        vehicle_count: int | None,
        best_bound: float | None,
        mip_gap: float | None,
        source: str,
    ) -> dict[str, Any]:
        return {
            "checkpoint_s": checkpoint_s,
            "elapsed_s": elapsed_s,
            "reached_checkpoint": reached_checkpoint,
            "solver_status": status,
            "has_incumbent": objective is not None,
            "objective_distance_km": objective,
            "best_bound": best_bound,
            "mip_gap": mip_gap,
            "vehicle_count": vehicle_count,
            "routes": routes,
            "route_sequence": solution_route_sequence(routes),
            "source": source,
        }

    def _record_due_checkpoints(self, trace: dict[str, Any], elapsed_s: float, status: str) -> None:
        checkpoints = trace["checkpoints_s"]
        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            if elapsed_s < checkpoint_s:
                break
            snap = dict(trace["last_snapshot"] or self._snapshot(checkpoint_s, elapsed_s, True, status, [], None, None, None, None, "checkpoint_no_incumbent"))
            snap["checkpoint_s"] = checkpoint_s
            snap["elapsed_s"] = elapsed_s
            snap["reached_checkpoint"] = True
            snap["solver_status"] = status
            trace["checkpoint_snapshots"].append(snap)
            trace["next_checkpoint_index"] += 1

    def _finalize_checkpoints(self, trace: dict[str, Any], final_snapshot: dict[str, Any], runtime_s: float, status: str) -> None:
        checkpoints = trace["checkpoints_s"]
        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            snap = dict(final_snapshot)
            snap["checkpoint_s"] = checkpoint_s
            snap["elapsed_s"] = runtime_s
            snap["reached_checkpoint"] = runtime_s >= checkpoint_s
            snap["solver_status"] = status
            snap["source"] = "final"
            trace["checkpoint_snapshots"].append(snap)
            trace["next_checkpoint_index"] += 1
