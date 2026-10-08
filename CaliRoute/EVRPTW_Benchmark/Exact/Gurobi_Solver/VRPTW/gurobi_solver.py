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


class GurobiVRPTWSolver:
    name = "gurobi_vrptw_arcflow"

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
        start_depot = 0
        end_depot = n + 1
        customers = range(1, n + 1)
        nodes = range(n + 2)

        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        travel_time = np.asarray(instance.travel_time_matrix_s, dtype=float)
        if travel_time.shape != distance.shape:
            raise ValueError("VRPTW benchmark requires travel_time_matrix_s matching distance_matrix_km; refusing an implicit constant-speed fallback")

        demand = np.zeros(n + 1, dtype=float)
        demand[1:] = np.asarray(instance.demands_cm3, dtype=float)
        capacity = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        if not np.isfinite(capacity) or capacity <= 0.0:
            capacity = max(float(demand.sum()), 1.0)

        service_time = np.zeros(n + 1, dtype=float)
        if instance.service_time_s is not None:
            service_time[1:] = np.asarray(instance.service_time_s, dtype=float)

        if instance.tw_s is not None:
            tw = np.asarray(instance.tw_s, dtype=float)
            ready = np.zeros(n + 1, dtype=float)
            due = np.zeros(n + 1, dtype=float)
            ready[1:] = tw[:, 0]
            due[1:] = tw[:, 1]
        else:
            ready = np.full(n + 1, float(instance.working_start_s or 0.0), dtype=float)
            due = np.full(n + 1, float(instance.working_end_s or 24 * 3600), dtype=float)
        working_start = float(instance.working_start_s if instance.working_start_s is not None else min(ready[1:], default=0.0))
        working_end = float(instance.working_end_s if instance.working_end_s is not None else max(due[1:], default=24 * 3600.0))

        def terminal(node: int) -> int:
            return 0 if node in (start_depot, end_depot) else node

        arcs: list[tuple[int, int]] = []
        arcs.extend((start_depot, j) for j in customers)
        arcs.extend((i, j) for i in customers for j in customers if i != j)
        arcs.extend((i, end_depot) for i in customers)

        max_travel = float(np.nanmax(travel_time)) if travel_time.size else 0.0
        max_service = float(np.nanmax(service_time)) if service_time.size else 0.0
        big_m_time = max(1.0, working_end - working_start + max_travel + max_service + 1.0)

        model = Model(f"VRPTW_{instance.instance_id}")
        model.Params.TimeLimit = capped_time_limit_s(self.config.time_limit_s)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.OutputFlag = int(self.config.output_flag)
        if self.config.threads is not None:
            model.Params.Threads = int(self.config.threads)

        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        load = model.addVars(customers, lb=0.0, ub=capacity, vtype=GRB.CONTINUOUS, name="load")
        arrival = model.addVars(nodes, lb=working_start, ub=working_end, vtype=GRB.CONTINUOUS, name="arrival")
        model.setObjective(
            quicksum(float(distance[terminal(i), terminal(j)]) * x[i, j] for i, j in arcs),
            GRB.MINIMIZE,
        )

        model.addConstr(arrival[start_depot] == working_start, name="start_time")
        model.addConstr(arrival[end_depot] <= working_end, name="end_time")

        for j in customers:
            model.addConstr(quicksum(x[i, j] for i in [start_depot, *customers] if i != j) == 1, name=f"in_{j}")
            model.addConstr(quicksum(x[j, k] for k in [*customers, end_depot] if k != j) == 1, name=f"out_{j}")
            model.addConstr(arrival[j] >= float(ready[j]), name=f"ready_{j}")
            model.addConstr(arrival[j] <= float(due[j]), name=f"due_{j}")
            model.addConstr(load[j] >= float(demand[j]), name=f"load_lb_{j}")

        model.addConstr(
            quicksum(x[start_depot, j] for j in customers) == quicksum(x[j, end_depot] for j in customers),
            name="depot_balance",
        )
        model.addConstr(quicksum(x[start_depot, j] for j in customers) >= 1, name="at_least_one_vehicle")

        for i, j in arcs:
            ti = terminal(i)
            tj = terminal(j)
            depart_service = 0.0 if i in (start_depot, end_depot) else float(service_time[ti])
            model.addConstr(
                arrival[j] >= arrival[i] + depart_service + float(travel_time[ti, tj]) - big_m_time * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )

        for j in customers:
            model.addConstr(load[j] >= float(demand[j]) * x[start_depot, j], name=f"start_load_{j}")
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

    def _extract_routes(self, x: dict[tuple[int, int], Any], num_customers: int) -> list[list[int]]:
        end_depot = num_customers + 1
        successor: dict[int, int] = {}
        for (i, j), var in x.items():
            if var.X > 0.5:
                successor[int(i)] = int(j)
        routes: list[list[int]] = []
        starts = sorted(j for (i, j), var in x.items() if i == 0 and var.X > 0.5)
        for first in starts:
            route = [0]
            node = int(first)
            seen: set[int] = set()
            while node != end_depot and node not in seen:
                seen.add(node)
                route.append(node)
                node = successor.get(node, end_depot)
            route.append(0)
            routes.append(route)
        return routes

    def _make_callback(self, trace: dict[str, Any], instance: ClassicalVRPInstance, x: dict[tuple[int, int], Any]):
        checkpoints = tuple(float(v) for v in self.config.checkpoints_s)
        if not checkpoints:
            return None
        trace["pending_checkpoints"] = list(checkpoints)
        start_time = time.perf_counter()

        def callback(model: Model, where: int) -> None:
            if where not in (GRB.Callback.MIPSOL, GRB.Callback.MIP):
                return
            elapsed = time.perf_counter() - start_time
            pending = trace.get("pending_checkpoints", [])
            if not pending or elapsed < pending[0]:
                return
            checkpoint = pending.pop(0)
            routes: list[list[int]] = []
            objective = None
            vehicle_count = None
            try:
                if model.SolCount > 0:
                    routes = self._extract_callback_routes(model, x, instance.num_customers)
                    objective = self._route_distance_km(routes, instance) if routes else None
                    vehicle_count = len(routes) if routes else None
            except Exception:
                routes = []
            best_bound = None
            mip_gap = None
            try:
                best_bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
                best_obj = float(model.cbGet(GRB.Callback.MIP_OBJBST))
                if np.isfinite(best_obj) and abs(best_obj) > 1e-12:
                    mip_gap = abs(best_obj - best_bound) / abs(best_obj)
            except Exception:
                pass
            trace["checkpoint_snapshots"].append(self._snapshot(
                checkpoint_s=checkpoint,
                elapsed_s=elapsed,
                reached_checkpoint=True,
                status="RUNNING",
                routes=routes,
                objective=objective,
                vehicle_count=vehicle_count,
                best_bound=best_bound,
                mip_gap=mip_gap,
                source="checkpoint_incumbent" if routes else "checkpoint_no_incumbent",
            ))
        return callback

    def _extract_callback_routes(self, model: Model, x: dict[tuple[int, int], Any], num_customers: int) -> list[list[int]]:
        end_depot = num_customers + 1
        selected = {(i, j): model.cbGetSolution(var) for (i, j), var in x.items()}
        successor = {int(i): int(j) for (i, j), value in selected.items() if value > 0.5}
        routes: list[list[int]] = []
        starts = sorted(j for (i, j), value in selected.items() if i == 0 and value > 0.5)
        for first in starts:
            route = [0]
            node = int(first)
            seen: set[int] = set()
            while node != end_depot and node not in seen:
                seen.add(node)
                route.append(node)
                node = successor.get(node, end_depot)
            route.append(0)
            routes.append(route)
        return routes

    def _new_trace(self) -> dict[str, Any]:
        return {"checkpoint_snapshots": [], "pending_checkpoints": list(self.config.checkpoints_s)}

    def _snapshot(
        self,
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
            "has_incumbent": bool(routes),
            "routes": routes,
            "route_sequence": solution_route_sequence(routes),
            "objective_distance_km": objective,
            "vehicle_count": vehicle_count,
            "best_bound": best_bound,
            "mip_gap": mip_gap,
            "source": source,
        }

    def _finalize_checkpoints(self, trace: dict[str, Any], final: dict[str, Any], runtime: float, status_name: str) -> None:
        recorded = {snap.get("checkpoint_s") for snap in trace["checkpoint_snapshots"]}
        for checkpoint in self.config.checkpoints_s:
            if checkpoint in recorded:
                continue
            clone = dict(final)
            clone["checkpoint_s"] = checkpoint
            clone["elapsed_s"] = runtime
            clone["reached_checkpoint"] = runtime >= checkpoint
            clone["solver_status"] = status_name
            clone["source"] = "final_before_checkpoint" if runtime < checkpoint else "final_fill"
            trace["checkpoint_snapshots"].append(clone)
        trace["checkpoint_snapshots"].append(final)

    @staticmethod
    def _status_name(model: Model) -> str:
        status_map = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INPROGRESS: "INPROGRESS",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
            GRB.WORK_LIMIT: "WORK_LIMIT",
            GRB.MEM_LIMIT: "MEM_LIMIT",
        }
        return status_map.get(int(model.Status), str(model.Status))
