from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gurobipy import GRB, Model, quicksum

from .evrptw_core.schema import EVRPTWInstance, EVRPTWSolution, merge_route_sequences


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
    cs_copies: int = 3
    output_flag: int = 0
    checkpoints_s: tuple[float, ...] = field(default_factory=tuple)
    record_incumbent_events: bool = True
    tie_break_vehicle_count: bool = True
    distance_tolerance_abs: float = 1e-6
    distance_tolerance_rel: float = 1e-8
    threads: int | None = None


@dataclass(frozen=True)
class NodeMap:
    solver_to_terminal: list[int]
    customer_nodes: list[int]
    cs_nodes: list[int]
    start_depot: int
    end_depot: int


class GurobiEVRPTWSolver:
    """Small-scale exact EVRP-TW-D solver using the canonical pickle schema.

    The model is an arc-flow MILP with duplicated charging-station nodes. Depot
    and charging stations reset the battery to full before departure. Departing
    from a charging station pays the instance's full-charge time. This matches
    the generator's full-charge EV transition semantics.

    Benchmark logging is callback-based: every checkpoint stores the incumbent
    route sequence available at or before that runtime, plus the current best
    bound/gap when Gurobi exposes them.
    """

    name = "gurobi_exact_arcflow"

    def __init__(self, config: GurobiSolverConfig | None = None):
        self.config = config or GurobiSolverConfig()
        self.model: Model | None = None
        self.node_map: NodeMap | None = None
        self.x: dict[tuple[int, int], Any] = {}

    def solve(self, instance: EVRPTWInstance, warm_start_routes: list[list[int]] | None = None) -> EVRPTWSolution:
        start = time.perf_counter()
        model, node_map, x, distance_expr, vehicle_expr = self._build_model(instance)
        self.model = model
        self.node_map = node_map
        self.x = x
        warm_start_info = self._apply_warm_start(instance, node_map, x, warm_start_routes)

        trace = self._new_trace()
        callback = self._make_callback(trace, node_map, x)
        model.optimize(callback)

        stage1_status = int(model.Status)
        stage1_status_name = self._status_name(stage1_status)
        stage1_has_solution = model.SolCount > 0
        stage1_best_distance = self._safe_model_float(model, "ObjVal") if stage1_has_solution else None
        stage1_best_bound = self._safe_model_float(model, "ObjBound")
        stage1_mip_gap = self._safe_model_float(model, "MIPGap") if stage1_has_solution else None
        tie_break_applied = False
        distance_tolerance = None
        tie_break_status = None
        tie_break_status_name = None

        if (
            self.config.tie_break_vehicle_count
            and stage1_has_solution
            and stage1_status == GRB.OPTIMAL
            and stage1_best_distance is not None
        ):
            tie_break_applied = True
            distance_tolerance = max(
                float(self.config.distance_tolerance_abs),
                float(self.config.distance_tolerance_rel) * abs(float(stage1_best_distance)),
            )
            model.addConstr(distance_expr <= float(stage1_best_distance) + distance_tolerance, name="distance_optimal_tolerance")
            model.setObjective(vehicle_expr, GRB.MINIMIZE)
            model.Params.TimeLimit = capped_time_limit_s(self.config.time_limit_s)
            model.optimize()
            tie_break_status = int(model.Status)
            tie_break_status_name = self._status_name(tie_break_status)

        runtime = time.perf_counter() - start

        status = int(model.Status)
        status_name = self._status_name(status)
        has_solution = model.SolCount > 0
        routes = self._extract_routes(node_map, x) if has_solution else []
        objective = self._route_distance_km(routes, instance) if has_solution else None
        best_bound = stage1_best_bound
        mip_gap = stage1_mip_gap
        feasible = status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED} and has_solution
        violations: dict[str, Any] = {}
        if not feasible:
            violations["gurobi_status"] = status

        if has_solution and trace["first_feasible_time_s"] is None:
            trace["first_feasible_time_s"] = runtime

        final_snapshot = self._make_snapshot(
            checkpoint_s=None,
            elapsed_s=runtime,
            reached_checkpoint=True,
            solver_status=status_name,
            objective_distance_km=objective,
            best_bound=objective if status == GRB.OPTIMAL and objective is not None else best_bound,
            routes=routes,
            source="final",
        )
        if status == GRB.OPTIMAL and objective is not None:
            final_snapshot["mip_gap"] = 0.0
        elif mip_gap is not None:
            final_snapshot["mip_gap"] = mip_gap
        self._finalize_checkpoints(trace, final_snapshot, runtime, status_name)

        return EVRPTWSolution(
            instance_id=instance.instance_id,
            solver_name=self.name,
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=len(routes) if routes else None,
            runtime_s=runtime,
            feasible=bool(feasible),
            violations=violations,
            metadata={
                "gurobi_status": status,
                "gurobi_status_name": status_name,
                "mip_gap": 0.0 if status == GRB.OPTIMAL and objective is not None else mip_gap,
                "best_bound": objective if status == GRB.OPTIMAL and objective is not None else best_bound,
                "cs_copies": int(self.config.cs_copies),
                "node_count_with_copies": len(node_map.solver_to_terminal),
                "checkpoints_s": list(trace["checkpoints_s"]),
                "first_feasible_time_s": trace["first_feasible_time_s"],
                "checkpoint_snapshots": trace["checkpoint_snapshots"],
                "incumbent_events": trace["incumbent_events"],
                "tie_break_vehicle_count": bool(self.config.tie_break_vehicle_count),
                "tie_break_applied": bool(tie_break_applied),
                "stage1_best_distance_km": stage1_best_distance,
                "stage1_gurobi_status": stage1_status,
                "stage1_gurobi_status_name": stage1_status_name,
                "stage1_mip_gap": stage1_mip_gap,
                "stage1_best_bound": stage1_best_bound,
                "distance_tolerance": distance_tolerance,
                "tie_break_gurobi_status": tie_break_status,
                "tie_break_gurobi_status_name": tie_break_status_name,
                "warm_start": warm_start_info,
            },
        )

    def _apply_warm_start(
        self,
        instance: EVRPTWInstance,
        node_map: NodeMap,
        x: dict[tuple[int, int], Any],
        warm_start_routes: list[list[int]] | None,
    ) -> dict[str, Any]:
        if not warm_start_routes:
            return {"requested": False, "routes": 0, "arcs_set": 0, "missing_arcs": 0, "mapping_errors": 0}

        for var in x.values():
            var.Start = 0.0

        terminal_to_solver_nodes: dict[int, list[int]] = {}
        for solver_node, terminal in enumerate(node_map.solver_to_terminal):
            terminal_to_solver_nodes.setdefault(int(terminal), []).append(int(solver_node))

        used_cs_nodes: set[int] = set()
        arcs_set = 0
        missing_arcs = 0
        mapping_errors = 0

        def solver_node_for_terminal(terminal: int, route_pos: int) -> int | None:
            nonlocal mapping_errors
            if terminal == 0:
                return node_map.start_depot if route_pos == 0 else node_map.end_depot
            if 1 <= terminal <= instance.num_customers:
                return terminal
            candidates = terminal_to_solver_nodes.get(int(terminal), [])
            for candidate in candidates:
                if candidate not in used_cs_nodes and candidate not in {node_map.start_depot, node_map.end_depot}:
                    used_cs_nodes.add(candidate)
                    return candidate
            if candidates:
                mapping_errors += 1
                return candidates[0]
            mapping_errors += 1
            return None

        for route in warm_start_routes:
            if len(route) < 2:
                continue
            solver_route: list[int] = []
            for pos, terminal_raw in enumerate(route):
                terminal = int(terminal_raw)
                solver_node = solver_node_for_terminal(terminal, pos)
                if solver_node is None:
                    solver_route = []
                    break
                solver_route.append(solver_node)
            for i, j in zip(solver_route, solver_route[1:]):
                arc = (int(i), int(j))
                var = x.get(arc)
                if var is None:
                    missing_arcs += 1
                    continue
                var.Start = 1.0
                arcs_set += 1

        return {
            "requested": True,
            "routes": len(warm_start_routes),
            "arcs_set": arcs_set,
            "missing_arcs": missing_arcs,
            "mapping_errors": mapping_errors,
        }

    @staticmethod
    def _check_physical_matrices(instance):
        """This historical benchmark only models constant speed/consumption.

        Do not silently solve a different instance when callers supply genuine
        per-edge T/E. Float32 storage roundoff in proportional matrices is fine.
        """
        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        speed = float(instance.speed_profile.get("effective_speed_kmh") or instance.vehicle.get("design_speed_kmh") or 40.)
        consumption = float(instance.vehicle.get("consumption_kwh_per_km", .404))
        for name, expected, atol in (
            ("travel_time_matrix_s", distance / max(speed, 1e-9) * 3600., .005),
            ("energy_matrix_kwh", distance * consumption, 1e-5),
        ):
            actual = getattr(instance, name, None)
            if actual is None:
                actual = (getattr(instance, "raw", None) or {}).get(name)
            if actual is not None:
                actual = np.asarray(actual, dtype=float)
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-6, atol=atol):
                    raise ValueError(f"Historical Gurobi EVRPTW benchmark does not support nonproportional {name}; refusing to discard explicit edge costs")

    def _build_model(self, instance: EVRPTWInstance) -> tuple[Model, NodeMap, dict[tuple[int, int], Any], Any, Any]:
        self._check_physical_matrices(instance)
        n = instance.num_customers
        m = instance.num_charging_stations
        cs_copies = max(1, int(self.config.cs_copies)) if m else 0

        solver_to_terminal = [0]
        solver_to_terminal.extend(range(1, n + 1))
        for _ in range(cs_copies):
            solver_to_terminal.extend(range(n + 1, n + 1 + m))
        solver_to_terminal.append(0)

        start_depot = 0
        end_depot = len(solver_to_terminal) - 1
        customer_nodes = list(range(1, n + 1))
        cs_nodes = list(range(n + 1, end_depot))
        node_map = NodeMap(solver_to_terminal, customer_nodes, cs_nodes, start_depot, end_depot)

        terminals = np.asarray(solver_to_terminal, dtype=int)
        distance = instance.distance_matrix_km[np.ix_(terminals, terminals)].astype(float)
        effective_speed = float(instance.speed_profile.get("effective_speed_kmh") or instance.vehicle.get("design_speed_kmh") or 40.0)
        travel_s = distance / max(effective_speed, 1e-9) * 3600.0

        battery_capacity = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
        consumption = float(instance.vehicle.get("consumption_kwh_per_km", 0.404))
        cargo_capacity = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        full_charge_s = float(instance.vehicle.get("full_charge_time_s", 0.0))

        demand = np.zeros(len(solver_to_terminal), dtype=float)
        service = np.zeros(len(solver_to_terminal), dtype=float)
        ready = np.full(len(solver_to_terminal), float(instance.working_start_s), dtype=float)
        due = np.full(len(solver_to_terminal), float(instance.working_end_s), dtype=float)
        for local, customer_node in enumerate(customer_nodes):
            demand[customer_node] = float(instance.demands_cm3[local])
            service[customer_node] = float(instance.service_time_s[local])
            ready[customer_node] = float(instance.tw_s[local, 0])
            due[customer_node] = float(instance.tw_s[local, 1])

        charge_departure = np.zeros(len(solver_to_terminal), dtype=float)
        for cs_node in cs_nodes:
            charge_departure[cs_node] = full_charge_s

        model = Model(f"EVRPTW_{instance.instance_id}")
        model.Params.TimeLimit = capped_time_limit_s(self.config.time_limit_s)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.OutputFlag = int(self.config.output_flag)
        if self.config.threads is not None:
            model.Params.Threads = int(self.config.threads)

        recharge_nodes = {start_depot, *cs_nodes}
        end_nodes = set(customer_nodes + cs_nodes + [end_depot])
        start_nodes = set([start_depot] + customer_nodes + cs_nodes)
        arcs: list[tuple[int, int]] = []
        for i in start_nodes:
            for j in end_nodes:
                if i == j:
                    continue
                if i == start_depot and j == end_depot:
                    continue
                if i in cs_nodes and j in cs_nodes and solver_to_terminal[i] == solver_to_terminal[j]:
                    continue
                if not np.isfinite(distance[i, j]):
                    continue
                if consumption * distance[i, j] > battery_capacity + 1e-7:
                    continue
                arcs.append((i, j))

        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        tau = model.addVars(range(len(solver_to_terminal)), lb=ready.tolist(), ub=due.tolist(), vtype=GRB.CONTINUOUS, name="tau")
        load = model.addVars(range(len(solver_to_terminal)), lb=0.0, ub=cargo_capacity, vtype=GRB.CONTINUOUS, name="load")
        battery = model.addVars(range(len(solver_to_terminal)), lb=0.0, ub=battery_capacity, vtype=GRB.CONTINUOUS, name="battery")

        distance_expr = quicksum(float(distance[i, j]) * x[i, j] for i, j in arcs)
        model.setObjective(distance_expr, GRB.MINIMIZE)

        incoming = {node: [] for node in range(len(solver_to_terminal))}
        outgoing = {node: [] for node in range(len(solver_to_terminal))}
        for i, j in arcs:
            outgoing[i].append((i, j))
            incoming[j].append((i, j))

        for c in customer_nodes:
            model.addConstr(quicksum(x[a] for a in incoming[c]) == 1, name=f"customer_in_{c}")
            model.addConstr(quicksum(x[a] for a in outgoing[c]) == 1, name=f"customer_out_{c}")

        for f in cs_nodes:
            model.addConstr(quicksum(x[a] for a in incoming[f]) == quicksum(x[a] for a in outgoing[f]), name=f"cs_flow_{f}")
            model.addConstr(quicksum(x[a] for a in incoming[f]) <= 1, name=f"cs_visit_{f}")

        vehicle_expr = quicksum(x[a] for a in outgoing[start_depot])
        model.addConstr(vehicle_expr == quicksum(x[a] for a in incoming[end_depot]), name="depot_balance")
        model.addConstr(vehicle_expr >= 1, name="at_least_one_route")
        model.addConstr(tau[start_depot] == float(instance.working_start_s), name="start_time")
        model.addConstr(load[start_depot] == 0.0, name="start_load")
        model.addConstr(battery[start_depot] == battery_capacity, name="start_battery")

        horizon = float(instance.working_end_s - instance.working_start_s)
        max_arc_time = float(np.nanmax(travel_s[np.isfinite(travel_s)])) if np.any(np.isfinite(travel_s)) else 0.0
        big_m_time = max(1.0, horizon + max_arc_time + float(np.max(service)) + full_charge_s + 1.0)
        big_m_load = max(1.0, cargo_capacity + float(np.sum(instance.demands_cm3)) + 1.0)
        big_m_battery = max(1.0, battery_capacity + float(consumption * np.nanmax(distance[np.isfinite(distance)])) + 1.0)

        for i, j in arcs:
            model.addConstr(
                tau[j] >= tau[i] + float(service[i]) + float(charge_departure[i]) + float(travel_s[i, j]) - big_m_time * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )
            model.addConstr(
                load[j] >= load[i] + float(demand[j]) - big_m_load * (1 - x[i, j]),
                name=f"load_{i}_{j}",
            )
            energy = float(consumption * distance[i, j])
            if i in recharge_nodes:
                model.addConstr(
                    battery[j] <= battery_capacity - energy + big_m_battery * (1 - x[i, j]),
                    name=f"battery_recharge_{i}_{j}",
                )
            else:
                model.addConstr(
                    battery[j] <= battery[i] - energy + big_m_battery * (1 - x[i, j]),
                    name=f"battery_{i}_{j}",
                )

        return model, node_map, x, distance_expr, vehicle_expr

    @staticmethod
    def _route_distance_km(routes: list[list[int]], instance: EVRPTWInstance) -> float:
        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += float(distance[int(route[i]), int(route[i + 1])])
        return total

    def _new_trace(self) -> dict[str, Any]:
        checkpoints = tuple(sorted({float(t) for t in self.config.checkpoints_s if float(t) >= 0.0}))
        return {
            "checkpoints_s": checkpoints,
            "next_checkpoint_index": 0,
            "first_feasible_time_s": None,
            "last_incumbent": None,
            "last_best_bound": None,
            "last_best_obj": None,
            "checkpoint_snapshots": [],
            "incumbent_events": [],
        }

    def _make_callback(self, trace: dict[str, Any], node_map: NodeMap, x: dict[tuple[int, int], Any]):
        arcs = list(x.keys())
        x_vars = [x[arc] for arc in arcs]

        def callback(model: Model, where: int) -> None:
            runtime = self._callback_float(model, GRB.Callback.RUNTIME)
            if runtime is None:
                return

            if where == GRB.Callback.MIPSOL:
                objective = self._callback_float(model, GRB.Callback.MIPSOL_OBJ)
                best_bound = self._callback_float(model, GRB.Callback.MIPSOL_OBJBND)
                if best_bound is None:
                    best_bound = trace.get("last_best_bound")
                values = model.cbGetSolution(x_vars)
                arc_values = {arc: float(value) for arc, value in zip(arcs, values)}
                routes = self._extract_routes_from_arc_values(node_map, arc_values)
                snapshot = self._make_snapshot(
                    checkpoint_s=None,
                    elapsed_s=runtime,
                    reached_checkpoint=True,
                    solver_status="RUNNING",
                    objective_distance_km=objective,
                    best_bound=best_bound,
                    routes=routes,
                    source="incumbent",
                )
                trace["last_incumbent"] = snapshot
                trace["last_best_obj"] = objective
                if best_bound is not None:
                    trace["last_best_bound"] = best_bound
                if trace["first_feasible_time_s"] is None:
                    trace["first_feasible_time_s"] = runtime
                if self.config.record_incumbent_events:
                    event = dict(snapshot)
                    event.pop("routes", None)
                    event.pop("route_sequence", None)
                    trace["incumbent_events"].append(event)
                self._record_due_checkpoints(trace, runtime, "RUNNING")

            elif where == GRB.Callback.MIP:
                best_obj = self._callback_float(model, GRB.Callback.MIP_OBJBST)
                best_bound = self._callback_float(model, GRB.Callback.MIP_OBJBND)
                if best_obj is not None and math.isfinite(best_obj):
                    trace["last_best_obj"] = best_obj
                if best_bound is not None and math.isfinite(best_bound):
                    trace["last_best_bound"] = best_bound
                self._record_due_checkpoints(trace, runtime, "RUNNING")

        return callback

    def _record_due_checkpoints(self, trace: dict[str, Any], elapsed_s: float, solver_status: str) -> None:
        checkpoints = trace["checkpoints_s"]
        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            if elapsed_s < checkpoint_s:
                break
            trace["checkpoint_snapshots"].append(self._snapshot_at_checkpoint(trace, checkpoint_s, elapsed_s, True, solver_status))
            trace["next_checkpoint_index"] += 1

    def _finalize_checkpoints(self, trace: dict[str, Any], final_snapshot: dict[str, Any], runtime_s: float, solver_status: str) -> None:
        checkpoints = trace["checkpoints_s"]
        if final_snapshot.get("has_incumbent"):
            trace["last_incumbent"] = final_snapshot
            trace["last_best_obj"] = final_snapshot.get("objective_distance_km")
        if final_snapshot.get("best_bound") is not None:
            trace["last_best_bound"] = final_snapshot.get("best_bound")

        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            reached = runtime_s >= checkpoint_s
            elapsed = checkpoint_s if reached else runtime_s
            trace["checkpoint_snapshots"].append(self._snapshot_at_checkpoint(trace, checkpoint_s, elapsed, reached, solver_status))
            trace["next_checkpoint_index"] += 1

    def _snapshot_at_checkpoint(
        self,
        trace: dict[str, Any],
        checkpoint_s: float,
        elapsed_s: float,
        reached_checkpoint: bool,
        solver_status: str,
    ) -> dict[str, Any]:
        incumbent = trace.get("last_incumbent")
        if incumbent is None:
            return self._make_snapshot(
                checkpoint_s=checkpoint_s,
                elapsed_s=elapsed_s,
                reached_checkpoint=reached_checkpoint,
                solver_status=solver_status,
                objective_distance_km=None,
                best_bound=trace.get("last_best_bound"),
                routes=[],
                source="checkpoint_no_incumbent" if reached_checkpoint else "final_no_incumbent",
            )

        return self._make_snapshot(
            checkpoint_s=checkpoint_s,
            elapsed_s=elapsed_s,
            reached_checkpoint=reached_checkpoint,
            solver_status=solver_status,
            objective_distance_km=incumbent.get("objective_distance_km"),
            best_bound=trace.get("last_best_bound", incumbent.get("best_bound")),
            routes=incumbent.get("routes", []),
            source="checkpoint_incumbent" if reached_checkpoint else "final_after_early_stop",
        )

    def _make_snapshot(
        self,
        checkpoint_s: float | None,
        elapsed_s: float,
        reached_checkpoint: bool,
        solver_status: str,
        objective_distance_km: float | None,
        best_bound: float | None,
        routes: list[list[int]],
        source: str,
    ) -> dict[str, Any]:
        gap = self._relative_gap(objective_distance_km, best_bound)
        return {
            "checkpoint_s": checkpoint_s,
            "elapsed_s": float(elapsed_s),
            "reached_checkpoint": bool(reached_checkpoint),
            "solver_status": solver_status,
            "has_incumbent": bool(routes),
            "objective_distance_km": objective_distance_km,
            "best_bound": best_bound,
            "mip_gap": gap,
            "vehicle_count": len(routes) if routes else None,
            "routes": routes,
            "route_sequence": self._flatten_routes(routes),
            "source": source,
        }

    def _extract_routes(self, node_map: NodeMap, x: dict[tuple[int, int], Any]) -> list[list[int]]:
        return self._extract_routes_from_arc_values(node_map, {arc: float(var.X) for arc, var in x.items()})

    def _extract_routes_from_arc_values(self, node_map: NodeMap, arc_values: dict[tuple[int, int], float]) -> list[list[int]]:
        outgoing: dict[int, list[int]] = {}
        for (i, j), value in arc_values.items():
            if value > 0.5:
                outgoing.setdefault(i, []).append(j)

        routes: list[list[int]] = []
        starts = sorted(outgoing.get(node_map.start_depot, []))
        for first in starts:
            route_solver_nodes = [node_map.start_depot, first]
            current = first
            seen = {node_map.start_depot}
            while current != node_map.end_depot:
                if current in seen:
                    break
                seen.add(current)
                next_nodes = sorted(outgoing.get(current, []))
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                route_solver_nodes.append(nxt)
                current = nxt

            mapped = []
            for solver_node in route_solver_nodes:
                terminal = int(node_map.solver_to_terminal[solver_node])
                if solver_node == node_map.end_depot:
                    terminal = 0
                if not mapped or mapped[-1] != terminal:
                    mapped.append(terminal)
            if mapped and mapped[-1] != 0:
                mapped.append(0)
            routes.append(mapped)
        return routes

    @staticmethod
    def _flatten_routes(routes: list[list[int]]) -> list[int]:
        return merge_route_sequences(routes)

    @staticmethod
    def _relative_gap(objective: float | None, bound: float | None) -> float | None:
        if objective is None or bound is None:
            return None
        if not math.isfinite(objective) or not math.isfinite(bound):
            return None
        denom = max(abs(objective), 1e-9)
        return float(abs(objective - bound) / denom)

    @staticmethod
    def _callback_float(model: Model, what: int) -> float | None:
        try:
            value = float(model.cbGet(what))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _safe_model_float(model: Model, attr: str) -> float | None:
        try:
            value = float(getattr(model, attr))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _status_name(status: int) -> str:
        names: dict[int, str] = {}
        for name in (
            "OPTIMAL",
            "INFEASIBLE",
            "INF_OR_UNBD",
            "UNBOUNDED",
            "CUTOFF",
            "ITERATION_LIMIT",
            "NODE_LIMIT",
            "TIME_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "NUMERIC",
            "SUBOPTIMAL",
            "INPROGRESS",
            "USER_OBJ_LIMIT",
        ):
            value = getattr(GRB, name, None)
            if value is not None:
                names[int(value)] = name
        return names.get(status, str(status))
