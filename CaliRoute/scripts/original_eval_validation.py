"""Independent NumPy-only measurements for an immutable original VRPTW policy.

The validation contract matches the modern route evaluator but imports no modern
model, trainer or environment code. The legacy environment may use D/v even when
an input contains authoritative edge times; we expose that discrepancy here.
"""
from __future__ import annotations
from typing import Any
import numpy as np


def resolve_validation_travel_time(instance, *, raw_payload=None):
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    value = getattr(instance, 'travel_time_matrix_s', None)
    if value is None:
        value = (getattr(instance, 'raw', None) or {}).get('travel_time_matrix_s')
    if value is None and raw_payload is not None:
        for name in ('travel_time_matrix_s', 'time_matrix_s', 'travel_time_matrix'):
            if raw_payload.get(name) is not None:
                value = raw_payload[name]
                break
    if value is not None:
        travel = np.asarray(value, dtype=np.float64)
        if travel.shape != distance.shape or np.isnan(travel).any() or (travel < 0).any():
            raise ValueError('Authoritative travel_time_matrix_s must be nonnegative and match distance shape')
        if not np.array_equal(np.isfinite(travel), np.isfinite(distance)):
            raise ValueError('Authoritative travel_time_matrix_s reachability must match distance_matrix_km')
        return travel, 'provided_travel_time_matrix_s'
    speed = float(instance.speed_profile.get('effective_speed_kmh')
                  or instance.vehicle.get('design_speed_kmh') or 40.)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('effective speed must be finite and positive')
    return distance / max(speed / 3600., 1e-12), 'distance_over_effective_speed'


def select_original_best_index(objective, success, served):
    """Exactly f388343's min selection, including unfinished/tie fallbacks."""
    objective = np.asarray(objective, dtype=np.float64).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    served = np.asarray(served, dtype=np.float64).reshape(-1)
    finite_obj = np.isfinite(objective)
    success_mask = np.zeros(objective.shape, dtype=bool)
    success_mask[:min(success.size, objective.size)] = success[:min(success.size, objective.size)]
    candidates = success_mask & finite_obj
    if not candidates.any():
        if served.size:
            served_pad = np.full(objective.shape, np.nan, dtype=np.float64)
            served_pad[:min(served.size, objective.size)] = served[:min(served.size, objective.size)]
            finite_served = np.isfinite(served_pad)
            if np.any(finite_served & finite_obj):
                maximum = float(np.nanmax(served_pad[finite_served & finite_obj]))
                candidates = finite_obj & (served_pad == maximum)
        if not candidates.any():
            candidates = finite_obj
    indices = np.where(candidates)[0]
    return int(indices[np.argsort(objective[indices])[0]]) if indices.size else None


def validate_cvrp_route(instance, row: dict[str, Any]) -> dict[str, Any]:
    """Independently verify the exported CVRP route against the instance arrays."""
    n = int(instance.num_customers)
    demands = np.asarray(instance.demands_cm3, dtype=np.float64)
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    capacity = float(instance.vehicle["cargo_capacity_cm3"])
    routes = row.get("routes", [])
    coverage = np.zeros(n, dtype=np.int64)
    indices_ok = True
    depot_ok = bool(routes)
    loads = []
    recomputed_distance = 0.0
    for route in routes:
        nodes = []
        for node in route:
            if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) <= n:
                indices_ok = False
                continue
            nodes.append(int(node))
        if len(nodes) != len(route) or len(nodes) < 2 or nodes[0] != 0 or nodes[-1] != 0 or 0 in nodes[1:-1]:
            depot_ok = False
        customers = [node for node in nodes if node != 0]
        for node in customers:
            coverage[node - 1] += 1
        loads.append(float(sum(demands[node - 1] for node in customers)))
        if len(nodes) == len(route):
            recomputed_distance += sum(float(distance[a, b]) for a, b in zip(nodes, nodes[1:]))
    coverage_ok = bool(np.all(coverage == 1))
    capacity_ok = bool(all(load <= capacity + 1e-6 * max(1.0, capacity) for load in loads))
    objective = float(row.get("objective_distance_km", float("nan")))
    distance_ok = bool(indices_ok and np.isfinite(objective) and np.isclose(
        recomputed_distance, objective, rtol=1e-6, atol=1e-5,
    ))
    return {
        "checked": True,
        "problem_type": "cvrp",
        "valid": bool(indices_ok and depot_ok and coverage_ok and capacity_ok and distance_ok),
        "indices_valid": indices_ok,
        "depot_endpoints_valid": depot_ok,
        "customer_coverage_valid": coverage_ok,
        "missing_customers": (np.where(coverage == 0)[0] + 1).tolist(),
        "repeated_customers": (np.where(coverage > 1)[0] + 1).tolist(),
        "capacity_valid": capacity_ok,
        "route_loads_cm3": loads,
        "cargo_capacity_cm3": capacity,
        "distance_matches": distance_ok,
        "recomputed_distance_km": recomputed_distance if indices_ok else None,
        "distance_error_km": recomputed_distance - objective if indices_ok and np.isfinite(objective) else None,
        "distance_atol_km": 1e-5,
        "distance_rtol": 1e-6,
    }


def validate_vrptw_route(instance, row: dict[str, Any], *, raw_payload=None) -> dict[str, Any]:
    """Verify exported VRPTW routes independently of the environment state.

    Times prefer authoritative directed edge seconds; D/v is the fallback.
    This measures old routes under the same physical contract as modern eval. Customer windows constrain service *start*,
    waiting is allowed, and each vehicle starts at working_start_s. Service
    completion and depot return must respect working_end_s. The environment's
    direct-return feasibility check is also applied after every customer.
    """
    result = validate_cvrp_route(instance, row)
    spatial_valid = result["valid"]
    tolerance = 1e-9  # Match the actual action mask; do not relax feasibility.
    n = int(instance.num_customers)
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    windows = np.asarray(instance.tw_s, dtype=np.float64)
    service = np.asarray(instance.service_time_s, dtype=np.float64)
    start = float(instance.working_start_s)
    end = float(instance.working_end_s)
    speed = float(
        instance.speed_profile.get("effective_speed_kmh")
        or instance.vehicle.get("design_speed_kmh")
        or 40.0
    )
    inputs_valid = bool(
        windows.shape == (n, 2) and service.shape == (n,)
        and distance.shape == (n + 1, n + 1)
        and np.all(np.isfinite(windows)) and np.all(windows[:, 0] <= windows[:, 1])
        and np.all(np.isfinite(service)) and np.all(service >= 0)
        and np.all(np.isfinite(distance)) and np.all(distance >= 0)
        and np.isfinite(start) and np.isfinite(end) and start <= end
        and np.isfinite(speed) and speed > 0
        and int(getattr(instance, "num_charging_stations", 0)) == 0
    )
    # CVRP's distance comparison allows accumulated roundoff; capacity follows
    # VRPTW's stricter environment tolerance independently of that comparison.
    capacity = float(instance.vehicle["cargo_capacity_cm3"])
    capacity_valid = bool(all(load <= capacity + tolerance for load in result["route_loads_cm3"]))
    result.update({
        "problem_type": "vrptw",
        "valid": False,
        "capacity_valid": capacity_valid,
        "capacity_atol_cm3": tolerance,
        "temporal_inputs_valid": inputs_valid,
        "time_windows_valid": False,
        "service_completion_valid": False,
        "depot_return_valid": False,
        "return_reachability_valid": False,
        "time_atol_s": tolerance,
        "time_window_semantics": "service_start_in_window; waiting_allowed; each_route_clock_resets",
        "travel_time_source": "adapted_directed_distance_divided_by_environment_effective_speed",
        "route_return_times_s": [],
        "route_waiting_times_s": [],
        "time_window_violations": [],
        "service_completion_violations": [],
        "depot_return_violations": [],
        "return_reachability_violations": [],
    })
    if not inputs_valid or not result["indices_valid"] or not result["depot_endpoints_valid"]:
        return result

    travel, travel_source = resolve_validation_travel_time(instance, raw_payload=raw_payload)
    result['travel_time_source'] = travel_source
    for route_index, route in enumerate(row.get("routes", [])):
        clock = start
        waiting = 0.0
        for previous, node in zip(route, route[1:]):
            clock += float(travel[previous, node])
            if node == 0:
                if clock > end + tolerance:
                    result["depot_return_violations"].append({
                        "route_index": route_index, "return_time_s": clock, "deadline_s": end,
                    })
                continue
            ready, due = windows[node - 1]
            service_start = max(clock, float(ready))
            waiting += service_start - clock
            if service_start > float(due) + tolerance:
                result["time_window_violations"].append({
                    "route_index": route_index, "customer": int(node),
                    "service_start_s": service_start, "due_s": float(due),
                })
            clock = service_start + float(service[node - 1])
            if clock > end + tolerance:
                result["service_completion_violations"].append({
                    "route_index": route_index, "customer": int(node),
                    "service_completion_s": clock, "deadline_s": end,
                })
            direct_return = clock + float(travel[node, 0])
            if direct_return > end + tolerance:
                result["return_reachability_violations"].append({
                    "route_index": route_index, "customer": int(node),
                    "earliest_direct_return_s": direct_return, "deadline_s": end,
                })
        result["route_return_times_s"].append(clock)
        result["route_waiting_times_s"].append(waiting)
    result["time_windows_valid"] = not result["time_window_violations"]
    result["service_completion_valid"] = not result["service_completion_violations"]
    result["depot_return_valid"] = not result["depot_return_violations"]
    result["return_reachability_valid"] = not result["return_reachability_violations"]
    result["valid"] = bool(spatial_valid and capacity_valid and all(result[key] for key in (
        "time_windows_valid", "service_completion_valid", "depot_return_valid", "return_reachability_valid",
    )))
    return result
