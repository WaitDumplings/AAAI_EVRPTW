"""Independent NumPy-only route measurements for immutable original policies.

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


def resolve_validation_energy(instance, *, raw_payload=None):
    """Preserve authoritative directed kWh; no inference from Euclidean inputs."""
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    value = getattr(instance, 'energy_matrix_kwh', None)
    if value is None:
        value = (getattr(instance, 'raw', None) or {}).get('energy_matrix_kwh')
    if value is None and raw_payload is not None:
        value = raw_payload.get('energy_matrix_kwh')
    if value is not None:
        energy = np.asarray(value, dtype=np.float64)
        if energy.shape != distance.shape or np.isnan(energy).any() or (energy < 0).any():
            raise ValueError('Authoritative energy_matrix_kwh must be nonnegative and match distance shape')
        if not np.array_equal(np.isfinite(energy), np.isfinite(distance)):
            raise ValueError('Authoritative energy_matrix_kwh reachability must match distance_matrix_km')
        return energy, 'provided_energy_matrix_kwh'
    consumption = float(instance.vehicle.get('consumption_kwh_per_km', 0.404))
    if not np.isfinite(consumption) or consumption < 0:
        raise ValueError('consumption_kwh_per_km must be finite and nonnegative')
    energy = np.full_like(distance, np.inf)
    np.multiply(distance, consumption, out=energy, where=np.isfinite(distance))
    return energy, 'distance_times_consumption'


def validate_evrptw_route(instance, row: dict[str, Any], *, raw_payload=None,
                          charging_mode='fixed_full') -> dict[str, Any]:
    """Independently replay physical EVRPTW routes in km, seconds and kWh.

    Supported scope: homogeneous vehicles, full batteries at each depot start,
    fixed-time full recharge at every CS visit, customer service-start windows,
    and independently starting routes. Matrix rows are depot, customers, then
    physical stations. No Gurobi virtual CS-copy budget is imposed on these
    physical node IDs. Repeated CS visits are physical events, never coverage
    errors. Environment action restrictions remain a separate success test.

    Actual route suffixes provide return-to-depot witnesses: this verifier does
    not reproduce an environment's conservative lookahead mask. Unsupported
    charging modes and malformed instance arrays fail explicitly; infeasible
    candidate routes return a diagnostic ``valid=False`` result. No project
    environment, trainer, model, or learned checkpoint is imported.
    """
    if charging_mode != 'fixed_full':
        raise ValueError('Independent EVRPTW route validation currently supports charging_mode=fixed_full only')
    tolerance = 1e-9
    n = int(instance.num_customers)
    m = int(instance.num_charging_stations)
    terminals = 1 + n + m
    if n < 1 or m < 0:
        raise ValueError('EVRPTW validation requires positive customer and nonnegative station counts')
    distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    if distance.shape != (terminals, terminals) or np.isnan(distance).any() or (distance < 0).any():
        raise ValueError('distance_matrix_km must match EVRPTW terminals and contain nonnegative costs or +inf')
    demands = np.asarray(instance.demands_cm3, dtype=np.float64)
    service = np.asarray(instance.service_time_s, dtype=np.float64)
    windows = np.asarray(instance.tw_s, dtype=np.float64)
    if (demands.shape != (n,) or not np.isfinite(demands).all() or (demands < 0).any()
            or service.shape != (n,) or not np.isfinite(service).all() or (service < 0).any()
            or windows.shape != (n, 2) or not np.isfinite(windows).all()
            or (windows[:, 0] > windows[:, 1]).any()):
        raise ValueError('EVRPTW demands/service/windows must match customers with finite valid values')
    start, end = float(instance.working_start_s), float(instance.working_end_s)
    capacity = float(instance.vehicle.get('cargo_capacity_cm3', np.inf))
    battery = float(instance.vehicle.get('battery_capacity_kwh', 100.0))
    charge_time = float(instance.vehicle.get('full_charge_time_s', 0.0))
    if (not np.isfinite(start) or not np.isfinite(end) or start > end
            or np.isnan(capacity) or capacity <= 0
            or not np.isfinite(battery) or battery <= 0
            or not np.isfinite(charge_time) or charge_time < 0):
        raise ValueError('EVRPTW working window and vehicle capacities/charge time are invalid')
    travel, travel_source = resolve_validation_travel_time(instance, raw_payload=raw_payload)
    energy, energy_source = resolve_validation_energy(instance, raw_payload=raw_payload)
    result = {
        'checked': True, 'problem_type': 'evrptw', 'charging_mode': charging_mode,
        'valid': False, 'physical_inputs_valid': True,
        'validation_scope': 'complete_route_physics; environment_action_masks_checked_separately',
        'station_visit_semantics': 'physical_station_ids; recharge_on_each_visit; no_virtual_copy_limit',
        'time_window_semantics': 'service_start_in_window; waiting_allowed; each_route_clock_resets',
        'travel_time_source': travel_source, 'energy_source': energy_source,
        'indices_valid': True, 'depot_endpoints_valid': True,
        'routes_have_customers': True, 'customer_coverage_valid': False,
        'capacity_valid': True, 'battery_valid': True, 'time_windows_valid': True,
        'service_completion_valid': True, 'charging_completion_valid': True,
        'depot_return_valid': True, 'traversed_edges_reachable': True,
        'distance_matches': False, 'recomputed_distance_km': None, 'distance_error_km': None,
        'cargo_capacity_cm3': capacity, 'battery_capacity_kwh': battery,
        'full_charge_time_s': charge_time, 'time_atol_s': tolerance,
        'capacity_atol_cm3': tolerance, 'energy_atol_kwh': tolerance,
        'distance_atol_km': 1e-5, 'distance_rtol': 1e-6,
        'missing_customers': [], 'repeated_customers': [],
        'route_loads_cm3': [], 'route_return_times_s': [], 'route_waiting_times_s': [],
        'route_max_battery_used_kwh': [], 'route_charge_counts': [],
        'charge_events': [], 'capacity_violations': [], 'battery_violations': [],
        'time_window_violations': [], 'service_completion_violations': [],
        'charging_completion_violations': [], 'depot_return_violations': [],
        'unreachable_edges': [],
    }
    supplied = row.get('routes', [])
    if not isinstance(supplied, (list, tuple)) or not supplied:
        result['depot_endpoints_valid'] = False
        result['missing_customers'] = list(range(1, n + 1))
        return result
    routes = []
    coverage = np.zeros(n, dtype=np.int64)
    for route in supplied:
        if not isinstance(route, (list, tuple, np.ndarray)) or (isinstance(route, np.ndarray) and route.ndim != 1):
            result['indices_valid'] = result['depot_endpoints_valid'] = False
            continue
        nodes = []
        for node in route:
            if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) < terminals:
                result['indices_valid'] = False
                continue
            nodes.append(int(node))
            if 1 <= int(node) <= n:
                coverage[int(node) - 1] += 1
        if len(nodes) != len(route) or len(nodes) < 3 or nodes[0] != 0 or nodes[-1] != 0 or 0 in nodes[1:-1]:
            result['depot_endpoints_valid'] = False
        if not any(1 <= node <= n for node in nodes):
            result['routes_have_customers'] = False
        routes.append(nodes)
    result['missing_customers'] = (np.where(coverage == 0)[0] + 1).tolist()
    result['repeated_customers'] = (np.where(coverage > 1)[0] + 1).tolist()
    result['customer_coverage_valid'] = bool(np.all(coverage == 1))
    result['recomputed_vehicle_count'] = len(routes)
    if not result['indices_valid'] or not result['depot_endpoints_valid']:
        return result

    total_distance = 0.0
    for route_index, nodes in enumerate(routes):
        clock, used_energy, load, waiting = start, 0.0, 0.0, 0.0
        peak_energy, charges = 0.0, 0
        for previous, node in zip(nodes, nodes[1:]):
            d, t, e = float(distance[previous, node]), float(travel[previous, node]), float(energy[previous, node])
            if not all(np.isfinite(value) for value in (d, t, e)):
                result['unreachable_edges'].append({'route_index': route_index, 'from': previous, 'to': node})
            total_distance += d
            clock += t
            used_energy += e
            peak_energy = max(peak_energy, used_energy)
            # A station is usable only if the vehicle can arrive before recharge.
            if used_energy > battery + tolerance:
                result['battery_violations'].append({'route_index': route_index, 'from': previous, 'to': node,
                                                    'battery_used_kwh': used_energy, 'capacity_kwh': battery})
            if node == 0:
                if clock > end + tolerance:
                    result['depot_return_violations'].append({'route_index': route_index, 'return_time_s': clock, 'deadline_s': end})
            elif node <= n:
                load += float(demands[node - 1])
                if load > capacity + tolerance:
                    result['capacity_violations'].append({'route_index': route_index, 'customer': node, 'load_cm3': load, 'capacity_cm3': capacity})
                ready, due = windows[node - 1]
                service_start = max(clock, float(ready))
                if np.isfinite(clock):
                    waiting += service_start - clock
                if service_start > float(due) + tolerance:
                    result['time_window_violations'].append({'route_index': route_index, 'customer': node, 'service_start_s': service_start, 'due_s': float(due)})
                clock = service_start + float(service[node - 1])
                if clock > end + tolerance:
                    result['service_completion_violations'].append({'route_index': route_index, 'customer': node, 'service_completion_s': clock, 'deadline_s': end})
            else:
                arrival = clock
                clock += charge_time
                result['charge_events'].append({'route_index': route_index, 'station': node,
                    'arrival_time_s': arrival, 'departure_time_s': clock,
                    'battery_used_before_charge_kwh': used_energy, 'battery_used_after_charge_kwh': 0.0,
                    'charge_time_s': charge_time})
                charges += 1
                if clock > end + tolerance:
                    result['charging_completion_violations'].append({'route_index': route_index, 'station': node, 'departure_time_s': clock, 'deadline_s': end})
                used_energy = 0.0
        result['route_loads_cm3'].append(load)
        result['route_return_times_s'].append(clock)
        result['route_waiting_times_s'].append(waiting)
        result['route_max_battery_used_kwh'].append(peak_energy)
        result['route_charge_counts'].append(charges)
    try:
        objective = float(row.get('objective_distance_km', np.nan))
    except (TypeError, ValueError):
        objective = np.nan
    result['distance_matches'] = bool(np.isfinite(total_distance) and np.isfinite(objective)
        and np.isclose(total_distance, objective, rtol=1e-6, atol=1e-5))
    result['recomputed_distance_km'] = total_distance if np.isfinite(total_distance) else None
    result['distance_error_km'] = total_distance - objective if np.isfinite(total_distance) and np.isfinite(objective) else None
    for valid_key, violations_key in (
        ('capacity_valid', 'capacity_violations'), ('battery_valid', 'battery_violations'),
        ('time_windows_valid', 'time_window_violations'), ('service_completion_valid', 'service_completion_violations'),
        ('charging_completion_valid', 'charging_completion_violations'), ('depot_return_valid', 'depot_return_violations'),
        ('traversed_edges_reachable', 'unreachable_edges'),
    ):
        result[valid_key] = not result[violations_key]
    result['valid'] = all(result[key] for key in ('indices_valid', 'depot_endpoints_valid',
        'routes_have_customers', 'customer_coverage_valid', 'capacity_valid', 'battery_valid',
        'time_windows_valid', 'service_completion_valid', 'charging_completion_valid',
        'depot_return_valid', 'traversed_edges_reachable', 'distance_matches'))
    return result
