from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without numba installed.
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _charge_time_s_jit(
        battery_used_kwh: float,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        fixed_full_charge: bool,
    ) -> float:
        if fixed_full_charge:
            return full_charge_time_s
        ratio = battery_used_kwh / max(battery_capacity_kwh, 1e-12)
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0
        return ratio * full_charge_time_s


    @njit(cache=True)
    def _direct_depot_feasible_jit(
        start: int,
        current_time_s: float,
        battery_used_kwh: float,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        battery_capacity_kwh: float,
        working_end_s: float,
    ) -> bool:
        battery_after = battery_used_kwh + energy_kwh[start, 0]
        arrival = current_time_s + travel_time_s[start, 0]
        return battery_after <= battery_capacity_kwh + 1e-9 and arrival <= working_end_s + 1e-9


    @njit(cache=True)
    def _available_stop_return_times_jit(station_start, num_nodes, travel_time_s, energy_kwh,
                                         visited_stations, battery_capacity_kwh,
                                         full_charge_time_s, fixed_full_charge):
        # Reverse Dijkstra on available stops, once per trajectory/mask. Every
        # stop departure has a full battery; exclude all previously used CS.
        costs = np.full(num_nodes, np.inf, dtype=np.float64)
        closed = np.zeros(num_nodes, dtype=np.bool_)
        costs[0] = 0.0
        for _ in range(num_nodes - station_start + 1):
            best, cost = -1, np.inf
            if not closed[0]:
                best, cost = 0, costs[0]
            for node in range(station_start, num_nodes):
                if not visited_stations[node] and not closed[node] and costs[node] < cost:
                    best, cost = node, costs[node]
            if best < 0 or not np.isfinite(cost):
                break
            closed[best] = True
            for previous in range(station_start, num_nodes):
                if visited_stations[previous] or closed[previous] or previous == best:
                    continue
                energy = energy_kwh[previous, best]
                if energy > battery_capacity_kwh + 1e-9:
                    continue
                charge = 0.0 if best == 0 else _charge_time_s_jit(
                    energy, battery_capacity_kwh, full_charge_time_s, fixed_full_charge)
                candidate = cost + travel_time_s[previous, best] + charge
                if candidate < costs[previous]:
                    costs[previous] = candidate
        return costs


    @njit(cache=True)
    def _can_return_to_depot_jit(
        start: int,
        current_time_s: float,
        battery_used_kwh: float,
        station_start: int,
        num_nodes: int,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        stop_to_depot_time_s: np.ndarray,
        cs_visited_current_route: np.ndarray,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        working_end_s: float,
        fixed_full_charge: bool,
    ) -> bool:
        if start == 0:
            return True
        if (battery_used_kwh + energy_kwh[start, 0] <= battery_capacity_kwh + 1e-9
                and current_time_s + travel_time_s[start, 0] <= working_end_s + 1e-9):
            return True

        for first in range(station_start, num_nodes):
            if first == start:
                continue
            if cs_visited_current_route[first]:
                continue
            battery_at_first = battery_used_kwh + energy_kwh[start, first]
            if battery_at_first > battery_capacity_kwh + 1e-9:
                continue
            time_at_first = current_time_s + travel_time_s[start, first]
            depart_first = time_at_first + _charge_time_s_jit(
                battery_at_first,
                battery_capacity_kwh,
                full_charge_time_s,
                fixed_full_charge,
            )
            stop_plan = stop_to_depot_time_s[first]
            if not np.isfinite(stop_plan):
                continue
            if depart_first + stop_plan <= working_end_s + 1e-9:
                return True
        return False


    @njit(cache=True)
    def _compute_action_mask_jit(
        n_traj: int,
        num_nodes: int,
        num_customers: int,
        station_start: int,
        last: np.ndarray,
        visited: np.ndarray,
        cs_visited_current_route: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        served_customers: np.ndarray,
        route_has_customer: np.ndarray,
        current_time_s: np.ndarray,
        battery_used_kwh: np.ndarray,
        load_cm3: np.ndarray,
        demand_cm3: np.ndarray,
        service_time_s: np.ndarray,
        tw_s: np.ndarray,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        stop_to_depot_time_s: np.ndarray,
        battery_capacity_kwh: float,
        cargo_capacity_cm3: float,
        full_charge_time_s: float,
        working_end_s: float,
        fixed_full_charge: bool,
    ) -> np.ndarray:
        mask = np.zeros((n_traj, num_nodes), dtype=np.bool_)
        for t in range(n_traj):
            if terminated[t] or truncated[t]:
                mask[t, 0] = True
                continue

            start = int(last[t])
            all_served = served_customers[t] == num_customers
            return_times = stop_to_depot_time_s
            if np.any(cs_visited_current_route[t]):
                return_times = _available_stop_return_times_jit(
                    station_start, num_nodes, travel_time_s, energy_kwh,
                    cs_visited_current_route[t], battery_capacity_kwh,
                    full_charge_time_s, fixed_full_charge)
            if all_served:
                if start == 0:
                    mask[t, 0] = True
                elif _direct_depot_feasible_jit(
                    start,
                    current_time_s[t],
                    battery_used_kwh[t],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    working_end_s,
                ):
                    mask[t, 0] = True
                # If direct return is impossible, fall through to available CS.
                if mask[t, 0]:
                    continue

            if (
                start != 0
                and route_has_customer[t]
                and _direct_depot_feasible_jit(
                    start,
                    current_time_s[t],
                    battery_used_kwh[t],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    working_end_s,
                )
            ):
                mask[t, 0] = True

            for customer in range(1, 1 + num_customers):
                if visited[t, customer]:
                    continue
                battery_after = battery_used_kwh[t] + energy_kwh[start, customer]
                if battery_after > battery_capacity_kwh + 1e-9:
                    continue
                if load_cm3[t] + demand_cm3[customer] > cargo_capacity_cm3 + 1e-9:
                    continue
                arrival = current_time_s[t] + travel_time_s[start, customer]
                ready = tw_s[customer, 0]
                due = tw_s[customer, 1]
                service_start = arrival
                if ready > service_start:
                    service_start = ready
                service_departure = service_start + service_time_s[customer]
                if service_start > due + 1e-9 or service_departure > working_end_s + 1e-9:
                    continue
                if _can_return_to_depot_jit(
                    customer,
                    service_departure,
                    battery_after,
                    station_start,
                    num_nodes,
                    travel_time_s,
                    energy_kwh,
                    return_times,
                    cs_visited_current_route[t],
                    battery_capacity_kwh,
                    full_charge_time_s,
                    working_end_s,
                    fixed_full_charge,
                ):
                    mask[t, customer] = True

            for station in range(station_start, num_nodes):
                if station == start or cs_visited_current_route[t, station]:
                    continue
                battery_after = battery_used_kwh[t] + energy_kwh[start, station]
                if battery_after > battery_capacity_kwh + 1e-9:
                    continue
                arrival = current_time_s[t] + travel_time_s[start, station]
                departure = arrival + _charge_time_s_jit(
                    battery_after,
                    battery_capacity_kwh,
                    full_charge_time_s,
                    fixed_full_charge,
                )
                if departure > working_end_s + 1e-9:
                    continue
                if _can_return_to_depot_jit(
                    station,
                    departure,
                    0.0,
                    station_start,
                    num_nodes,
                    travel_time_s,
                    energy_kwh,
                    return_times,
                    cs_visited_current_route[t],
                    battery_capacity_kwh,
                    full_charge_time_s,
                    working_end_s,
                    fixed_full_charge,
                ):
                    mask[t, station] = True
        return mask


def compute_action_mask_jit(
    *,
    n_traj: int,
    num_nodes: int,
    num_customers: int,
    station_start: int,
    last: np.ndarray,
    visited: np.ndarray,
    cs_visited_current_route: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    served_customers: np.ndarray,
    route_has_customer: np.ndarray,
    current_time_s: np.ndarray,
    battery_used_kwh: np.ndarray,
    load_cm3: np.ndarray,
    demand_cm3: np.ndarray,
    service_time_s: np.ndarray,
    tw_s: np.ndarray,
    travel_time_s: np.ndarray,
    energy_kwh: np.ndarray,
    stop_to_depot_time_s: np.ndarray,
    battery_capacity_kwh: float,
    cargo_capacity_cm3: float,
    full_charge_time_s: float,
    working_end_s: float,
    fixed_full_charge: bool,
) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    return _compute_action_mask_jit(
        int(n_traj),
        int(num_nodes),
        int(num_customers),
        int(station_start),
        last,
        visited,
        cs_visited_current_route,
        terminated,
        truncated,
        served_customers,
        route_has_customer,
        current_time_s,
        battery_used_kwh,
        load_cm3,
        demand_cm3,
        service_time_s,
        tw_s,
        travel_time_s,
        energy_kwh,
        stop_to_depot_time_s,
        float(battery_capacity_kwh),
        float(cargo_capacity_cm3),
        float(full_charge_time_s),
        float(working_end_s),
        bool(fixed_full_charge),
    )


__all__ = ["NUMBA_AVAILABLE", "compute_action_mask_jit"]
