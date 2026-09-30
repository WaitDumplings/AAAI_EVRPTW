from __future__ import annotations

import math
from typing import Any

import numpy as np

from .env import EVRPTWVectorEnv, Transition
from .mask_jit import NUMBA_AVAILABLE, compute_action_mask_jit


class EVRPTWVectorEnvFast(EVRPTWVectorEnv):
    """Experimental drop-in optimized EVRP-TW-D env.

    This class intentionally lives beside the reference implementation so the
    reference dynamics remain available. It keeps the exact EVRP transition
    semantics while removing repeated work that dominates larger Cus/CS settings:

    - cache the previous action mask and reuse it to validate the next action;
    - precompute stop-node shortest return times to depot once per instance;
    - cache static observation arrays that do not change during a rollout;
    - optionally compute action masks through a numba JIT array kernel;
    - optionally return light training info without route reconstruction.
    """

    def __init__(self, *args: Any, info_level: str = "full", use_jit_mask: bool = True, **kwargs: Any) -> None:
        if info_level not in {"full", "light"}:
            raise ValueError("info_level must be 'full' or 'light'")
        self.info_level = info_level
        self.use_jit_mask = bool(use_jit_mask and NUMBA_AVAILABLE)
        self._current_action_mask: np.ndarray | None = None
        self._stop_to_depot_time_s: np.ndarray | None = None
        self._static_obs_cache: dict[str, np.ndarray] | None = None
        super().__init__(*args, **kwargs)

    def set_instance(self, instance):  # noqa: ANN001 - match base signature
        super().set_instance(instance)
        self._precompute_stop_return_times()
        self._build_static_observation_cache()
        self._current_action_mask = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._current_action_mask = np.asarray(obs["action_mask"], dtype=bool).copy()
        return obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        if self._current_action_mask is None:
            mask_before = self._compute_action_mask()
        else:
            mask_before = self._current_action_mask
        reward = np.zeros(self.n_traj, dtype=np.float32)
        self.invalid_action.fill(False)

        for t in range(self.n_traj):
            if self.terminated[t] or self.truncated[t]:
                continue
            destination = int(action_arr[t])
            if destination < 0 or destination >= self.num_nodes or not mask_before[t, destination]:
                self.invalid_action[t] = True
                self.truncated[t] = True
                reward[t] += self.invalid_action_penalty
                continue
            reward[t] += self._apply_action(t, destination)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            unfinished = ~self.terminated
            self.truncated[unfinished] = True

        obs = self._make_observation()
        action_mask = obs["action_mask"]
        no_action = (~action_mask.any(axis=1)) & (~self.terminated) & (~self.truncated)
        if np.any(no_action):
            self.truncated[no_action] = True
            reward[no_action] += self.invalid_action_penalty
            obs = self._make_observation()
            action_mask = obs["action_mask"]

        self._current_action_mask = np.asarray(action_mask, dtype=bool).copy()
        info = self._make_info(action_mask)
        return obs, reward, self.terminated.copy(), self.truncated.copy(), info

    def _precompute_stop_return_times(self) -> None:
        out = np.full(self.num_nodes, math.inf, dtype=np.float64)
        for node in self.stop_nodes:
            if int(node) == 0:
                out[int(node)] = 0.0
                continue
            value = super()._shortest_stop_time(int(node), 0)
            if value is not None:
                out[int(node)] = float(value)
        self._stop_to_depot_time_s = out

    def _compute_action_mask(self) -> np.ndarray:
        if not self.use_jit_mask or self._stop_to_depot_time_s is None:
            return super()._compute_action_mask()
        return compute_action_mask_jit(
            n_traj=self.n_traj,
            num_nodes=self.num_nodes,
            num_customers=self.num_customers,
            station_start=self.station_start,
            last=self.last,
            visited=self.visited,
            cs_visited_current_route=self.cs_visited_current_route,
            terminated=self.terminated,
            truncated=self.truncated,
            served_customers=self.served_customers,
            route_has_customer=self.route_has_customer,
            current_time_s=self.current_time_s,
            battery_used_kwh=self.battery_used_kwh,
            load_cm3=self.load_cm3,
            demand_cm3=self.demand_cm3,
            service_time_s=self.service_time_s,
            tw_s=self.tw_s,
            travel_time_s=self.travel_time_s,
            energy_kwh=self.energy_kwh,
            stop_to_depot_time_s=self._stop_to_depot_time_s,
            battery_capacity_kwh=self.battery_capacity_kwh,
            cargo_capacity_cm3=self.cargo_capacity_cm3,
            full_charge_time_s=self.full_charge_time_s,
            working_end_s=self.working_end_s,
            fixed_full_charge=self.charging_mode == "fixed_full",
        )

    def _build_static_observation_cache(self) -> None:
        coords = self._normalized_coords().astype(np.float32)
        demand_norm = (self.demand_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        tw_norm = ((self.tw_s - self.working_start_s) / self.horizon_s).astype(np.float32)
        service_norm = (self.service_time_s / self.horizon_s).astype(np.float32)
        edge_distance = (self.distance_km / max(self.reward_distance_scale_km, 1e-12)).astype(np.float32)
        edge_time = (self.travel_time_s / max(self.horizon_s, 1e-12)).astype(np.float32)
        edge_energy = (self.energy_kwh / max(self.battery_capacity_kwh, 1e-12)).astype(np.float32)
        self._static_obs_cache = {
            "cus_loc": coords[self.customer_start:self.station_start],
            "depot_loc": coords[0:1],
            "rs_loc": coords[self.station_start:],
            "edge_distance": edge_distance,
            "edge_time": edge_time,
            "edge_energy": edge_energy,
            "demand": demand_norm,
            "time_window": tw_norm,
            "service_time": service_norm,
            "battery_capacity": np.array([1.0], dtype=np.float32),
            "loading_capacity": np.array([1.0], dtype=np.float32),
        }

    def _can_return_to_depot(self, start: int, current_time_s: float, battery_used_kwh: float, traj_idx: int | None = None) -> bool:
        if start == 0:
            return True
        if battery_used_kwh + self.energy_kwh[start, 0] <= self.battery_capacity_kwh + 1e-9:
            return current_time_s + self.travel_time_s[start, 0] <= self.working_end_s + 1e-9
        stop_to_depot = self._stop_to_depot_time_s
        for first_station in self.station_nodes:
            first = int(first_station)
            if first == int(start):
                continue
            if traj_idx is not None and self.cs_visited_current_route[int(traj_idx), first]:
                continue
            battery_at_first = battery_used_kwh + self.energy_kwh[start, first]
            if battery_at_first > self.battery_capacity_kwh + 1e-9:
                continue
            time_at_first = current_time_s + self.travel_time_s[start, first]
            depart_first = time_at_first + self._charge_time_s(battery_at_first)
            if stop_to_depot is None:
                stop_plan = super()._shortest_stop_time(first, 0)
                if stop_plan is None:
                    continue
            else:
                stop_plan = float(stop_to_depot[first])
                if not np.isfinite(stop_plan):
                    continue
            if depart_first + stop_plan <= self.working_end_s + 1e-9:
                return True
        return False

    def _make_observation(self) -> dict[str, np.ndarray]:
        action_mask = self._compute_action_mask()
        self._current_action_mask = np.asarray(action_mask, dtype=bool).copy()
        feasible_customer_count = action_mask[:, self.customer_start:self.station_start].sum(axis=1, keepdims=True)
        visited_ratio = (self.served_customers.astype(np.float32) / max(float(self.num_customers), 1.0))[:, None]
        remain_feasible_ratio = feasible_customer_count.astype(np.float32) / max(float(self.num_customers), 1.0)
        route_customer_ratio = (self.current_route_customer_count.astype(np.float32) / max(float(self.num_customers), 1.0))[:, None]
        route_step_ratio = (self.current_route_step_count.astype(np.float32) / max(float(self.max_steps), 1.0))[:, None]
        rs_streak_ratio = (self.rs_streak_count.astype(np.float32) / max(float(self.num_stations), 1.0))[:, None]

        static = self._static_obs_cache
        if static is None:
            self._build_static_observation_cache()
            static = self._static_obs_cache
        assert static is not None
        current_battery = (self.battery_used_kwh / max(self.battery_capacity_kwh, 1e-12)).astype(np.float32)
        remaining = (1.0 - current_battery).astype(np.float32)
        current_load = (self.load_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        current_time = ((self.current_time_s - self.working_start_s) / self.horizon_s).astype(np.float32)

        return {
            "cus_loc": static["cus_loc"],
            "depot_loc": static["depot_loc"],
            "rs_loc": static["rs_loc"],
            "edge_distance": static["edge_distance"],
            "edge_time": static["edge_time"],
            "edge_energy": static["edge_energy"],
            "demand": static["demand"],
            "time_window": static["time_window"],
            "service_time": static["service_time"],
            "action_mask": action_mask,
            "last_node_idx": self.last.copy(),
            "prev_node_idx": self.prev.copy(),
            "node_visit_count": self.node_visit_count.copy(),
            "customer_visited": self.visited.copy(),
            "cs_visited_current_route": self.cs_visited_current_route.copy(),
            "route_membership_current": self.route_membership_current.copy(),
            "route_order_rank": self.route_order_rank.copy(),
            "current_load": current_load,
            "current_battery": current_battery,
            "remaining_battery": remaining,
            "current_time": current_time,
            "battery_capacity": static["battery_capacity"],
            "loading_capacity": static["loading_capacity"],
            "visited_customers_ratio": visited_ratio,
            "visited_customers_raio": visited_ratio,
            "remain_feasible_customers_ratio": remain_feasible_ratio,
            "remain_feasible_customers_raio": remain_feasible_ratio,
            "route_served_customers_ratio": route_customer_ratio,
            "rs_streak_ratio": rs_streak_ratio,
            "current_route_step_count": route_step_ratio,
            "current_route_customer_count": route_customer_ratio,
        }

    def _make_info(self, action_mask: np.ndarray) -> dict[str, Any]:
        success = self.terminated & (self.served_customers == self.num_customers) & (self.last == 0)
        info: dict[str, Any] = {
            "action_mask": action_mask.copy(),
            "objective_distance_km": self.objective_distance_km.copy(),
            "vehicle_count": self.vehicle_count.copy(),
            "success": success.copy(),
            "served_customers": self.served_customers.copy(),
            "invalid_action": self.invalid_action.copy(),
        }
        if self.info_level == "full":
            routes = self.get_routes()
            info["routes"] = routes
            from evrptw_core.schema import merge_route_sequences

            info["route_sequence"] = [merge_route_sequences(route_set) for route_set in routes]
        return info


__all__ = ["EVRPTWVectorEnvFast", "Transition"]
