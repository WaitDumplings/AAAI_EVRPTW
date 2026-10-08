"""Physical input units and portable, fixed-width road/resource context.

Version 1 uses kilometres, seconds, kWh and cm3 before encoding. Distance has
one explicit frozen kilometre unit; time and energy use the actual horizon and
battery capacity. Contexts describe observations only, never rewards/dynamics.
Positive continuous context features use log1p; flags/fractions are unchanged.
"""
from __future__ import annotations

import math
import numpy as np

INPUT_CONTEXT_SCHEMA = "physical_input_context_v1"
INPUT_CONTEXT_BASE_UNITS = {"time_s": 3600.0, "cargo_cm3": 1_000_000.0, "battery_kwh": 100.0}
NODE_INPUT_CONTEXT_FEATURES = (
    "log1p_depot_to_node_distance", "log1p_node_to_depot_distance",
    "log1p_mean_outgoing_distance", "log1p_mean_incoming_distance",
    "log1p_mean_outgoing_time", "log1p_mean_incoming_time",
    "log1p_mean_outgoing_energy", "log1p_mean_incoming_energy",
    "depot_to_node_reachable", "node_to_depot_reachable",
    "outgoing_reachable_fraction", "incoming_reachable_fraction",
)
GRAPH_INPUT_CONTEXT_FEATURES = (
    "log1p_horizon_hours", "log1p_cargo_capacity_m3", "log1p_battery_capacity_100kwh",
    "log1p_speed_horizon_per_distance_unit", "log1p_energy_per_distance_unit_over_battery",
    "full_charge_time_over_horizon", "battery_active", "capacity_active",
    "time_window_active", "fixed_full_charge",
)
NODE_INPUT_CONTEXT_DIM = len(NODE_INPUT_CONTEXT_FEATURES)
GRAPH_INPUT_CONTEXT_DIM = len(GRAPH_INPUT_CONTEXT_FEATURES)


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def input_normalization_signature(cfg):
    """Serializable observation contract for checkpoints/config comparisons."""
    env = cfg.get("env", {})
    mode = str(env.get("observation_coordinate_mode", "legacy_minmax"))
    context = bool(env.get("observation_input_context", False))
    if mode not in {"legacy_minmax", "depot_fixed"}:
        raise ValueError("observation_coordinate_mode must be legacy_minmax or depot_fixed")
    unit = env.get("observation_distance_scale_km")
    if unit is None and (context or mode == "depot_fixed"):
        raise ValueError("Physical input normalization requires explicit observation_distance_scale_km")
    if unit is not None:
        unit = _positive(unit, "observation_distance_scale_km")
    switches = {"strict_road_metric": cfg.get("data", {}).get("strict_road_metric", False),
                "prefer_explicit_edge_matrices": env.get("prefer_explicit_edge_matrices", False)}
    if any(not isinstance(value, bool) for value in switches.values()):
        raise ValueError("strict_road_metric and prefer_explicit_edge_matrices must be booleans")
    return {
        **switches,
        "observation_coordinate_mode": mode,
        "observation_distance_scale_km": unit,
        "observation_input_context": context,
        "context_schema": INPUT_CONTEXT_SCHEMA if context else None,
        "context_base_units": dict(INPUT_CONTEXT_BASE_UNITS) if context else None,
        "node_context_features": list(NODE_INPUT_CONTEXT_FEATURES) if context else [],
        "graph_context_features": list(GRAPH_INPUT_CONTEXT_FEATURES) if context else [],
    }


def build_input_context(*, distance_km, travel_time_s, energy_kwh,
                        distance_scale_km, horizon_s, cargo_capacity_cm3,
                        battery_capacity_kwh, speed_kmh, energy_per_km,
                        full_charge_time_s, charging_mode, metadata=None):
    """Encode finite directed-edge summaries and explicit physical resources.

    Reachability means a finite nonnegative road transition, not current vehicle
    feasibility. Zero-length edges and zero energy remain legal. Matrix summary
    means exclude self edges and unreachable entries. Inactive physical resource
    features are zero and have a separate activity flag, avoiding dummy values.
    Global speed/consumption are nominal resource descriptors; explicit edge T/E
    matrices remain authoritative when road-specific costs are provided.
    """
    metadata = metadata or {}
    length = _positive(distance_scale_km, "observation_distance_scale_km")
    horizon = _positive(horizon_s, "horizon_s")
    battery, cargo = float(battery_capacity_kwh), float(cargo_capacity_cm3)
    consumption = float(energy_per_km)
    energy = np.asarray(energy_kwh, dtype=np.float64)
    # Explicit heterogeneous energy may be meaningful even if the nominal
    # vehicle consumption is zero. The task activity flag remains authoritative.
    energy_active = consumption > 0 or bool(np.any(np.isfinite(energy) & (energy > 0)))
    battery_active = (bool(metadata.get("charging_constraint", True))
                      and math.isfinite(battery) and battery > 0 and energy_active)
    capacity_active = (bool(metadata.get("capacity_constraint", True))
                       and math.isfinite(cargo) and cargo > 0)
    time_active = bool(metadata.get("time_window_constraint", True))
    distance = np.asarray(distance_km, dtype=np.float64)
    travel = np.asarray(travel_time_s, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or distance.shape[0] < 1:
        raise ValueError("distance_km must be a nonempty square matrix")
    if travel.shape != distance.shape or energy.shape != distance.shape:
        raise ValueError("distance/time/energy matrices must have identical shapes")
    reachable = np.isfinite(distance) & (distance >= 0)
    if time_active:
        reachable &= np.isfinite(travel) & (travel >= 0)
    if battery_active:
        reachable &= np.isfinite(energy) & (energy >= 0)
    valid = reachable & ~np.eye(len(distance), dtype=bool)
    outgoing = valid.sum(axis=1)
    incoming = valid.sum(axis=0)

    def means(values):
        finite = np.where(valid, values, 0.0)
        return (finite.sum(axis=1) / np.maximum(outgoing, 1),
                finite.sum(axis=0) / np.maximum(incoming, 1))

    scaled_distance = np.where(reachable, distance / length, 0.0)
    scaled_time = np.where(reachable, travel / horizon, 0.0) if time_active else np.zeros_like(distance)
    scaled_energy = np.where(reachable, energy / battery, 0.0) if battery_active else np.zeros_like(distance)
    node = np.stack((scaled_distance[0], scaled_distance[:, 0],
                     *means(scaled_distance), *means(scaled_time), *means(scaled_energy),
                     reachable[0], reachable[:, 0],
                     outgoing / max(len(distance) - 1, 1),
                     incoming / max(len(distance) - 1, 1)), axis=-1)
    node[:, :8] = np.log1p(node[:, :8])
    speed = _positive(speed_kmh, "speed_kmh") if time_active else 0.0
    charge_time = float(full_charge_time_s) if battery_active else 0.0
    if battery_active and (not math.isfinite(consumption) or not math.isfinite(charge_time) or charge_time < 0):
        raise ValueError("Active energy and charging parameters must be finite and nonnegative")
    graph = np.asarray((
        horizon / INPUT_CONTEXT_BASE_UNITS["time_s"] if time_active else 0.0,
        cargo / INPUT_CONTEXT_BASE_UNITS["cargo_cm3"] if capacity_active else 0.0,
        battery / INPUT_CONTEXT_BASE_UNITS["battery_kwh"] if battery_active else 0.0,
        speed / 3600.0 * horizon / length if time_active else 0.0,
        consumption * length / battery if battery_active else 0.0,
        charge_time / horizon,
        float(battery_active), float(capacity_active), float(time_active),
        float(battery_active and charging_mode == "fixed_full"),
    ), dtype=np.float64)
    graph[:5] = np.log1p(graph[:5])
    node, graph = node.astype(np.float32), graph.astype(np.float32)
    if not np.isfinite(node).all() or not np.isfinite(graph).all():
        raise ValueError("Physical input context is not finite in float32")
    return {"node_input_context": node, "graph_input_context": graph}
