from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evrptw_hierarchy.core.models import VehicleConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def vehicle_from_config(config: dict[str, Any]) -> VehicleConfig:
    vehicle = config.get("vehicle", {})
    charging = config.get("charging", {})
    return VehicleConfig(
        name=str(vehicle.get("name", "Rivian_EDV_700")),
        design_speed_kmh=float(vehicle.get("design_speed_kmh", 40.0)),
        battery_capacity_kwh=float(vehicle.get("battery_capacity_kwh", 100.0)),
        consumption_kwh_per_km=float(vehicle.get("consumption_kwh_per_km", 0.404)),
        cargo_capacity_cm3=float(vehicle.get("cargo_capacity_cm3", 3607178.0)),
        charging_power_kw=float(charging.get("power_kw", 150.0)),
    )


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out
