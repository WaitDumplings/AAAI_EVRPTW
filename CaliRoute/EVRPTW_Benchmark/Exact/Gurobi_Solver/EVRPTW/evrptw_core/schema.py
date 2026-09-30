from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EVRPTWInstance:
    """Canonical daily-instance schema used by both Dataset and Benchmark.

    The generator stores instances as pickle dictionaries. ``from_dict`` keeps the
    benchmark layer independent from generator internals while preserving the
    current field names.
    """

    instance_id: str
    region_id: str
    mother_board_id: str
    operating_day_id: str
    day_type: str
    working_start_s: int
    working_end_s: int
    depot: np.ndarray
    customers: np.ndarray
    charging_stations: np.ndarray
    distance_matrix_km: np.ndarray
    demands_cm3: np.ndarray
    package_counts: np.ndarray
    service_time_s: np.ndarray
    tw_s: np.ndarray
    cs_time_to_depot_s: np.ndarray
    vehicle: dict[str, Any]
    speed_profile: dict[str, Any] = field(default_factory=dict)
    cs_activation: dict[str, Any] = field(default_factory=dict)
    greedy_audit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EVRPTWInstance":
        return cls(
            instance_id=str(data["instance_id"]),
            region_id=str(data.get("region_id", "")),
            mother_board_id=str(data.get("mother_board_id", "")),
            operating_day_id=str(data.get("operating_day_id", "")),
            day_type=str(data.get("day_type", "")),
            working_start_s=int(data["working_start_s"]),
            working_end_s=int(data["working_end_s"]),
            depot=np.asarray(data["depot"], dtype=np.float32),
            customers=np.asarray(data["customers"], dtype=np.float32),
            charging_stations=np.asarray(data["charging_stations"], dtype=np.float32),
            distance_matrix_km=np.asarray(data["distance_matrix_km"], dtype=np.float32),
            demands_cm3=np.asarray(data["demands_cm3"], dtype=np.float32),
            package_counts=np.asarray(data["package_counts"], dtype=np.int32),
            service_time_s=np.asarray(data["service_time_s"], dtype=np.float32),
            tw_s=np.asarray(data["tw_s"], dtype=np.float32),
            cs_time_to_depot_s=np.asarray(data["cs_time_to_depot_s"], dtype=np.float32),
            vehicle=dict(data.get("vehicle", {})),
            speed_profile=dict(data.get("speed_profile", {})),
            cs_activation=dict(data.get("cs_activation", {})),
            greedy_audit=dict(data.get("greedy_audit", {})),
            metadata=dict(data.get("metadata", {})),
            raw=data,
        )

    @property
    def num_customers(self) -> int:
        return int(self.customers.shape[0])

    @property
    def num_charging_stations(self) -> int:
        return int(self.charging_stations.shape[0])

    @property
    def num_terminals(self) -> int:
        return 1 + self.num_customers + self.num_charging_stations


def merge_route_sequences(routes: list[list[int]]) -> list[int]:
    """Merge per-vehicle routes into one depot-separated sequence.

    Route convention: each route usually starts and ends with depot node 0.
    When concatenating routes, the end depot of the previous route is also the
    separator before the next route, so the next route's leading depot is
    removed. Example:

    [[0, 3, 2, 1, 0], [0, 7, 5, 0]] -> [0, 3, 2, 1, 0, 7, 5, 0]
    """
    merged: list[int] = []
    for route in routes:
        if not route:
            continue
        clean_route = [int(node) for node in route]
        if not merged:
            merged.extend(clean_route)
        elif clean_route[0] == 0 and merged[-1] == 0:
            merged.extend(clean_route[1:])
        else:
            merged.extend(clean_route)
    return merged


def solution_route_sequence(solution: "EVRPTWSolution") -> list[int]:
    return merge_route_sequences(solution.routes)


@dataclass
class EVRPTWSolution:
    """Solver output schema.

    Route node convention for benchmark solvers:
    - depot is node 0;
    - customers are 1..num_customers;
    - charging stations are num_customers+1 .. num_terminals-1.
    """

    instance_id: str
    solver_name: str
    routes: list[list[int]]
    objective_distance_km: float | None = None
    vehicle_count: int | None = None
    runtime_s: float | None = None
    feasible: bool | None = None
    violations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "solver_name": self.solver_name,
            "routes": self.routes,
            "objective_distance_km": self.objective_distance_km,
            "vehicle_count": self.vehicle_count,
            "runtime_s": self.runtime_s,
            "feasible": self.feasible,
            "violations": self.violations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EVRPTWSolution":
        return cls(
            instance_id=str(data["instance_id"]),
            solver_name=str(data.get("solver_name", "unknown")),
            routes=[list(map(int, route)) for route in data.get("routes", [])],
            objective_distance_km=data.get("objective_distance_km"),
            vehicle_count=data.get("vehicle_count"),
            runtime_s=data.get("runtime_s"),
            feasible=data.get("feasible"),
            violations=dict(data.get("violations", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
