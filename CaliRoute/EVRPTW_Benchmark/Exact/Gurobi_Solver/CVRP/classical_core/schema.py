from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassicalVRPInstance:
    problem_class: str
    instance_id: str
    region_id: str
    num_customers: int
    distance_matrix_km: np.ndarray
    demands_cm3: np.ndarray
    vehicle: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassicalVRPInstance":
        demands = np.asarray(data["demands_cm3"], dtype=float)
        return cls(
            problem_class=str(data.get("problem_class", "CVRP")),
            instance_id=str(data["instance_id"]),
            region_id=str(data.get("region_id", "")),
            num_customers=int(demands.shape[0]),
            distance_matrix_km=np.asarray(data["distance_matrix_km"], dtype=float),
            demands_cm3=demands,
            vehicle=dict(data.get("vehicle", {})),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ClassicalVRPSolution:
    instance_id: str
    solver_name: str
    routes: list[list[int]]
    objective_distance_km: float | None
    vehicle_count: int | None
    runtime_s: float | None
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def solution_route_sequence(routes: list[list[int]]) -> list[int]:
    seq: list[int] = []
    for route in routes:
        if not route:
            continue
        if not seq:
            seq.extend(int(x) for x in route)
        else:
            if seq[-1] == 0 and int(route[0]) == 0:
                seq.extend(int(x) for x in route[1:])
            else:
                seq.extend(int(x) for x in route)
    return seq
