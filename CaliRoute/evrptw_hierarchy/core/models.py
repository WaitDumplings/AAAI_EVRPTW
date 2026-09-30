from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VehicleConfig:
    name: str
    design_speed_kmh: float
    battery_capacity_kwh: float
    consumption_kwh_per_km: float
    cargo_capacity_cm3: float
    charging_power_kw: float

    @property
    def battery_range_km(self) -> float:
        return self.battery_capacity_kwh / max(self.consumption_kwh_per_km, 1e-12)

    @property
    def full_charge_time_min(self) -> float:
        return self.battery_capacity_kwh / max(self.charging_power_kw, 1e-12) * 60.0


@dataclass
class RegionBoard:
    region_id: str
    mother_board_id: str
    region_profile: str
    area_size_km: list[list[float]]
    depot_node_id: int
    road_nodes: np.ndarray
    road_edges: np.ndarray
    road_edge_lengths_km: np.ndarray
    customers: np.ndarray
    customer_node_ids: np.ndarray
    charging_stations: np.ndarray
    cs_node_ids: np.ndarray
    cluster_centers: np.ndarray
    cluster_gateway_node_ids: np.ndarray
    cluster_labels: np.ndarray
    micro_zone_labels: np.ndarray
    customer_candidate_cs_ids: list[np.ndarray]
    cluster_candidate_cs_ids: list[np.ndarray]
    region_validation: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    depot_candidates: np.ndarray | None = None
    depot_candidate_node_ids: np.ndarray | None = None
    depot_candidate_metadata: list[dict[str, Any]] = field(default_factory=list)

    def to_pickle_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "mother_board_id": self.mother_board_id,
            "region_profile": self.region_profile,
            "area_size_km": self.area_size_km,
            "depot_node_id": self.depot_node_id,
            "road_nodes": self.road_nodes,
            "road_edges": self.road_edges,
            "road_edge_lengths_km": self.road_edge_lengths_km,
            "customers": self.customers,
            "customer_node_ids": self.customer_node_ids,
            "charging_stations": self.charging_stations,
            "cs_node_ids": self.cs_node_ids,
            "cluster_centers": self.cluster_centers,
            "cluster_gateway_node_ids": self.cluster_gateway_node_ids,
            "cluster_labels": self.cluster_labels,
            "micro_zone_labels": self.micro_zone_labels,
            "customer_candidate_cs_ids": self.customer_candidate_cs_ids,
            "cluster_candidate_cs_ids": self.cluster_candidate_cs_ids,
            "region_validation": self.region_validation,
            "metadata": self.metadata,
            "depot_candidates": self.depot_candidates,
            "depot_candidate_node_ids": self.depot_candidate_node_ids,
            "depot_candidate_metadata": self.depot_candidate_metadata,
        }


@dataclass
class RegionUsage:
    region_id: str
    sampled_days: int
    customer_activation_counts: np.ndarray
    cluster_activation_counts: np.ndarray
    recent_active_customer_sets: list[set[int]] = field(default_factory=list)

    def record_day(self, active_customer_ids: np.ndarray, cluster_labels: np.ndarray, recent_window: int) -> None:
        ids = np.asarray(active_customer_ids, dtype=int)
        self.sampled_days += 1
        self.customer_activation_counts[ids] += 1
        clusters, counts = np.unique(cluster_labels[ids], return_counts=True)
        self.cluster_activation_counts[clusters.astype(int)] += counts.astype(int)
        self.recent_active_customer_sets.append(set(int(x) for x in ids.tolist()))
        if len(self.recent_active_customer_sets) > recent_window:
            self.recent_active_customer_sets = self.recent_active_customer_sets[-recent_window:]

    @property
    def customer_exposure_rate(self) -> float:
        return float(np.mean(self.customer_activation_counts > 0))

    @property
    def cluster_exposure_entropy(self) -> float:
        counts = self.cluster_activation_counts.astype(float)
        total = counts.sum()
        if total <= 0 or counts.size <= 1:
            return 0.0
        p = counts[counts > 0] / total
        return float(-(p * np.log(p)).sum() / np.log(counts.size))

    @property
    def recent_mean_jaccard_distance(self) -> float:
        sets = self.recent_active_customer_sets
        if len(sets) < 2:
            return 1.0
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = len(sets[i] | sets[j])
                inter = len(sets[i] & sets[j])
                vals.append(1.0 - inter / max(union, 1))
        return float(np.mean(vals)) if vals else 1.0


@dataclass
class ActiveInstance:
    instance_id: str
    region_id: str
    mother_board_id: str
    operating_day_id: str
    day_type: str
    working_start_s: int
    working_end_s: int
    active_customer_ids: np.ndarray
    active_cs_ids: np.ndarray
    depot: np.ndarray
    customers: np.ndarray
    charging_stations: np.ndarray
    distance_matrix_km: np.ndarray
    raw_travel_time_matrix_s: np.ndarray | None
    ev_transition_time_matrix_s: np.ndarray | None
    shortest_time_matrix_s: np.ndarray | None
    cs_time_to_depot_s: np.ndarray
    demands_cm3: np.ndarray
    package_counts: np.ndarray
    service_time_s: np.ndarray
    tw_s: np.ndarray
    vehicle: dict[str, Any]
    speed_profile: dict[str, Any]
    cs_activation: dict[str, Any]
    greedy_audit: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pickle_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "region_id": self.region_id,
            "mother_board_id": self.mother_board_id,
            "operating_day_id": self.operating_day_id,
            "day_type": self.day_type,
            "working_start_s": self.working_start_s,
            "working_end_s": self.working_end_s,
            "active_customer_ids": self.active_customer_ids,
            "active_cs_ids": self.active_cs_ids,
            "depot": self.depot,
            "customers": self.customers,
            "charging_stations": self.charging_stations,
            "distance_matrix_km": self.distance_matrix_km,
            "raw_travel_time_matrix_s": self.raw_travel_time_matrix_s,
            "ev_transition_time_matrix_s": self.ev_transition_time_matrix_s,
            "shortest_time_matrix_s": self.shortest_time_matrix_s,
            "cs_time_to_depot_s": self.cs_time_to_depot_s,
            "demands_cm3": self.demands_cm3,
            "package_counts": self.package_counts,
            "service_time_s": self.service_time_s,
            "tw_s": self.tw_s,
            "vehicle": self.vehicle,
            "speed_profile": self.speed_profile,
            "cs_activation": self.cs_activation,
            "greedy_audit": self.greedy_audit,
            "metadata": self.metadata,
        }
