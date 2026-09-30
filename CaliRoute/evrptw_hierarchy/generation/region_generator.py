from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from evrptw_hierarchy.core.models import RegionBoard, VehicleConfig
from evrptw_hierarchy.graph.road_graph import (
    add_unique_edge,
    connect_knn,
    euclidean_edges,
    nearest_node,
    sample_dirichlet_counts,
    sample_truncated_lognormal,
)
from evrptw_hierarchy.graph.shortest_path import build_adjacency, dijkstra_one


@dataclass
class RegionGenerator:
    config: dict[str, Any]
    vehicle: VehicleConfig
    rng: np.random.Generator

    def _auto_cluster_count(self, n_customers: int) -> int:
        cfg = self.config.get("region", {}).get("cluster_count", {})
        if "value" in cfg and cfg["value"] not in (None, "auto"):
            return int(cfg["value"])
        ref_n = float(cfg.get("reference_customers", 1800))
        ref_k = float(cfg.get("reference_clusters", 18))
        exponent = float(cfg.get("scale_exponent", 0.8))
        raw = math.ceil(ref_k * (max(n_customers, 1) / ref_n) ** exponent)
        return int(np.clip(raw, int(cfg.get("min", 2)), int(cfg.get("max", 100))))

    def _sample_depot(self, area: list[list[float]]) -> np.ndarray:
        cfg = self.config.get("region", {}).get("depot_sampling", {})
        (xmin, xmax), (ymin, ymax) = area
        fraction = float(cfg.get("central_fraction", 0.48))
        cx0 = xmin + (1.0 - fraction) * 0.5 * (xmax - xmin)
        cx1 = xmax - (1.0 - fraction) * 0.5 * (xmax - xmin)
        cy0 = ymin + (1.0 - fraction) * 0.5 * (ymax - ymin)
        cy1 = ymax - (1.0 - fraction) * 0.5 * (ymax - ymin)
        return np.asarray([self.rng.uniform(cx0, cx1), self.rng.uniform(cy0, cy1)], dtype=np.float32)

    def _sample_cluster_centers(self, depot: np.ndarray, k: int, area: list[list[float]]) -> np.ndarray:
        cfg = self.config.get("region", {}).get("cluster_geometry", {})
        (xmin, xmax), (ymin, ymax) = area
        min_r = float(cfg.get("radius_min_km", 4.0))
        max_r = float(cfg.get("radius_max_km", 34.0))
        model = str(cfg.get("angular_model", "corridor_mixture"))
        centers: list[np.ndarray] = []

        def sample_radius(far_bias: bool = False, near_bias: bool = False) -> float:
            if str(cfg.get("radius_distribution", "lognormal")) == "lognormal":
                median = float(cfg.get("radius_median_km", 16.8))
                sigma = float(cfg.get("radius_lognormal_sigma", 0.48))
                value = float(self.rng.lognormal(math.log(max(median, 1e-6)), sigma))
            else:
                value = float(self.rng.normal(float(cfg.get("radius_mean_km", 16.2)), float(cfg.get("radius_std_km", 5.0))))
            if far_bias:
                value = max(value, float(self.rng.uniform(float(cfg.get("far_radius_min_km", 22.0)), max_r)))
            if near_bias:
                value = min(value, float(self.rng.uniform(min_r, float(cfg.get("near_radius_max_km", 11.0)))))
            return float(np.clip(value, min_r, max_r))

        if model == "corridor_mixture":
            corridor_min = int(cfg.get("corridor_count_min", 2))
            corridor_max = int(cfg.get("corridor_count_max", 4))
            corridor_count = int(self.rng.integers(corridor_min, corridor_max + 1))
            fan = math.radians(float(cfg.get("service_sector_angle_deg", 235.0)))
            base = float(self.rng.uniform(0.0, 2.0 * math.pi))
            if corridor_count == 1:
                corridor_angles = np.asarray([base])
            else:
                offsets = np.linspace(-0.5 * fan, 0.5 * fan, corridor_count)
                offsets += self.rng.normal(0.0, math.radians(float(cfg.get("corridor_angle_jitter_deg", 10.0))), size=corridor_count)
                corridor_angles = base + offsets
            corridor_weights = self.rng.dirichlet(np.full(corridor_count, float(cfg.get("corridor_weight_alpha", 1.4))))
            angular_sigma = math.radians(float(cfg.get("corridor_angular_sigma_deg", 13.0)))
            infill_probability = float(cfg.get("infill_probability", 0.28))
            min_center_spacing = float(cfg.get("min_center_spacing_km", 1.2))
        else:
            corridor_angles = self.rng.uniform(0.0, 2.0 * math.pi, size=max(k, 1))
            corridor_weights = np.ones(len(corridor_angles)) / len(corridor_angles)
            angular_sigma = 0.18
            infill_probability = 1.0
            fan = 2.0 * math.pi
            base = 0.0
            min_center_spacing = float(cfg.get("min_center_spacing_km", 1.2))

        for idx in range(k):
            point = depot.copy()
            for attempt in range(300):
                near_bias = bool(k >= 8 and idx % 7 == 0)
                far_bias = bool(k >= 8 and idx % 9 == 0)
                r = sample_radius(far_bias=far_bias, near_bias=near_bias)
                if self.rng.random() < infill_probability:
                    angle = base + self.rng.uniform(-0.5 * fan, 0.5 * fan)
                    if near_bias:
                        r = sample_radius(near_bias=True)
                else:
                    cidx = int(self.rng.choice(len(corridor_angles), p=corridor_weights))
                    angle = float(corridor_angles[cidx] + self.rng.normal(0.0, angular_sigma))
                point = depot + r * np.asarray([math.cos(angle), math.sin(angle)])
                in_bounds = xmin + 2.0 <= point[0] <= xmax - 2.0 and ymin + 2.0 <= point[1] <= ymax - 2.0
                if not in_bounds:
                    continue
                if centers and attempt < 220:
                    min_dist = min(float(np.linalg.norm(point - prev)) for prev in centers)
                    if min_dist < min_center_spacing:
                        continue
                centers.append(point.astype(np.float32))
                break
            else:
                centers.append(np.asarray([
                    np.clip(point[0], xmin + 2.0, xmax - 2.0),
                    np.clip(point[1], ymin + 2.0, ymax - 2.0),
                ], dtype=np.float32))
        return np.vstack(centers).astype(np.float32)

    def _new_node(self, nodes: list[np.ndarray], point: np.ndarray) -> int:
        nodes.append(np.asarray(point, dtype=np.float32))
        return len(nodes) - 1

    def _build_region_graph(
        self,
        area: list[list[float]],
        depot: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_counts: np.ndarray,
        n_cs: int,
    ) -> dict[str, Any]:
        cfg = self.config.get("region", {})
        graph_cfg = cfg.get("road_graph", {})
        customer_cfg = cfg.get("customers", {})
        cs_cfg = cfg.get("charging_stations", {})

        nodes: list[np.ndarray] = []
        depot_id = self._new_node(nodes, depot)
        gateway_ids = [self._new_node(nodes, point) for point in cluster_centers]

        edge_set: set[tuple[int, int]] = set()
        all_core_points = np.vstack([depot.reshape(1, 2), cluster_centers])
        core_ids = [depot_id] + gateway_ids

        # Backbone: connect each gateway to depot or a nearer previous gateway.
        for local_idx, gateway_id in enumerate(gateway_ids, start=1):
            dist = np.linalg.norm(all_core_points[:local_idx] - all_core_points[local_idx], axis=1)
            nearest_prev = core_ids[int(np.argmin(dist))]
            add_unique_edge(edge_set, gateway_id, nearest_prev)

        connect_knn(np.asarray(nodes), core_ids, int(graph_cfg.get("gateway_knn", 3)), edge_set, float(graph_cfg.get("gateway_max_edge_km", 18.0)))

        # Add arterial junction nodes on depot/gateway corridors.
        arterial_junction_ids = []
        for gateway_id in gateway_ids:
            gateway = np.asarray(nodes[gateway_id])
            t = float(self.rng.uniform(0.32, 0.68))
            lateral = self.rng.normal(0.0, float(graph_cfg.get("arterial_lateral_sigma_km", 1.2)), size=2)
            base = (1.0 - t) * depot + t * gateway + lateral
            jid = self._new_node(nodes, base)
            arterial_junction_ids.append(jid)
            add_unique_edge(edge_set, depot_id, jid)
            add_unique_edge(edge_set, jid, gateway_id)

        connect_knn(np.asarray(nodes), arterial_junction_ids, int(graph_cfg.get("junction_knn", 2)), edge_set, float(graph_cfg.get("junction_max_edge_km", 14.0)))

        customer_points: list[np.ndarray] = []
        customer_node_ids: list[int] = []
        cluster_labels: list[int] = []
        micro_zone_labels: list[int] = []
        local_street_ids: list[int] = []
        micro_zone_id = 0

        for cluster_id, (gateway_id, count) in enumerate(zip(gateway_ids, cluster_counts)):
            count_i = int(count)
            gateway = np.asarray(nodes[gateway_id])
            n_street = max(3, int(math.ceil(math.sqrt(count_i) * float(graph_cfg.get("local_street_factor", 0.9)))))
            radial = cluster_centers[cluster_id] - depot
            theta = math.atan2(float(radial[1]), float(radial[0])) + math.pi / 2.0
            axis = np.asarray([math.cos(theta), math.sin(theta)])
            perp = np.asarray([-axis[1], axis[0]])
            scale = float(customer_cfg.get("cluster_spread_km", 1.2)) * max((count_i / 135.0) ** 0.5, 0.35)
            street_ids = []
            for s_idx in range(n_street):
                offset = self.rng.normal(0.0, scale * 0.55) * axis + self.rng.normal(0.0, scale * 0.35) * perp
                point = gateway + offset
                nid = self._new_node(nodes, point)
                street_ids.append(nid)
                local_street_ids.append(nid)
            connect_knn(np.asarray(nodes), [gateway_id] + street_ids, int(graph_cfg.get("local_knn", 2)), edge_set, float(graph_cfg.get("local_max_edge_km", 4.0)))
            # The bounded kNN street model gives realistic local sparsity, but a
            # service territory must remain connected. Each local street receives
            # at least one access edge to its cluster gateway, analogous to a
            # community collector road.
            for street_id in street_ids:
                add_unique_edge(edge_set, gateway_id, street_id)

            zone_mean = float(customer_cfg.get("micro_zone_size_mean", 7.3))
            n_zones = max(1, int(round(count_i / max(zone_mean, 1.0))))
            zone_counts = sample_dirichlet_counts(self.rng, count_i, n_zones, alpha=3.5, min_count=1)

            # Customers are address-like stops inside a community, not a pure
            # Gaussian blob and not points exactly on a line. We first sample
            # micro-zone centers from a road-oriented community ellipse, then
            # expand each micro-zone mostly along the local road direction with
            # a smaller perpendicular spread.
            road_angle_jitter = math.radians(float(customer_cfg.get("road_angle_jitter_deg", 16.0)))
            community_long = scale * float(customer_cfg.get("community_long_axis_scale", 0.62))
            community_lat = scale * float(customer_cfg.get("community_lateral_axis_scale", 0.38))
            zone_long_base = float(customer_cfg.get("zone_longitudinal_median_km", 0.16))
            zone_lat_base = float(customer_cfg.get("zone_lateral_median_km", 0.07))
            address_jitter = float(customer_cfg.get("address_jitter_km", 0.006))
            min_spacing = float(customer_cfg.get("min_spacing_km", 0.025))

            for z_idx, z_count in enumerate(zone_counts):
                angle = self.rng.normal(0.0, road_angle_jitter)
                ca = math.cos(angle)
                sa = math.sin(angle)
                direction = ca * axis + sa * perp
                normal = np.asarray([-direction[1], direction[0]])

                center = gateway.copy()
                for _try in range(80):
                    center = gateway + self.rng.normal(0.0, community_long) * direction + self.rng.normal(0.0, community_lat) * normal
                    nearest_street = street_ids[nearest_node(center, np.asarray([nodes[x] for x in street_ids]))]
                    if np.linalg.norm(center - np.asarray(nodes[nearest_street])) <= float(customer_cfg.get("max_zone_center_to_street_km", 1.35)):
                        break

                zone_long_sigma = float(sample_truncated_lognormal(
                    self.rng,
                    zone_long_base,
                    float(customer_cfg.get("zone_longitudinal_sigma", 0.55)),
                    float(customer_cfg.get("zone_longitudinal_min_km", 0.035)),
                    float(customer_cfg.get("zone_longitudinal_max_km", 0.55)),
                    1,
                )[0])
                zone_lat_sigma = float(sample_truncated_lognormal(
                    self.rng,
                    zone_lat_base,
                    float(customer_cfg.get("zone_lateral_sigma", 0.55)),
                    float(customer_cfg.get("zone_lateral_min_km", 0.018)),
                    float(customer_cfg.get("zone_lateral_max_km", 0.24)),
                    1,
                )[0])

                for _ in range(int(z_count)):
                    point = center.copy()
                    for _try in range(120):
                        point = center + self.rng.normal(0.0, zone_long_sigma) * direction + self.rng.normal(0.0, zone_lat_sigma) * normal
                        if address_jitter > 0.0:
                            point = point + self.rng.normal(0.0, address_jitter, size=2)
                        if all(np.linalg.norm(point - prev) >= min_spacing for prev in customer_points[-250:]):
                            break
                    cid = self._new_node(nodes, point)
                    nearest_street = street_ids[nearest_node(point, np.asarray([nodes[x] for x in street_ids]))]
                    add_unique_edge(edge_set, cid, nearest_street)
                    customer_points.append(np.asarray(point, dtype=np.float32))
                    customer_node_ids.append(cid)
                    cluster_labels.append(cluster_id)
                    micro_zone_labels.append(micro_zone_id)
                micro_zone_id += 1

        cs_points: list[np.ndarray] = []
        cs_node_ids: list[int] = []
        corridor_ids = [depot_id] + gateway_ids + arterial_junction_ids
        # Pick candidate CS on arterial/corridor edges. Avoid customer-only edges.
        corridor_edges = [e for e in edge_set if e[0] in corridor_ids and e[1] in corridor_ids]
        if not corridor_edges:
            corridor_edges = list(edge_set)
        for idx in range(int(n_cs)):
            u, v = corridor_edges[int(self.rng.integers(0, len(corridor_edges)))]
            t_values = cs_cfg.get("candidate_t_values", [0.25, 0.40, 0.60, 0.75])
            t = float(t_values[idx % len(t_values)])
            point = (1.0 - t) * np.asarray(nodes[u]) + t * np.asarray(nodes[v])
            point += self.rng.normal(0.0, float(cs_cfg.get("position_jitter_km", 0.08)), size=2)
            cs_id = self._new_node(nodes, point)
            add_unique_edge(edge_set, cs_id, u)
            add_unique_edge(edge_set, cs_id, v)
            cs_points.append(np.asarray(point, dtype=np.float32))
            cs_node_ids.append(cs_id)

        node_arr = np.vstack(nodes).astype(np.float32)
        edge_arr, lengths = euclidean_edges(node_arr, sorted(edge_set), stretch=float(graph_cfg.get("road_stretch_factor", 1.12)))
        return {
            "road_nodes": node_arr,
            "road_edges": edge_arr,
            "road_edge_lengths_km": lengths,
            "depot_node_id": depot_id,
            "cluster_gateway_node_ids": np.asarray(gateway_ids, dtype=np.int32),
            "customers": np.vstack(customer_points).astype(np.float32),
            "customer_node_ids": np.asarray(customer_node_ids, dtype=np.int32),
            "charging_stations": np.vstack(cs_points).astype(np.float32),
            "cs_node_ids": np.asarray(cs_node_ids, dtype=np.int32),
            "cluster_labels": np.asarray(cluster_labels, dtype=np.int32),
            "micro_zone_labels": np.asarray(micro_zone_labels, dtype=np.int32),
        }

    def _candidate_cs_index(self, graph: dict[str, Any], k: int) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        adjacency = build_adjacency(len(graph["road_nodes"]), graph["road_edges"], graph["road_edge_lengths_km"])
        cs_node_ids = graph["cs_node_ids"]
        customer_node_ids = graph["customer_node_ids"]
        gateway_node_ids = graph["cluster_gateway_node_ids"]
        depot_id = int(graph["depot_node_id"])
        range_km = self.vehicle.battery_range_km

        road_nodes = graph["road_nodes"]
        cs_points = road_nodes[cs_node_ids]
        candidate_lists = []
        feasible_count = 0
        depot_dist = dijkstra_one(adjacency, depot_id)
        for node_id in customer_node_ids:
            # Candidate lists are only a pre-filter for daily CS activation.
            # The active instance still uses road shortest paths for all selected
            # terminals. Euclidean pre-filtering keeps 5k-customer service territories
            # cheap to build while preserving local infrastructure candidates.
            euclid = np.linalg.norm(cs_points - road_nodes[int(node_id)], axis=1)
            cs_order = np.argsort(euclid)[:k]
            candidate = cs_order.astype(np.int32)
            candidate_lists.append(candidate)
            depot_to_customer = depot_dist[int(node_id)]
            direct = depot_to_customer <= range_km
            assisted = bool(np.any(euclid[candidate] <= range_km))
            feasible_count += int(direct or assisted)

        cluster_lists = []
        gateway_feasible = 0
        for node_id in gateway_node_ids:
            euclid = np.linalg.norm(cs_points - road_nodes[int(node_id)], axis=1)
            cs_order = np.argsort(euclid)[:k]
            cluster_lists.append(cs_order.astype(np.int32))
            gateway_feasible += int(depot_dist[int(node_id)] <= range_km or np.any(euclid[cs_order] <= range_km))

        summary = {
            "candidate_cs_per_customer": int(k),
            "customer_reachability_rate": float(feasible_count / max(len(customer_node_ids), 1)),
            "cluster_gateway_feasible_rate": float(gateway_feasible / max(len(gateway_node_ids), 1)),
            "road_connected_from_depot_rate": float(np.mean(np.isfinite(depot_dist))),
            "battery_range_km": float(range_km),
            "graph_node_count": int(len(graph["road_nodes"])),
            "graph_edge_count": int(len(graph["road_edges"])),
        }
        return candidate_lists, cluster_lists, summary

    def generate(self, region_index: int, mother_num_customers: int, mother_num_charging_stations: int) -> RegionBoard:
        area = self.config.get("region", {}).get("area_size_km", [[0.0, 100.0], [0.0, 100.0]])
        region_profile = str(self.config.get("region", {}).get("profile", "mixed_amazon_station_region"))
        n_clusters = self._auto_cluster_count(mother_num_customers)
        depot = self._sample_depot(area)
        centers = self._sample_cluster_centers(depot, n_clusters, area)
        counts = sample_dirichlet_counts(
            self.rng,
            int(mother_num_customers),
            n_clusters,
            alpha=float(self.config.get("region", {}).get("cluster_assignment_alpha", 2.2)),
            min_count=max(1, int(self.config.get("region", {}).get("min_customers_per_cluster", 8))),
        )
        graph = self._build_region_graph(area, depot, centers, counts, int(mother_num_charging_stations))
        k = int(self.config.get("region", {}).get("candidate_cs_per_customer", 8))
        customer_candidates, cluster_candidates, feasibility = self._candidate_cs_index(graph, k)
        board = RegionBoard(
            region_id=f"region_{region_index:03d}",
            mother_board_id=f"board_{region_index:03d}",
            region_profile=region_profile,
            area_size_km=area,
            depot_node_id=int(graph["depot_node_id"]),
            road_nodes=graph["road_nodes"],
            road_edges=graph["road_edges"],
            road_edge_lengths_km=graph["road_edge_lengths_km"],
            customers=graph["customers"],
            customer_node_ids=graph["customer_node_ids"],
            charging_stations=graph["charging_stations"],
            cs_node_ids=graph["cs_node_ids"],
            cluster_centers=centers,
            cluster_gateway_node_ids=graph["cluster_gateway_node_ids"],
            cluster_labels=graph["cluster_labels"],
            micro_zone_labels=graph["micro_zone_labels"],
            customer_candidate_cs_ids=customer_candidates,
            cluster_candidate_cs_ids=cluster_candidates,
            region_validation=feasibility,
            metadata={
                "cluster_counts": counts.astype(int).tolist(),
                "generation_model": "road_network_first_service_territory_v1",
            },
            depot_candidates=graph["road_nodes"][[int(graph["depot_node_id"])]].astype(np.float32),
            depot_candidate_node_ids=np.asarray([int(graph["depot_node_id"])], dtype=np.int32),
            depot_candidate_metadata=[{"candidate_id": "depot_000", "source": "synthetic_depot"}],
        )
        min_customer_rate = float(self.config.get("region", {}).get("min_customer_reachability_rate", 0.995))
        min_connected_rate = float(self.config.get("region", {}).get("min_road_connected_from_depot_rate", 1.0))
        if board.region_validation["customer_reachability_rate"] < min_customer_rate:
            raise RuntimeError(f"Generated region is not structurally reachable enough: {board.region_validation}")
        if board.region_validation.get("road_connected_from_depot_rate", 0.0) < min_connected_rate:
            raise RuntimeError(f"Generated region road graph is disconnected: {board.region_validation}")
        return board
