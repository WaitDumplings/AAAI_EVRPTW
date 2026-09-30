from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from evrptw_hierarchy.configs.config import deep_update
from evrptw_hierarchy.core.models import RegionBoard, VehicleConfig
from evrptw_hierarchy.generation.region_generator import RegionGenerator
from evrptw_hierarchy.graph.shortest_path import build_adjacency, dijkstra_one
from evrptw_hierarchy.graph.road_graph import add_unique_edge, euclidean_edges, nearest_node


def _read_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path in (None, ""):
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], keys: tuple[str, ...], default: float | None = None) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    if default is None:
        raise KeyError(f"Missing numeric column; expected one of {keys}")
    return float(default)


def _bbox_center(bbox: list[float] | tuple[float, ...] | None) -> tuple[float, float]:
    if not bbox or len(bbox) != 4:
        return -96.0, 39.0
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    return 0.5 * (min_lon + max_lon), 0.5 * (min_lat + max_lat)


def _lonlat_to_km(lon: float, lat: float, lon0: float, lat0: float) -> np.ndarray:
    x = (float(lon) - lon0) * 111.320 * np.cos(np.deg2rad(lat0))
    y = (float(lat) - lat0) * 110.574
    return np.asarray([x, y], dtype=np.float32)


def _coords_from_rows(rows: list[dict[str, str]], spec: dict[str, Any]) -> np.ndarray:
    if not rows:
        return np.empty((0, 2), dtype=np.float32)
    lon0, lat0 = _bbox_center(spec.get("bbox_lonlat"))
    out = []
    for row in rows:
        if row.get("x_km") not in (None, "") and row.get("y_km") not in (None, ""):
            out.append([float(row["x_km"]), float(row["y_km"])])
        else:
            lon = _as_float(row, ("lon", "longitude", "lng", "x"))
            lat = _as_float(row, ("lat", "latitude", "y"))
            out.append(_lonlat_to_km(lon, lat, lon0, lat0))
    return np.asarray(out, dtype=np.float32)


def _nearest_indices(points: np.ndarray, candidates: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    out = np.empty(points.shape[0], dtype=np.int32)
    for start in range(0, points.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), points.shape[0])
        block = points[start:stop]
        dist = np.sum((block[:, None, :] - candidates[None, :, :]) ** 2, axis=2)
        out[start:stop] = np.argmin(dist, axis=1).astype(np.int32)
    return out


def _nearest_distances(points: np.ndarray, candidates: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    out = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), points.shape[0])
        block = points[start:stop]
        dist = np.linalg.norm(block[:, None, :] - candidates[None, :, :], axis=2)
        out[start:stop] = np.min(dist, axis=1).astype(np.float32)
    return out


def _node_ids_from_rows(
    rows: list[dict[str, str]],
    key: str,
    node_id_to_idx: dict[str, int],
) -> np.ndarray | None:
    if not rows:
        return None
    out = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            return None
        idx = node_id_to_idx.get(str(value))
        if idx is None:
            return None
        out.append(idx)
    return np.asarray(out, dtype=np.int32)


def _spread_node_ids(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    count = min(int(count), points.shape[0])
    selected = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=1))
    return np.asarray(selected, dtype=np.int32)


class GeospatialTerritoryBuilder:
    """Build ``RegionBoard`` objects from county geospatial source tables.

    The first production path expects normalized local CSV files. This keeps the
    benchmark reproducible and avoids embedding network/API terms in the
    generator. If required files are not available, the builder falls back to the
    existing synthetic region generator and annotates the territory as a
    deterministic geospatial scaffold.
    """

    def __init__(self, base_config: dict[str, Any], vehicle: VehicleConfig, rng: np.random.Generator):
        self.base_config = base_config
        self.vehicle = vehicle
        self.rng = rng

    def build(self, spec: dict[str, Any], region_index: int) -> RegionBoard:
        sources = spec.get("source_files", {}) or {}
        required = ("road_nodes_csv", "road_edges_csv", "customer_seed_csv")
        if all(sources.get(key) not in (None, "") and Path(str(sources.get(key))).exists() for key in required):
            return self._build_from_standard_csv(spec, region_index)
        return self._build_fallback(spec, region_index)

    def _config_for_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = deep_update(self.base_config, spec.get("config_overrides", {}) or {})
        area = spec.get("area_size_km")
        if area is not None:
            cfg = deep_update(cfg, {"region": {"area_size_km": area}})
        cfg = deep_update(
            cfg,
            {
                "geospatial": {
                    "depot_catchment": spec.get(
                        "depot_catchment",
                        {"start_radius_km": 40.0, "max_radius_km": 55.0},
                    )
                }
            },
        )
        return cfg

    def _relabel_board(self, board: RegionBoard, spec: dict[str, Any], source_mode: str) -> RegionBoard:
        territory_id = str(spec["territory_id"])
        board.region_id = territory_id
        board.mother_board_id = f"{territory_id}_county_board"
        board.region_profile = "real_geography_semi_synthetic_geo_ac_v1"
        board.metadata.update(
            {
                "geospatial_profile": True,
                "generation_model": "county_container_depot_catchment_v1",
                "source_mode": source_mode,
                "territory_id": territory_id,
                "display_name": spec.get("display_name", territory_id),
                "county_name": spec.get("county_name", ""),
                "state": spec.get("state", ""),
                "county_fips": str(spec.get("county_fips", "")),
                "bbox_lonlat": spec.get("bbox_lonlat", []),
                "latent_customer_pool_size": int(spec.get("latent_customer_pool_size", len(board.customers))),
            }
        )
        data_source_versions = spec.get("data_source_versions", {})
        data_filters = spec.get("data_filters", {})
        if data_source_versions:
            board.metadata["data_source_versions"] = data_source_versions
        if data_filters:
            board.metadata["data_filters"] = data_filters
        self._add_common_validation(board)
        return board

    def _build_fallback(self, spec: dict[str, Any], region_index: int) -> RegionBoard:
        cfg = self._config_for_spec(spec)
        generator = RegionGenerator(cfg, self.vehicle, self.rng)
        latent = int(spec.get("latent_customer_pool_size", spec.get("latent_customers", 20_000)))
        chargers = int(spec.get("cs_candidate_pool_size", spec.get("charging_station_pool_size", 120)))
        board = generator.generate(region_index, latent, chargers)
        self._attach_depot_candidates_from_roads(board, spec)
        return self._relabel_board(board, spec, "fallback_synthetic_county_scaffold")

    def _build_from_standard_csv(self, spec: dict[str, Any], region_index: int) -> RegionBoard:
        del region_index
        sources = spec.get("source_files", {}) or {}
        node_rows = _read_csv(sources.get("road_nodes_csv"))
        edge_rows = _read_csv(sources.get("road_edges_csv"))
        seed_rows = _read_csv(sources.get("customer_seed_csv"))
        latent_rows = _read_csv(sources.get("latent_customer_csv"))
        charger_rows = _read_csv(sources.get("charging_station_csv"))
        depot_rows = _read_csv(sources.get("depot_candidate_csv"))

        base_nodes = _coords_from_rows(node_rows, spec)
        if base_nodes.shape[0] < 2:
            raise ValueError(f"{spec['territory_id']} needs at least two road nodes.")
        node_id_to_idx = {str(row.get("node_id", row.get("id", idx))): idx for idx, row in enumerate(node_rows)}
        edge_set: set[tuple[int, int]] = set()
        explicit_lengths: dict[tuple[int, int], float] = {}
        for row in edge_rows:
            u_raw = str(row.get("u", row.get("from", row.get("source", ""))))
            v_raw = str(row.get("v", row.get("to", row.get("target", ""))))
            if u_raw not in node_id_to_idx or v_raw not in node_id_to_idx:
                continue
            u = node_id_to_idx[u_raw]
            v = node_id_to_idx[v_raw]
            add_unique_edge(edge_set, u, v)
            if row.get("length_km") not in (None, ""):
                explicit_lengths[tuple(sorted((u, v)))] = float(row["length_km"])
        if not edge_set:
            raise ValueError(f"{spec['territory_id']} has no usable road edges.")

        seed_points = _coords_from_rows(seed_rows, spec)
        if seed_points.shape[0] == 0:
            raise ValueError(f"{spec['territory_id']} has no customer seed rows.")
        customer_connector_km: np.ndarray | None = None
        customer_connection_mode = "eager_terminal_nodes"
        if latent_rows:
            customers = _coords_from_rows(latent_rows, spec)
            cluster_labels, micro_zone_labels = self._labels_from_customer_rows(latent_rows)
            customer_node_ids_precomputed = _node_ids_from_rows(latent_rows, "snap_node_id", node_id_to_idx)
            if customer_node_ids_precomputed is None:
                customer_node_ids_precomputed = _nearest_indices(customers, base_nodes)
            customer_connector_km = np.asarray(
                [
                    _as_float(row, ("connector_distance_km", "snap_distance_km", "access_distance_km"), 0.0)
                    for row in latent_rows
                ],
                dtype=np.float32,
            )
            customer_connection_mode = "lazy_snap_node_connector"
        else:
            latent = int(spec.get("latent_customer_pool_size", 20_000))
            customers, cluster_labels, micro_zone_labels = self._sample_customers_from_seeds(seed_rows, seed_points, latent, spec)
            customer_node_ids_precomputed = None

        chargers = _coords_from_rows(charger_rows, spec)
        if chargers.shape[0] == 0:
            chargers = self._fallback_chargers_on_roads(base_nodes, edge_set, int(spec.get("cs_candidate_pool_size", 120)))
        else:
            max_cs = int(spec.get("cs_candidate_pool_size", chargers.shape[0]))
            chargers = chargers[:max_cs]

        depots = _coords_from_rows(depot_rows, spec)
        if depots.shape[0] == 0:
            depot_node_local = _spread_node_ids(base_nodes, int(spec.get("depot_candidate_count", 6)))
            depots = base_nodes[depot_node_local]
            depot_meta = [{"candidate_id": f"depot_{idx:03d}", "source": "road_node_fallback"} for idx in range(len(depots))]
        else:
            depot_meta = [
                {"candidate_id": row.get("candidate_id", f"depot_{idx:03d}"), "source": row.get("source", "open_poi")}
                for idx, row in enumerate(depot_rows[: len(depots)])
            ]

        nodes = [np.asarray(point, dtype=np.float32) for point in base_nodes]
        if customer_node_ids_precomputed is None:
            customer_node_ids = self._append_terminals(nodes, edge_set, customers, base_nodes)
        else:
            customer_node_ids = np.asarray(customer_node_ids_precomputed, dtype=np.int32)
        cs_node_ids = self._append_terminals(nodes, edge_set, chargers, base_nodes)
        depot_candidate_node_ids = self._append_terminals(nodes, edge_set, depots, base_nodes)
        road_nodes = np.vstack(nodes).astype(np.float32)
        edge_arr, lengths = euclidean_edges(road_nodes, sorted(edge_set), stretch=1.0)
        if explicit_lengths:
            for idx, (u, v) in enumerate(edge_arr):
                value = explicit_lengths.get(tuple(sorted((int(u), int(v)))))
                if value is not None:
                    lengths[idx] = float(value)

        cluster_centers, cluster_gateway_node_ids = self._cluster_centers_and_gateways(
            customers,
            cluster_labels,
            road_nodes,
            np.arange(base_nodes.shape[0], dtype=np.int32),
        )
        graph = {
            "road_nodes": road_nodes,
            "road_edges": edge_arr,
            "road_edge_lengths_km": lengths,
            "depot_node_id": int(depot_candidate_node_ids[0]),
            "cluster_gateway_node_ids": cluster_gateway_node_ids,
            "customer_node_ids": customer_node_ids,
            "cs_node_ids": cs_node_ids,
        }
        cfg = self._config_for_spec(spec)
        k = int(cfg.get("region", {}).get("candidate_cs_per_customer", 8))
        helper = RegionGenerator(cfg, self.vehicle, self.rng)
        if customer_connection_mode.startswith("lazy_"):
            customer_candidates, cluster_candidates, validation = self._lazy_candidate_cs_index(graph, k)
        else:
            customer_candidates, cluster_candidates, validation = helper._candidate_cs_index(graph, k)
        validation.update(
            {
                "source_mode": "standard_geospatial_csv",
                "num_depot_candidates": int(len(depot_candidate_node_ids)),
                "customer_connection_mode": customer_connection_mode,
                **self._customer_connector_summary(customer_connector_km),
                **self._customer_spacing_summary(customers),
                **(
                    self._customer_snap_summary_from_rows(latent_rows)
                    if latent_rows
                    else self._snap_summary("customer", customers, base_nodes)
                ),
                **self._snap_summary("charging_station", chargers, base_nodes),
                **self._snap_summary("depot_candidate", depots, base_nodes),
                **self._occupancy_summary(seed_rows),
            }
        )

        board = RegionBoard(
            region_id=str(spec["territory_id"]),
            mother_board_id=f"{spec['territory_id']}_county_board",
            region_profile="real_geography_semi_synthetic_geo_ac_v1",
            area_size_km=self._area_size_from_points(road_nodes),
            depot_node_id=int(depot_candidate_node_ids[0]),
            road_nodes=road_nodes,
            road_edges=edge_arr,
            road_edge_lengths_km=lengths,
            customers=customers.astype(np.float32),
            customer_node_ids=customer_node_ids.astype(np.int32),
            charging_stations=chargers.astype(np.float32),
            cs_node_ids=cs_node_ids.astype(np.int32),
            cluster_centers=cluster_centers.astype(np.float32),
            cluster_gateway_node_ids=cluster_gateway_node_ids.astype(np.int32),
            cluster_labels=cluster_labels.astype(np.int32),
            micro_zone_labels=micro_zone_labels.astype(np.int32),
            customer_candidate_cs_ids=customer_candidates,
            cluster_candidate_cs_ids=cluster_candidates,
            region_validation=validation,
            metadata={
                "source_files": sources,
                "customer_seed_count": int(len(seed_rows)),
                "latent_customer_count": int(len(customers)),
                "customer_connection_mode": customer_connection_mode,
                "raw_charging_station_count": int(len(charger_rows)),
                "raw_depot_candidate_count": int(len(depot_rows)),
                **({"customer_connector_km": customer_connector_km} if customer_connector_km is not None else {}),
            },
            depot_candidates=depots.astype(np.float32),
            depot_candidate_node_ids=depot_candidate_node_ids.astype(np.int32),
            depot_candidate_metadata=depot_meta,
        )
        return self._relabel_board(board, spec, "standard_geospatial_csv")

    def _append_terminals(
        self,
        nodes: list[np.ndarray],
        edge_set: set[tuple[int, int]],
        points: np.ndarray,
        base_nodes: np.ndarray,
    ) -> np.ndarray:
        nearest = _nearest_indices(points, base_nodes)
        out = []
        for point, base_idx in zip(np.asarray(points, dtype=np.float32), nearest):
            node_id = len(nodes)
            nodes.append(point)
            add_unique_edge(edge_set, node_id, int(base_idx))
            out.append(node_id)
        return np.asarray(out, dtype=np.int32)

    def _labels_from_customer_rows(self, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        community_lookup: dict[str, int] = {}
        cluster_labels = []
        micro_zone_labels = []
        for idx, row in enumerate(rows):
            community = row.get("community_id") or row.get("block_group") or row.get("tract") or str(idx)
            if community not in community_lookup:
                community_lookup[community] = len(community_lookup)
            label = int(community_lookup[community])
            cluster_labels.append(label)
            micro_zone_labels.append(label)
        return np.asarray(cluster_labels, dtype=np.int32), np.asarray(micro_zone_labels, dtype=np.int32)

    def _sample_customers_from_seeds(
        self,
        seed_rows: list[dict[str, str]],
        seed_points: np.ndarray,
        latent: int,
        spec: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.asarray([
            _as_float(row, ("occupancy", "occupied_units", "housing_units", "weight"), 1.0)
            for row in seed_rows
        ], dtype=np.float64)
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum()
        min_spacing = float(spec.get("customer_min_spacing_km", 0.05))
        jitter = float(spec.get("customer_seed_jitter_km", 0.12))
        chosen = self.rng.choice(seed_points.shape[0], size=int(latent), replace=True, p=weights)
        customers = np.empty((int(latent), 2), dtype=np.float32)
        grid: dict[tuple[int, int], list[int]] = {}

        def cell(point: np.ndarray) -> tuple[int, int]:
            return int(np.floor(point[0] / min_spacing)), int(np.floor(point[1] / min_spacing))

        def far_enough(point: np.ndarray) -> bool:
            cx, cy = cell(point)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for idx in grid.get((cx + dx, cy + dy), []):
                        if float(np.linalg.norm(point - customers[idx])) < min_spacing:
                            return False
            return True

        for idx, seed_idx in enumerate(chosen):
            base = seed_points[int(seed_idx)]
            point = base.copy()
            for _ in range(40):
                point = base + self.rng.normal(0.0, jitter, size=2)
                if far_enough(point):
                    break
            customers[idx] = point.astype(np.float32)
            grid.setdefault(cell(customers[idx]), []).append(idx)

        raw_community = [
            row.get("community_id") or row.get("block_group") or row.get("tract") or str(i)
            for i, row in enumerate(seed_rows)
        ]
        community_lookup: dict[str, int] = {}
        seed_cluster = []
        for value in raw_community:
            if value not in community_lookup:
                community_lookup[value] = len(community_lookup)
            seed_cluster.append(community_lookup[value])
        cluster_labels = np.asarray([seed_cluster[int(idx)] for idx in chosen], dtype=np.int32)
        micro_zone_labels = chosen.astype(np.int32)
        return customers, cluster_labels, micro_zone_labels

    def _fallback_chargers_on_roads(
        self,
        nodes: np.ndarray,
        edge_set: set[tuple[int, int]],
        count: int,
    ) -> np.ndarray:
        edges = list(edge_set)
        out = []
        for idx in range(int(count)):
            u, v = edges[int(self.rng.integers(0, len(edges)))]
            t = [0.25, 0.4, 0.6, 0.75][idx % 4]
            point = (1.0 - t) * nodes[u] + t * nodes[v] + self.rng.normal(0.0, 0.04, size=2)
            out.append(point)
        return np.asarray(out, dtype=np.float32)

    def _cluster_centers_and_gateways(
        self,
        customers: np.ndarray,
        cluster_labels: np.ndarray,
        road_nodes: np.ndarray,
        base_node_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cluster_count = int(cluster_labels.max()) + 1
        centers = []
        gateways = []
        base_points = road_nodes[base_node_ids]
        for cluster_id in range(cluster_count):
            pts = customers[cluster_labels == cluster_id]
            center = pts.mean(axis=0) if pts.size else customers.mean(axis=0)
            centers.append(center)
            gateways.append(int(base_node_ids[nearest_node(center, base_points)]))
        return np.asarray(centers, dtype=np.float32), np.asarray(gateways, dtype=np.int32)

    def _lazy_candidate_cs_index(self, graph: dict[str, Any], k: int) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        road_nodes = graph["road_nodes"]
        cs_node_ids = np.asarray(graph["cs_node_ids"], dtype=np.int32)
        gateway_node_ids = np.asarray(graph["cluster_gateway_node_ids"], dtype=np.int32)
        cs_points = road_nodes[cs_node_ids]
        cluster_lists = []
        for node_id in gateway_node_ids:
            euclid = np.linalg.norm(cs_points - road_nodes[int(node_id)], axis=1)
            cluster_lists.append(np.argsort(euclid, kind="mergesort")[: int(k)].astype(np.int32))

        adjacency = build_adjacency(len(graph["road_nodes"]), graph["road_edges"], graph["road_edge_lengths_km"])
        depot_dist = dijkstra_one(adjacency, int(graph["depot_node_id"]))
        customer_node_ids = np.asarray(graph["customer_node_ids"], dtype=np.int32)
        if customer_node_ids.size:
            sample = customer_node_ids[: min(customer_node_ids.size, 5000)]
            reachable = np.isfinite(depot_dist[sample])
            customer_reachability = float(np.mean(reachable))
        else:
            customer_reachability = 0.0
        gateway_feasible = 0
        range_km = self.vehicle.battery_range_km
        for node_id, candidates in zip(gateway_node_ids, cluster_lists):
            euclid = np.linalg.norm(cs_points - road_nodes[int(node_id)], axis=1)
            gateway_feasible += int(depot_dist[int(node_id)] <= range_km or np.any(euclid[candidates] <= range_km))
        return [], cluster_lists, {
            "candidate_cs_per_customer": int(k),
            "customer_candidate_cs_mode": "lazy_active_day_euclidean",
            "customer_reachability_rate": customer_reachability,
            "customer_reachability_sample_size": int(min(customer_node_ids.size, 5000)),
            "cluster_gateway_feasible_rate": float(gateway_feasible / max(len(gateway_node_ids), 1)),
            "road_connected_from_depot_rate": float(np.mean(np.isfinite(depot_dist))),
            "battery_range_km": float(range_km),
            "graph_node_count": int(len(graph["road_nodes"])),
            "graph_edge_count": int(len(graph["road_edges"])),
        }

    def _attach_depot_candidates_from_roads(self, board: RegionBoard, spec: dict[str, Any]) -> None:
        count = int(spec.get("depot_candidate_count", 6))
        base_count = min(len(board.cluster_gateway_node_ids), max(count - 1, 0))
        gateway_points = board.road_nodes[board.cluster_gateway_node_ids] if base_count else np.empty((0, 2), dtype=np.float32)
        spread_local = _spread_node_ids(gateway_points, base_count) if base_count else np.empty(0, dtype=np.int32)
        depot_nodes = [int(board.depot_node_id)]
        depot_nodes.extend(int(board.cluster_gateway_node_ids[int(idx)]) for idx in spread_local)
        board.depot_candidate_node_ids = np.asarray(list(dict.fromkeys(depot_nodes)), dtype=np.int32)
        board.depot_candidates = board.road_nodes[board.depot_candidate_node_ids].astype(np.float32)
        board.depot_candidate_metadata = [
            {"candidate_id": f"depot_{idx:03d}", "source": "county_scaffold_candidate"}
            for idx in range(len(board.depot_candidate_node_ids))
        ]
        board.region_validation["num_depot_candidates"] = int(len(board.depot_candidate_node_ids))

    def _area_size_from_points(self, points: np.ndarray) -> list[list[float]]:
        lo = np.min(points, axis=0)
        hi = np.max(points, axis=0)
        return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]

    def _customer_spacing_summary(self, customers: np.ndarray) -> dict[str, float]:
        sample = customers
        if len(sample) > 2000:
            sample = sample[self.rng.choice(len(sample), size=2000, replace=False)]
        if len(sample) < 2:
            return {
                "customer_spacing_p10_km": float("nan"),
                "customer_spacing_p50_km": float("nan"),
                "customer_spacing_p90_km": float("nan"),
            }
        nearest = []
        for start in range(0, len(sample), 256):
            block = sample[start:start + 256]
            dist = np.linalg.norm(block[:, None, :] - sample[None, :, :], axis=2)
            dist[:, start:start + len(block)] += np.eye(len(block)) * 1e9
            nearest.extend(np.min(dist, axis=1).tolist())
        arr = np.asarray(nearest, dtype=float)
        if arr.size == 0:
            return {
                "customer_spacing_p10_km": float("nan"),
                "customer_spacing_p50_km": float("nan"),
                "customer_spacing_p90_km": float("nan"),
            }
        return {
            "customer_spacing_p10_km": float(np.quantile(arr, 0.10)),
            "customer_spacing_p50_km": float(np.quantile(arr, 0.50)),
            "customer_spacing_p90_km": float(np.quantile(arr, 0.90)),
        }

    def _customer_connector_summary(self, connector_km: np.ndarray | None) -> dict[str, float | str]:
        if connector_km is None or len(connector_km) == 0:
            return {
                "customer_connector_mode": "eager_terminal_nodes",
                "customer_connector_p50_km": float("nan"),
                "customer_connector_p90_km": float("nan"),
            }
        arr = np.asarray(connector_km, dtype=np.float32)
        return {
            "customer_connector_mode": "lazy_snap_node_connector",
            "customer_connector_p50_km": float(np.quantile(arr, 0.50)),
            "customer_connector_p90_km": float(np.quantile(arr, 0.90)),
        }

    def _customer_snap_summary_from_rows(self, rows: list[dict[str, str]]) -> dict[str, float]:
        if not rows:
            return {
                "customer_snap_p50_km": float("nan"),
                "customer_snap_p90_km": float("nan"),
                "customer_snap_max_km": float("nan"),
            }
        snap = np.asarray([
            _as_float(row, ("snap_distance_km", "access_distance_km"), 0.0)
            for row in rows
        ], dtype=np.float32)
        return {
            "customer_snap_p50_km": float(np.quantile(snap, 0.50)),
            "customer_snap_p90_km": float(np.quantile(snap, 0.90)),
            "customer_snap_max_km": float(np.max(snap)),
        }

    def _snap_summary(self, prefix: str, points: np.ndarray, base_nodes: np.ndarray) -> dict[str, float]:
        if len(points) == 0 or len(base_nodes) == 0:
            return {
                f"{prefix}_snap_p50_km": float("nan"),
                f"{prefix}_snap_p90_km": float("nan"),
                f"{prefix}_snap_max_km": float("nan"),
            }
        dist = _nearest_distances(points, base_nodes)
        return {
            f"{prefix}_snap_p50_km": float(np.quantile(dist, 0.50)),
            f"{prefix}_snap_p90_km": float(np.quantile(dist, 0.90)),
            f"{prefix}_snap_max_km": float(np.max(dist)),
        }

    def _occupancy_summary(self, seed_rows: list[dict[str, str]]) -> dict[str, float | int]:
        if not seed_rows:
            return {
                "customer_seed_count": 0,
                "customer_community_count": 0,
                "occupancy_total": 0.0,
                "occupancy_weight_entropy": float("nan"),
            }
        occupancy = np.asarray([
            _as_float(row, ("occupancy", "occupied_units", "housing_units", "weight"), 1.0)
            for row in seed_rows
        ], dtype=np.float64)
        occupancy = np.maximum(occupancy, 0.0)
        total = float(occupancy.sum())
        probs = occupancy / max(total, 1e-12)
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum() / max(np.log(len(probs)), 1e-12)) if len(probs) > 1 else 0.0
        communities = {
            row.get("community_id") or row.get("block_group") or row.get("tract") or str(idx)
            for idx, row in enumerate(seed_rows)
        }
        return {
            "customer_seed_count": int(len(seed_rows)),
            "customer_community_count": int(len(communities)),
            "occupancy_total": total,
            "occupancy_weight_entropy": entropy,
        }

    def _add_common_validation(self, board: RegionBoard) -> None:
        spacing_keys = {"customer_spacing_p10_km", "customer_spacing_p50_km", "customer_spacing_p90_km"}
        if not spacing_keys.issubset(board.region_validation):
            board.region_validation.update(self._customer_spacing_summary(board.customers))
        board.region_validation["num_depot_candidates"] = int(
            0 if board.depot_candidate_node_ids is None else len(board.depot_candidate_node_ids)
        )
