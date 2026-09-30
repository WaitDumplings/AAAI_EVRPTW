from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evrptw_hierarchy.core.models import ActiveInstance, RegionBoard, RegionUsage
from evrptw_hierarchy.io.persistence import ensure_dir, write_csv


def _finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _optional_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_region(board: RegionBoard, usage: RegionUsage | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region_id": board.region_id,
        "mother_board_id": board.mother_board_id,
        "region_profile": board.region_profile,
        "num_road_nodes": int(len(board.road_nodes)),
        "num_road_edges": int(len(board.road_edges)),
        "num_latent_customers": int(len(board.customers)),
        "num_charging_station_pool": int(len(board.charging_stations)),
        "num_depot_candidates": int(0 if board.depot_candidate_node_ids is None else len(board.depot_candidate_node_ids)),
        "num_clusters": int(len(board.cluster_centers)),
        "num_micro_zones": int(np.max(board.micro_zone_labels) + 1) if len(board.micro_zone_labels) else 0,
        "customer_reachability_rate": float(board.region_validation.get("customer_reachability_rate", float("nan"))),
        "cluster_gateway_feasible_rate": float(board.region_validation.get("cluster_gateway_feasible_rate", float("nan"))),
        "road_connected_from_depot_rate": float(board.region_validation.get("road_connected_from_depot_rate", float("nan"))),
        "battery_range_km": float(board.region_validation.get("battery_range_km", float("nan"))),
        "geospatial_profile": bool(board.metadata.get("geospatial_profile", False)),
        "territory_id": board.metadata.get("territory_id", board.region_id),
        "county_fips": board.metadata.get("county_fips", ""),
        "customer_spacing_p10_km": _optional_float(board.region_validation.get("customer_spacing_p10_km")),
        "customer_spacing_p50_km": _optional_float(board.region_validation.get("customer_spacing_p50_km")),
        "customer_spacing_p90_km": _optional_float(board.region_validation.get("customer_spacing_p90_km")),
        "customer_snap_p90_km": _optional_float(board.region_validation.get("customer_snap_p90_km")),
        "charging_station_snap_p90_km": _optional_float(board.region_validation.get("charging_station_snap_p90_km")),
        "depot_candidate_snap_p90_km": _optional_float(board.region_validation.get("depot_candidate_snap_p90_km")),
        "customer_seed_count": int(board.region_validation.get("customer_seed_count", 0)),
        "customer_community_count": int(board.region_validation.get("customer_community_count", 0)),
        "occupancy_weight_entropy": _optional_float(board.region_validation.get("occupancy_weight_entropy")),
    }
    if usage is not None:
        row.update({
            "sampled_days": int(usage.sampled_days),
            "customer_exposure_rate": float(usage.customer_exposure_rate),
            "cluster_exposure_entropy": float(usage.cluster_exposure_entropy),
            "recent_mean_jaccard_distance": float(usage.recent_mean_jaccard_distance),
        })
    return row


def summarize_instance(instance: ActiveInstance) -> dict[str, Any]:
    tw_presence = np.mean(
        (instance.tw_s[:, 0] > instance.working_start_s) | (instance.tw_s[:, 1] < instance.working_end_s)
    ) if len(instance.tw_s) else 0.0
    offdiag = instance.distance_matrix_km[~np.eye(instance.distance_matrix_km.shape[0], dtype=bool)]
    return {
        "instance_id": instance.instance_id,
        "region_id": instance.region_id,
        "mother_board_id": instance.mother_board_id,
        "operating_day_id": instance.operating_day_id,
        "day_type": instance.day_type,
        "num_customers": int(len(instance.customers)),
        "num_charging_stations": int(len(instance.charging_stations)),
        "working_start_s": int(instance.working_start_s),
        "working_end_s": int(instance.working_end_s),
        "effective_speed_kmh": float(instance.speed_profile.get("effective_speed_kmh", float("nan"))),
        "congestion_factor": float(instance.speed_profile.get("congestion_factor", float("nan"))),
        "total_demand_cm3": float(np.sum(instance.demands_cm3)),
        "mean_demand_cm3": float(np.mean(instance.demands_cm3)) if len(instance.demands_cm3) else 0.0,
        "mean_package_count": float(np.mean(instance.package_counts)) if len(instance.package_counts) else 0.0,
        "mean_service_time_s": float(np.mean(instance.service_time_s)) if len(instance.service_time_s) else 0.0,
        "p90_service_time_s": _finite_quantile(instance.service_time_s, 0.90),
        "tw_presence_rate": float(tw_presence),
        "mean_road_distance_km": _finite_quantile(offdiag, 0.50),
        "p90_road_distance_km": _finite_quantile(offdiag, 0.90),
        "vehicle_upper_bound": int(instance.greedy_audit.get("vehicle_upper_bound") or 0),
        "greedy_success": bool(instance.greedy_audit.get("success", False)),
        "cs_candidate_pool_size": int(instance.cs_activation.get("candidate_pool_size", 0)),
        "mean_customer_to_nearest_cs_km": float(instance.cs_activation.get("mean_customer_to_nearest_cs_km", float("nan"))),
        "p90_customer_to_nearest_cs_km": float(instance.cs_activation.get("p90_customer_to_nearest_cs_km", float("nan"))),
        "mean_cs_time_to_depot_s": float(np.mean(instance.cs_time_to_depot_s)) if len(instance.cs_time_to_depot_s) else 0.0,
        "p90_cs_time_to_depot_s": _finite_quantile(instance.cs_time_to_depot_s, 0.90),
        "selected_depot_node_id": int(instance.metadata.get("depot_catchment", {}).get("selected_depot_node_id", -1)),
        "depot_catchment_customer_count": int(instance.metadata.get("depot_catchment", {}).get("catchment_customer_count", 0)),
        "depot_catchment_radius_km": _optional_float(
            instance.metadata.get("depot_catchment", {}).get("catchment_radius_km", float("nan"))
        ),
    }


def write_reports(
    save_path: str | Path,
    region_rows: list[dict[str, Any]],
    instance_rows: list[dict[str, Any]],
    failed_attempt_rows: list[dict[str, Any]],
) -> None:
    root = Path(save_path)
    metadata = ensure_dir(root / "metadata")
    analysis = ensure_dir(root / "analysis_outputs")
    write_csv(metadata / "region_usage.csv", region_rows)
    write_csv(metadata / "generation_summary.csv", instance_rows)
    write_csv(metadata / "failed_attempts.csv", failed_attempt_rows, fieldnames=["instance_index", "outer_attempt", "region_id", "error"])

    success_count = sum(1 for row in instance_rows if str(row.get("greedy_success", "True")) in {"True", "true", "1"})
    lines = [
        "# EVRP-TW-Hierarchy-D Generation Summary",
        "",
        f"Generated instances: {len(instance_rows)}",
        f"Greedy feasible instances: {success_count}",
        f"Failed outer attempts: {len(failed_attempt_rows)}",
        "",
        "## Region Freshness",
        "",
        "| region_id | sampled_days | exposure | cluster_entropy | recent_jaccard_distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in region_rows:
        lines.append(
            f"| {row.get('region_id')} | {row.get('sampled_days', 0)} | "
            f"{float(row.get('customer_exposure_rate', 0.0)):.3f} | "
            f"{float(row.get('cluster_exposure_entropy', 0.0)):.3f} | "
            f"{float(row.get('recent_mean_jaccard_distance', 1.0)):.3f} |"
        )
    lines.extend([
        "",
        "## Instance Summary",
        "",
        "| metric | mean | p10 | p90 |",
        "|---|---:|---:|---:|",
    ])
    for metric in [
        "effective_speed_kmh",
        "tw_presence_rate",
        "mean_service_time_s",
        "mean_demand_cm3",
        "vehicle_upper_bound",
        "p90_customer_to_nearest_cs_km",
        "mean_cs_time_to_depot_s",
    ]:
        vals = np.asarray([float(row.get(metric, float("nan"))) for row in instance_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lines.append(f"| {metric} | {np.mean(vals):.4f} | {np.quantile(vals, 0.10):.4f} | {np.quantile(vals, 0.90):.4f} |")
    (analysis / "daily_instance_vs_amazon.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
