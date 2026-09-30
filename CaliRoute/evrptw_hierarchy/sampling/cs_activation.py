from __future__ import annotations

from typing import Any

import numpy as np

from evrptw_hierarchy.core.models import RegionBoard
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle


def _customer_connector_km(board: RegionBoard, customer_ids: np.ndarray) -> np.ndarray:
    values = board.metadata.get("customer_connector_km")
    ids = np.asarray(customer_ids, dtype=np.int32)
    if values is None:
        return np.zeros(ids.size, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < len(board.customers):
        return np.zeros(ids.size, dtype=np.float32)
    return arr[ids].astype(np.float32, copy=False)


def build_candidate_pool(board: RegionBoard, active_customer_ids: np.ndarray, max_candidates: int | None = None) -> np.ndarray:
    lazy_customer_candidates = len(board.customer_candidate_cs_ids) < len(board.customers)
    cs_count = len(board.charging_stations)
    candidate_mask = np.zeros(cs_count, dtype=bool)
    active_customer_ids = np.asarray(active_customer_ids, dtype=int)
    if lazy_customer_candidates:
        k = 8 if max_candidates is None else max(1, min(int(max_candidates), cs_count))
        cs_points = board.road_nodes[board.cs_node_ids]
        customer_points = board.road_nodes[board.customer_node_ids[active_customer_ids]]
        for point in customer_points:
            euclid = np.linalg.norm(cs_points - point.reshape(1, 2), axis=1)
            candidate_mask[np.argsort(euclid, kind="mergesort")[:k]] = True
    else:
        for cid in active_customer_ids:
            ids = np.asarray(board.customer_candidate_cs_ids[int(cid)], dtype=np.int32)
            candidate_mask[ids] = True
    active_clusters = np.unique(board.cluster_labels[active_customer_ids])
    for cluster_id in active_clusters:
        ids = np.asarray(board.cluster_candidate_cs_ids[int(cluster_id)], dtype=np.int32)
        candidate_mask[ids] = True
    if not np.any(candidate_mask):
        candidate_mask[:] = True
    return np.flatnonzero(candidate_mask).astype(np.int32)


def activate_charging_stations(
    board: RegionBoard,
    active_customer_ids: np.ndarray,
    num_charging_stations: int,
    oracle: DistanceOracle,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config.get("cs_activation", {})
    k = int(num_charging_stations)
    cs_count = len(board.charging_stations)
    active_customer_ids = np.asarray(active_customer_ids, dtype=int)
    if k >= cs_count:
        selected = np.arange(cs_count, dtype=np.int32)
        customer_nodes = board.customer_node_ids[active_customer_ids]
        cs_nodes = board.cs_node_ids[selected]
        dist_customer_cs = oracle.matrix_between(customer_nodes, cs_nodes).astype(np.float32, copy=False)
        connector = _customer_connector_km(board, active_customer_ids)
        if connector.size:
            dist_customer_cs = dist_customer_cs + connector[:, None]
        final_d = np.min(dist_customer_cs, axis=1) if dist_customer_cs.size else np.full(len(active_customer_ids), np.inf)
        finite_final = np.isfinite(final_d)
        safe_final = final_d[finite_final] if np.any(finite_final) else np.asarray([float("nan")])
        cluster_labels = board.cluster_labels[active_customer_ids]
        covered_clusters = len(np.unique(cluster_labels[finite_final]))
        metadata = {
            "policy": "all_available_charging_stations",
            "candidate_pool_size": int(cs_count),
            "original_candidate_pool_size": int(cs_count),
            "candidate_truncation_policy": "none",
            "selected_count": int(selected.size),
            "mean_customer_to_nearest_cs_km": float(np.mean(safe_final)),
            "p90_customer_to_nearest_cs_km": float(np.quantile(safe_final, 0.90)),
            "max_customer_to_nearest_cs_km": float(np.max(safe_final)),
            "active_cluster_count": int(len(np.unique(cluster_labels))),
            "covered_cluster_count": int(covered_clusters),
            "unreachable_customer_rate_to_selected_cs": float(np.mean(~finite_final)),
            "objective": float(np.mean(safe_final)) if np.any(finite_final) else float("inf"),
        }
        return selected, metadata

    configured_max_candidates = int(cfg.get("max_candidate_pool", max(4 * k, 64)))
    max_candidates = max(k, configured_max_candidates)
    candidate_ids = build_candidate_pool(board, active_customer_ids, max_candidates=None)
    if candidate_ids.size < k:
        missing_mask = np.ones(len(board.charging_stations), dtype=bool)
        missing_mask[candidate_ids] = False
        extra = np.flatnonzero(missing_mask).astype(np.int32)[: k - candidate_ids.size]
        if extra.size:
            candidate_ids = np.concatenate([candidate_ids, extra])
    customer_nodes = board.customer_node_ids[active_customer_ids]
    cs_nodes = board.cs_node_ids[candidate_ids]

    dist_customer_cs = oracle.matrix_between(customer_nodes, cs_nodes).astype(np.float32, copy=False)
    connector = _customer_connector_km(board, active_customer_ids)
    if connector.size:
        dist_customer_cs = dist_customer_cs + connector[:, None]

    cluster_labels = board.cluster_labels[active_customer_ids]
    cluster_weights = np.ones(len(active_customer_ids), dtype=np.float32)
    weights = cluster_weights / max(float(cluster_weights.sum()), 1e-12)
    original_candidate_pool_size = int(candidate_ids.size)
    if candidate_ids.size > max_candidates:
        finite = np.isfinite(dist_customer_cs)
        safe_nan = np.where(finite, dist_customer_cs, np.nan)
        with np.errstate(invalid="ignore"):
            mean_score = np.nanmean(safe_nan, axis=0)
            p90_score = np.nanquantile(safe_nan, 0.90, axis=0)
        mean_score = np.where(np.isfinite(mean_score), mean_score, np.inf)
        p90_score = np.where(np.isfinite(p90_score), p90_score, np.inf)
        candidate_score = mean_score + 0.25 * p90_score
        keep = np.argsort(candidate_score, kind="mergesort")[:max_candidates]
        candidate_ids = candidate_ids[keep]
        cs_nodes = cs_nodes[keep]
        dist_customer_cs = dist_customer_cs[:, keep]

    selected_local: list[int] = []
    available = list(range(len(candidate_ids)))
    repulsion_km = float(cfg.get("repulsion_km", 1.5))
    alpha = float(cfg.get("alpha_mean", 1.0))
    beta = float(cfg.get("beta_p90", 0.6))
    gamma = float(cfg.get("gamma_max", 0.4))
    eta = float(cfg.get("eta_redundancy", 0.3))
    unreachable_penalty = float(cfg.get("unreachable_customer_penalty", 10000.0))
    cs_pair_dist = oracle.matrix_between(cs_nodes, cs_nodes).astype(np.float32, copy=False)
    current_nearest = np.full(len(active_customer_ids), np.inf, dtype=np.float32)
    current_min_pair = np.inf

    def objective(local_ids: list[int]) -> float:
        if not local_ids:
            return float("inf")
        d = np.min(dist_customer_cs[:, local_ids], axis=1)
        finite = np.isfinite(d)
        if not np.any(finite):
            return float("inf")
        penalty = unreachable_penalty * float(np.mean(~finite))
        safe_d = d.copy()
        safe_d[~finite] = float(np.max(d[finite]) + penalty + 1.0)
        score = alpha * float(np.sum(weights * safe_d))
        score += beta * float(np.quantile(safe_d, 0.90))
        score += gamma * float(np.max(safe_d))
        score += penalty
        if len(local_ids) >= 2:
            pair = cs_pair_dist[np.ix_(local_ids, local_ids)]
            pair = pair[pair > 1e-9]
            min_pair = float(pair.min()) if pair.size else np.inf
            if np.isfinite(min_pair) and min_pair < repulsion_km:
                score += eta * (repulsion_km - min_pair) * 10.0
        return score

    def batch_scores(avail: list[int]) -> np.ndarray:
        if not avail:
            return np.asarray([], dtype=np.float64)
        cand = dist_customer_cs[:, np.asarray(avail, dtype=int)]
        d = np.minimum(current_nearest[:, None], cand)
        finite = np.isfinite(d)
        any_finite = np.any(finite, axis=0)
        finite_counts = np.maximum(finite.sum(axis=0), 1)
        finite_max = np.where(finite, d, -np.inf).max(axis=0)
        finite_max = np.where(any_finite, finite_max, 0.0)
        penalty = unreachable_penalty * np.mean(~finite, axis=0)
        safe = np.where(finite, d, finite_max[None, :] + penalty[None, :] + 1.0)
        weighted_mean = np.sum(safe * weights[:, None], axis=0)
        scores = alpha * weighted_mean
        scores += beta * np.quantile(safe, 0.90, axis=0)
        scores += gamma * np.max(safe, axis=0)
        scores += penalty
        scores = np.where(any_finite & (finite_counts > 0), scores, np.inf)
        if selected_local:
            pair_to_selected = cs_pair_dist[np.ix_(np.asarray(avail, dtype=int), np.asarray(selected_local, dtype=int))]
            cand_min_pair = np.min(pair_to_selected, axis=1)
            min_pair = np.minimum(current_min_pair, cand_min_pair)
            close = np.isfinite(min_pair) & (min_pair < repulsion_km)
            scores += np.where(close, eta * (repulsion_km - min_pair) * 10.0, 0.0)
        return scores.astype(np.float64, copy=False)

    for _ in range(k):
        if not available:
            break
        scores = batch_scores(available)
        if scores.size == 0 or not np.any(np.isfinite(scores)):
            break
        best_pos = int(np.argmin(scores))
        best_idx = int(available.pop(best_pos))
        if selected_local:
            pair = cs_pair_dist[np.ix_(np.asarray(selected_local, dtype=int), np.asarray([best_idx], dtype=int))]
            if pair.size:
                current_min_pair = min(float(current_min_pair), float(np.min(pair)))
        selected_local.append(best_idx)
        current_nearest = np.minimum(current_nearest, dist_customer_cs[:, best_idx])

    if len(selected_local) < k:
        remaining = [x for x in range(len(candidate_ids)) if x not in selected_local]
        rng.shuffle(remaining)
        selected_local.extend(remaining[: k - len(selected_local)])

    selected = candidate_ids[np.asarray(selected_local[:k], dtype=int)]
    final_d = np.min(dist_customer_cs[:, selected_local[:k]], axis=1) if len(selected_local) else np.full(len(active_customer_ids), np.inf)
    finite_final = np.isfinite(final_d)
    safe_final = final_d[finite_final] if np.any(finite_final) else np.asarray([float("nan")])
    covered_clusters = len(np.unique(cluster_labels[finite_final]))
    metadata = {
        "policy": "graph_facility_location_greedy",
        "candidate_pool_size": int(candidate_ids.size),
        "original_candidate_pool_size": int(original_candidate_pool_size),
        "candidate_truncation_policy": "mean_plus_p90_customer_coverage" if original_candidate_pool_size > int(candidate_ids.size) else "none",
        "selected_count": int(selected.size),
        "mean_customer_to_nearest_cs_km": float(np.mean(safe_final)),
        "p90_customer_to_nearest_cs_km": float(np.quantile(safe_final, 0.90)),
        "max_customer_to_nearest_cs_km": float(np.max(safe_final)),
        "active_cluster_count": int(len(np.unique(cluster_labels))),
        "covered_cluster_count": int(covered_clusters),
        "unreachable_customer_rate_to_selected_cs": float(np.mean(~finite_final)),
        "objective": float(objective(selected_local[:k])) if selected_local else float("inf"),
    }
    return selected.astype(np.int32), metadata
