from __future__ import annotations

import numpy as np


def euclidean_edges(points: np.ndarray, edges: list[tuple[int, int]], stretch: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=float)
    edge_arr = np.asarray(edges, dtype=np.int32)
    if edge_arr.size == 0:
        return edge_arr.reshape(0, 2), np.empty(0, dtype=np.float32)
    diff = arr[edge_arr[:, 0]] - arr[edge_arr[:, 1]]
    lengths = np.linalg.norm(diff, axis=1) * float(stretch)
    return edge_arr, lengths.astype(np.float32)


def add_unique_edge(edge_set: set[tuple[int, int]], u: int, v: int) -> None:
    if int(u) == int(v):
        return
    a, b = sorted((int(u), int(v)))
    edge_set.add((a, b))


def connect_knn(points: np.ndarray, node_ids: list[int], k: int, edge_set: set[tuple[int, int]], max_distance: float | None = None) -> None:
    arr = np.asarray(points, dtype=float)
    ids = np.asarray(node_ids, dtype=int)
    if len(ids) <= 1:
        return
    sub = arr[ids]
    dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    for i, u in enumerate(ids):
        order = np.argsort(dist[i])
        added = 0
        for j in order:
            if i == j:
                continue
            if max_distance is not None and dist[i, j] > max_distance:
                continue
            add_unique_edge(edge_set, int(u), int(ids[j]))
            added += 1
            if added >= k:
                break


def nearest_node(point: np.ndarray, candidates: np.ndarray) -> int:
    dist = np.linalg.norm(np.asarray(candidates, dtype=float) - np.asarray(point, dtype=float).reshape(1, 2), axis=1)
    return int(np.argmin(dist))


def sample_truncated_lognormal(rng: np.random.Generator, median: float, sigma: float, low: float, high: float, size: int) -> np.ndarray:
    vals = rng.lognormal(mean=np.log(max(median, 1e-9)), sigma=float(sigma), size=int(size))
    return np.clip(vals, low, high)


def sample_dirichlet_counts(
    rng: np.random.Generator,
    total: int,
    groups: int,
    alpha: float,
    min_count: int = 1,
) -> np.ndarray:
    total = int(total)
    groups = int(groups)
    if groups <= 0:
        raise ValueError("groups must be positive.")
    if total < groups * min_count:
        min_count = max(0, total // groups)
    counts = np.full(groups, int(min_count), dtype=int)
    remaining = total - int(counts.sum())
    if remaining <= 0:
        counts[: total - int(counts.sum())] += 1
        return counts
    p = rng.dirichlet(np.full(groups, float(alpha)))
    extra = rng.multinomial(remaining, p)
    return counts + extra
