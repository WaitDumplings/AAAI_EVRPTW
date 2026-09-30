from __future__ import annotations

import heapq
from collections.abc import Iterable

import numpy as np


def build_adjacency(num_nodes: int, edges: np.ndarray, lengths: np.ndarray) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(int(num_nodes))]
    for (u, v), w in zip(np.asarray(edges, dtype=int), np.asarray(lengths, dtype=float)):
        u_i, v_i = int(u), int(v)
        weight = float(w)
        if weight < 0:
            raise ValueError("Road edge lengths must be non-negative.")
        adjacency[u_i].append((v_i, weight))
        adjacency[v_i].append((u_i, weight))
    return adjacency


def dijkstra_one(adjacency: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[int(source)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist > dist[node]:
            continue
        for nbr, weight in adjacency[node]:
            cand = cur_dist + weight
            if cand < dist[nbr]:
                dist[nbr] = cand
                heapq.heappush(heap, (cand, nbr))
    return dist


def terminal_distance_matrix(
    adjacency: list[list[tuple[int, float]]],
    terminal_node_ids: Iterable[int],
) -> np.ndarray:
    terminals = np.asarray(list(terminal_node_ids), dtype=int)
    out = np.empty((len(terminals), len(terminals)), dtype=np.float32)
    for row, source in enumerate(terminals):
        dist = dijkstra_one(adjacency, int(source))
        out[row] = dist[terminals].astype(np.float32)
    return out


def all_reachable(matrix: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(matrix, dtype=float))))


def floyd_warshall_time(cost: np.ndarray) -> np.ndarray:
    """APSP over a small active terminal transition graph."""
    dist = np.asarray(cost, dtype=np.float64).copy()
    n = dist.shape[0]
    for k in range(n):
        cand = dist[:, k:k + 1] + dist[k:k + 1, :]
        mask = cand < dist
        dist[mask] = cand[mask]
    return dist.astype(np.float32)
