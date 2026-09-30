from __future__ import annotations

import numpy as np

from evrptw_hierarchy.graph.shortest_path import build_adjacency, dijkstra_one

try:  # SciPy is optional at import time; fall back to Python Dijkstra if absent.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
except Exception:  # pragma: no cover - exercised only on minimal environments.
    csr_matrix = None
    scipy_dijkstra = None


def _build_sparse_adjacency(num_nodes: int, edges: np.ndarray, lengths_km: np.ndarray):
    if csr_matrix is None:
        return None
    # Keep the minimum edge weight for duplicated road links. SciPy sparse
    # construction sums duplicate entries, which would change shortest paths.
    weights: dict[tuple[int, int], float] = {}
    for (u_raw, v_raw), w_raw in zip(np.asarray(edges, dtype=int), np.asarray(lengths_km, dtype=float)):
        u = int(u_raw)
        v = int(v_raw)
        w = float(w_raw)
        if w < 0:
            raise ValueError("Road edge lengths must be non-negative.")
        prev = weights.get((u, v))
        if prev is None or w < prev:
            weights[(u, v)] = w
        prev = weights.get((v, u))
        if prev is None or w < prev:
            weights[(v, u)] = w
    if not weights:
        return csr_matrix((int(num_nodes), int(num_nodes)), dtype=np.float64)
    rows = np.fromiter((key[0] for key in weights.keys()), dtype=np.int32, count=len(weights))
    cols = np.fromiter((key[1] for key in weights.keys()), dtype=np.int32, count=len(weights))
    data = np.fromiter(weights.values(), dtype=np.float64, count=len(weights))
    return csr_matrix((data, (rows, cols)), shape=(int(num_nodes), int(num_nodes)), dtype=np.float64)


class DistanceOracle:
    """Road shortest-path cache for one region board.

    Two modes are supported:

    - source_cache: cache source-to-all Dijkstra vectors on demand.
    - terminal_matrix: precompute shortest distances among depot/customers/CS
      terminals once, then answer active-day matrices by slicing.

    When SciPy is available, Dijkstra is run by scipy.sparse.csgraph in C and
    batched over all missing source nodes. The Python heap implementation remains
    a fallback for environments without SciPy.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: np.ndarray,
        lengths_km: np.ndarray,
        terminal_node_ids: np.ndarray | None = None,
        use_terminal_matrix: bool = False,
    ):
        self.adjacency = build_adjacency(num_nodes, edges, lengths_km)
        self._sparse_adjacency = _build_sparse_adjacency(num_nodes, edges, lengths_km)
        self._cache: dict[int, np.ndarray] = {}
        self.terminal_node_ids = None if terminal_node_ids is None else np.asarray(terminal_node_ids, dtype=np.int32)
        self._terminal_lookup: dict[int, int] = {}
        if self.terminal_node_ids is not None:
            self._terminal_lookup = {int(node): idx for idx, node in enumerate(self.terminal_node_ids.tolist())}
        self._terminal_matrix: np.ndarray | None = None
        self.use_terminal_matrix = bool(use_terminal_matrix and self.terminal_node_ids is not None)
        if self.use_terminal_matrix:
            self.precompute_terminal_matrix()

    @property
    def terminal_matrix_ready(self) -> bool:
        return self._terminal_matrix is not None

    @property
    def terminal_count(self) -> int:
        return 0 if self.terminal_node_ids is None else int(self.terminal_node_ids.size)

    @property
    def terminal_matrix_size_mb(self) -> float:
        if self._terminal_matrix is None:
            return 0.0
        return float(self._terminal_matrix.nbytes / (1024.0 * 1024.0))

    def _scipy_distances(self, source_node_ids: np.ndarray) -> np.ndarray | None:
        if self._sparse_adjacency is None or scipy_dijkstra is None:
            return None
        sources = np.asarray(source_node_ids, dtype=np.int32)
        dist = scipy_dijkstra(
            self._sparse_adjacency,
            directed=False,
            indices=sources,
            return_predecessors=False,
        )
        dist = np.asarray(dist, dtype=np.float64)
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
        return dist

    def precompute_terminal_matrix(self) -> None:
        if self.terminal_node_ids is None:
            raise ValueError("terminal_node_ids are required for terminal matrix precomputation.")
        terminals = self.terminal_node_ids
        dist = self._scipy_distances(terminals)
        if dist is not None:
            self._terminal_matrix = dist[:, terminals].astype(np.float32)
            return
        out = np.empty((terminals.size, terminals.size), dtype=np.float32)
        for row, node_id in enumerate(terminals):
            dist_one = dijkstra_one(self.adjacency, int(node_id))
            out[row] = dist_one[terminals].astype(np.float32)
        self._terminal_matrix = out

    def _terminal_indices(self, node_ids: np.ndarray) -> np.ndarray | None:
        if self._terminal_matrix is None:
            return None
        idx = []
        for node_id in np.asarray(node_ids, dtype=int).tolist():
            value = self._terminal_lookup.get(int(node_id))
            if value is None:
                return None
            idx.append(value)
        return np.asarray(idx, dtype=np.int32)

    def _fill_cache(self, source_node_ids: np.ndarray) -> None:
        unique_sources = list(dict.fromkeys(int(x) for x in np.asarray(source_node_ids, dtype=int).tolist()))
        missing = [node_id for node_id in unique_sources if node_id not in self._cache]
        if not missing:
            return
        batch_dist = self._scipy_distances(np.asarray(missing, dtype=np.int32))
        if batch_dist is not None:
            for node_id, row in zip(missing, batch_dist):
                self._cache[int(node_id)] = np.asarray(row, dtype=np.float64)
            return
        for node_id in missing:
            self._cache[int(node_id)] = dijkstra_one(self.adjacency, int(node_id))

    def distances_from(self, source_node_id: int) -> np.ndarray:
        key = int(source_node_id)
        if key not in self._cache:
            self._fill_cache(np.asarray([key], dtype=np.int32))
        return self._cache[key]

    def matrix_between(self, source_node_ids: np.ndarray, target_node_ids: np.ndarray) -> np.ndarray:
        sources = np.asarray(source_node_ids, dtype=np.int32)
        targets = np.asarray(target_node_ids, dtype=np.int32)
        src_idx = self._terminal_indices(sources)
        tgt_idx = self._terminal_indices(targets)
        if src_idx is not None and tgt_idx is not None:
            return self._terminal_matrix[np.ix_(src_idx, tgt_idx)].astype(np.float32, copy=False)
        self._fill_cache(sources)
        out = np.empty((sources.size, targets.size), dtype=np.float32)
        for row, node_id in enumerate(sources):
            out[row] = self._cache[int(node_id)][targets].astype(np.float32)
        return out

    def matrix(self, terminal_node_ids: np.ndarray) -> np.ndarray:
        terminals = np.asarray(terminal_node_ids, dtype=np.int32)
        return self.matrix_between(terminals, terminals)
