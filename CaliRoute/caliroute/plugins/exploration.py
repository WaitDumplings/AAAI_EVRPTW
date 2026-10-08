"""Bounded CPU structure selection for routing exploration, independent of PPO.

The host must verify selected routes before admitting them to an archive. These
utilities do not attach a positive imitation advantage to exploration routes.
Vehicle ordering is ignored; travel edges retain their direction. Charging
stations participate in edges but never in customer co-vehicle partitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RouteStructure:
    edges: frozenset[tuple[int, int]]
    partition: tuple[tuple[int, ...], ...]


def route_structure(actions: Iterable[int], num_customers: int) -> RouteStructure:
    if int(num_customers) < 1:
        raise ValueError("num_customers must be positive")
    sequence = tuple(map(int, actions))
    if any(node < 0 for node in sequence):
        raise ValueError("Route node indices must be nonnegative")
    # Ignore depot self-loops; they do not change either travel or assignment.
    edges = frozenset((left, right) for left, right in zip((0,) + sequence, sequence) if left != right)
    blocks, customers = [], set()
    for node in sequence + (0,):
        if node == 0:
            if customers:
                blocks.append(tuple(sorted(customers)))
                customers = set()
        elif node <= num_customers:
            customers.add(node)
    return RouteStructure(edges, tuple(sorted(blocks)))


def structure_distances(left: RouteStructure, right: RouteStructure) -> tuple[float, float]:
    """Directed-edge and customer co-membership Jaccard distances, each [0, 1].

    Pair intersections are counted from partition contingencies in O(customers)
    space, without materializing an O(customers**2) same-vehicle matrix.
    """
    union = left.edges | right.edges
    edge = 1.0 - len(left.edges & right.edges) / len(union) if union else 0.0
    left_membership = {node: index for index, block in enumerate(left.partition) for node in block}
    intersections = Counter()
    for index, block in enumerate(right.partition):
        for node in block:
            if node in left_membership:
                intersections[(left_membership[node], index)] += 1
    pairs = lambda count: count * (count - 1) // 2
    overlap = sum(pairs(count) for count in intersections.values())
    left_pairs = sum(pairs(len(block)) for block in left.partition)
    right_pairs = sum(pairs(len(block)) for block in right.partition)
    pair_union = left_pairs + right_pairs - overlap
    partition = 1.0 - overlap / pair_union if pair_union else float(left.partition != right.partition)
    return float(edge), float(partition)


def structure_distance(left: RouteStructure, right: RouteStructure, partition_weight: float = 0.5) -> float:
    if not math.isfinite(partition_weight) or not 0 <= partition_weight <= 1:
        raise ValueError("partition_weight must be in [0, 1]")
    edge, partition = structure_distances(left, right)
    return (1.0 - partition_weight) * edge + partition_weight * partition


def diverse_route_indices(actions, objectives, num_customers, *, capacity=3,
                          max_relative_gap=0.25, min_distance=0.10,
                          partition_weight=0.5, quality_weight=1.0,
                          anchors=(), prioritize_partition=False):
    """Greedy quality/diversity subset; returns original indices deterministically.

    The best candidate is retained first without anchors. With anchors (the
    exploration reservoir), novelty relative to those anchors is required.
    ``prioritize_partition`` prefers a changed customer assignment before the
    combined novelty/quality score; it is useful for independent search seeds.
    """
    if capacity < 0 or not math.isfinite(max_relative_gap) or max_relative_gap < 0:
        raise ValueError("capacity and relative quality band must be nonnegative")
    if not math.isfinite(min_distance) or not 0 <= min_distance <= 1:
        raise ValueError("min_distance must be in [0, 1]")
    if not math.isfinite(quality_weight) or quality_weight < 0:
        raise ValueError("quality_weight must be finite and nonnegative")
    if not math.isfinite(partition_weight) or not 0 <= partition_weight <= 1:
        raise ValueError("partition_weight must be in [0, 1]")
    costs = np.asarray(objectives, dtype=np.float64)
    if costs.ndim != 1 or len(actions) != costs.size:
        raise ValueError("actions/objectives must have matching route counts")
    feasible = np.flatnonzero(np.isfinite(costs) & (costs > 0))
    if not capacity or not feasible.size:
        return []
    best = float(costs[feasible].min())
    remaining = [int(index) for index in feasible if costs[index] <= best * (1 + max_relative_gap)]
    structures = {index: route_structure(actions[index], num_customers) for index in remaining}
    reference = list(anchors)
    selected = []
    while remaining and len(selected) < capacity:
        eligible = []
        for index in remaining:
            distances = [structure_distances(structures[index], other) for other in reference]
            duplicate = any(structures[index] == other for other in reference)
            novelty = min(((1 - partition_weight) * edge + partition_weight * part for edge, part in distances), default=1.0)
            if duplicate or (reference and novelty < min_distance):
                continue
            partition_changed = min((part for _, part in distances), default=0.0) > 1e-12
            gap = (float(costs[index]) - best) / best
            score = novelty - quality_weight * gap if reference else -gap
            eligible.append(((int(prioritize_partition and partition_changed), score, -float(costs[index]), -index), index))
        if not eligible:
            break
        index = max(eligible)[1]
        selected.append(index)
        reference.append(structures[index])
        remaining.remove(index)
    return selected


@dataclass(frozen=True)
class RolloutCandidate:
    env_idx: int
    traj_idx: int
    actions: tuple[int, ...]
    objective: float


def select_rollout_candidates(actions, valid, objectives, feasible, *, num_customers,
                              budget=32, cursor=0, per_instance_capacity=3,
                              max_relative_gap=0.25, min_distance=0.10,
                              partition_weight=0.5, quality_weight=1.0):
    """Select a few structures per instance, then rotate the instance budget.

    With 64 instances / budget 32 / capacity 3 this visits about 11 instances,
    yielding multiple structures for each; subsequent calls start at the next
    instance. The cursor is an intake cursor, separate from imitation sampling.
    ``actions`` and ``valid`` are [time, instance, trajectory] CPU arrays.
    """
    actions, valid = np.asarray(actions), np.asarray(valid, dtype=bool)
    costs, feasible = np.asarray(objectives, dtype=np.float64), np.asarray(feasible, dtype=bool)
    if actions.ndim != 3 or actions.shape != valid.shape or costs.shape != actions.shape[1:] or costs.shape != feasible.shape:
        raise ValueError("Expected actions/valid [T,B,K] and objectives/feasible [B,K]")
    if budget < 0 or per_instance_capacity < 1:
        raise ValueError("budget must be nonnegative and per_instance_capacity positive")
    num_envs = costs.shape[0]
    if not num_envs or not budget:
        return [], int(cursor), {"selection_instances": 0.0, "selection_routes": 0.0}
    selected, attempted = [], 0
    for offset in range(num_envs):
        env = (int(cursor) + offset) % num_envs
        attempted += 1
        indices = np.flatnonzero(feasible[env] & np.isfinite(costs[env]) & (costs[env] > 0))
        sequences = [tuple(map(int, actions[:, env, traj][valid[:, env, traj]])) for traj in indices]
        chosen = diverse_route_indices(sequences, costs[env, indices], num_customers,
                                      capacity=min(per_instance_capacity, budget - len(selected)),
                                      max_relative_gap=max_relative_gap, min_distance=min_distance,
                                      partition_weight=partition_weight, quality_weight=quality_weight)
        selected.extend(RolloutCandidate(env, int(indices[index]), sequences[index], float(costs[env, indices[index]])) for index in chosen)
        if len(selected) >= budget:
            break
    next_cursor = (int(cursor) + attempted) % num_envs
    counts = Counter(route.env_idx for route in selected)
    return selected, next_cursor, {
        "selection_instances": float(len(counts)), "selection_routes": float(len(selected)),
        "selection_routes_per_instance": float(np.mean(list(counts.values()))) if counts else 0.0,
        "selection_multi_route_instances": float(sum(count > 1 for count in counts.values())),
        "selection_instance_budget_fraction": len(counts) / num_envs,
    }


def rollout_structure_diagnostics(actions, valid, feasible, *, num_customers, max_pairs=32):
    """Canonical within-rollout coverage, with bounded deterministic pair checks."""
    actions, valid, feasible = np.asarray(actions), np.asarray(valid, bool), np.asarray(feasible, bool)
    if actions.ndim != 3 or actions.shape != valid.shape or actions.shape[1:] != feasible.shape or max_pairs < 1:
        raise ValueError("Invalid rollout diagnostic shapes or pair budget")
    unique, first, edges, partitions, groups = [], [], [], [], 0
    for env in range(actions.shape[1]):
        indices = np.flatnonzero(feasible[env])
        sequences = [tuple(map(int, actions[:, env, index][valid[:, env, index]])) for index in indices]
        if not sequences:
            continue
        structures = [route_structure(sequence, num_customers) for sequence in sequences]
        groups += 1
        unique.append(len(set(structures)) / len(structures))
        first.append(len({next((node for node in sequence if 1 <= node <= num_customers), -1) for sequence in sequences}))
        # Equally spaced triangular-pair indices cover the group without random RNG use.
        count = len(structures)
        pair_count = count * (count - 1) // 2
        targets = set(np.linspace(0, pair_count - 1, min(max_pairs, pair_count), dtype=int)) if pair_count else set()
        pair = 0
        for left in range(count):
            for right in range(left + 1, count):
                if pair in targets:
                    edge, partition = structure_distances(structures[left], structures[right])
                    edges.append(edge)
                    partitions.append(partition)
                pair += 1
    return {"rollout_structure_instances": float(groups),
            "rollout_unique_structure_fraction": float(np.mean(unique)) if unique else 0.0,
            "rollout_first_customer_count": float(np.mean(first)) if first else 0.0,
            "rollout_directed_edge_distance": float(np.mean(edges)) if edges else 0.0,
            "rollout_partition_distance": float(np.mean(partitions)) if partitions else 0.0,
            "rollout_structure_pairs": float(len(edges))}
