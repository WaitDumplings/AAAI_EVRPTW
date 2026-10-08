"""Bounded training-only memory of independently verified policy solutions.

Only compact action sequences are persisted. Observations and old-policy log
probabilities are reconstructed for the current policy when a route is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from caliroute.plugins.exploration import (diverse_route_indices, route_structure,
                                           structure_distances)


@dataclass(frozen=True)
class PolicyRoute:
    instance_id: str
    fingerprint: str
    actions: tuple[int, ...]
    objective: float

    @property
    def edges(self):
        nodes = (0,) + self.actions
        return frozenset(zip(nodes, nodes[1:]))


def edge_distance(left: PolicyRoute, right: PolicyRoute) -> float:
    """Directed-edge Jaccard distance, invariant to permutation of vehicle routes."""
    union = left.edges | right.edges
    return 1.0 - len(left.edges & right.edges) / max(len(union), 1)


def require_train_dataset(path: str | Path) -> None:
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    metadata = directory / "metadata.json"
    split = json.loads(metadata.read_text()).get("split") if metadata.exists() else None
    if split is not None:
        if str(split).lower() != "train":
            raise ValueError("Policy route replay requires the train split")
    elif "train" not in {part.lower() for part in directory.parts}:
        raise ValueError("Cannot establish that policy replay dataset is the train split")


class PolicyRoutePool:
    def __init__(self, instances, env_config: dict[str, Any], *, capacity=3,
                 max_relative_gap=0.05, min_edge_distance=0.10,
                 structure_enabled=False, partition_weight=0.5,
                 min_structure_distance=0.10, exploration_capacity=0,
                 exploration_max_relative_gap=0.25, exploration_stagnation_epochs=10):
        self.instances = {str(instance.instance_id): instance for instance in instances}
        self.env_config = dict(env_config)
        self.capacity = int(capacity)
        if not 1 <= self.capacity <= 4:
            raise ValueError("policy_replay_capacity must be between 1 and 4")
        self.max_relative_gap = float(max_relative_gap)
        self.min_edge_distance = float(min_edge_distance)
        if not np.isfinite(self.max_relative_gap) or not np.isfinite(self.min_edge_distance) or self.max_relative_gap < 0 or not 0 <= self.min_edge_distance <= 1:
            raise ValueError("Invalid policy replay quality/diversity thresholds")
        self.routes: dict[str, list[PolicyRoute]] = {}
        self._fingerprints: dict[str, str] = {}
        self.namespace = hashlib.sha256("\n".join(sorted(self.instances)).encode()).hexdigest()
        self.cursor = 0
        self.structure_enabled = bool(structure_enabled)
        self.partition_weight = float(partition_weight)
        self.min_structure_distance = float(min_structure_distance)
        self.exploration_capacity = int(exploration_capacity)
        self.exploration_max_relative_gap = float(exploration_max_relative_gap)
        self.exploration_stagnation_epochs = int(exploration_stagnation_epochs)
        if not 0 <= self.exploration_capacity <= 8 or self.exploration_stagnation_epochs < 0:
            raise ValueError("exploration capacity must be 0..8 and stagnation epochs nonnegative")
        if (not np.isfinite(self.partition_weight) or not 0 <= self.partition_weight <= 1
                or not np.isfinite(self.min_structure_distance) or not 0 <= self.min_structure_distance <= 1
                or not np.isfinite(self.exploration_max_relative_gap) or self.exploration_max_relative_gap < 0):
            raise ValueError("Invalid exploration structure/quality thresholds")
        # These routes are search seeds only, never positive imitation examples.
        self.exploration_routes: dict[str, list[PolicyRoute]] = {}
        self.intake_cursor = 0
        self.best_objectives: dict[str, float] = {}
        self.best_improvement_epoch: dict[str, int] = {}
        self.last_seen_epoch: dict[str, int] = {}
        self.anchor_cursors: dict[str, int] = {}

    def __len__(self):
        return sum(map(len, self.routes.values()))

    def fingerprint(self, instance_id):
        instance_id = str(instance_id)
        if instance_id not in self._fingerprints:
            instance = self.instances[instance_id]
            digest = hashlib.sha256()
            for name in ("depot", "customers", "charging_stations", "distance_matrix_km", "demands_cm3", "service_time_s", "tw_s"):
                value = np.asarray(getattr(instance, name))
                digest.update(name.encode())
                digest.update(str(value.shape).encode())
                digest.update(value.astype(np.float64).tobytes())
            digest.update(json.dumps({"vehicle": instance.vehicle, "speed": instance.speed_profile,
                                      "start": instance.working_start_s, "end": instance.working_end_s,
                                      "charging_mode": self.env_config.get("charging_mode", "fixed_full"),
                                      "max_steps_factor": self.env_config.get("max_steps_factor", 4)},
                                     sort_keys=True, default=str).encode())
            # Preserve the historical fingerprint byte-for-byte when explicit
            # physical matrices are disabled. When enabled, bind archive state
            # to the resolved T/E actually used by the environment, including
            # distance/speed/consumption fallbacks from the shared resolver.
            if bool(self.env_config.get("prefer_explicit_edge_matrices", False)):
                from evrptw_core.physical import resolve_physical_edge_matrices
                physical = resolve_physical_edge_matrices(instance, prefer_explicit_edge_matrices=True)
                digest.update(b"prefer_explicit_edge_matrices=true;physical_edges_v1")
                for name in ("travel_time_s", "energy_kwh"):
                    value = np.asarray(physical[name], dtype=np.float64)
                    digest.update(name.encode())
                    digest.update(str(value.shape).encode())
                    digest.update(value.tobytes())
            self._fingerprints[instance_id] = digest.hexdigest()
        return self._fingerprints[instance_id]

    def replay(self, route: PolicyRoute):
        """Return observations only after every action and final objective verify."""
        if route.instance_id not in self.instances or route.fingerprint != self.fingerprint(route.instance_id):
            return None
        if not route.actions or not np.isfinite(route.objective) or route.objective <= 0:
            return None
        env = make_terran_env(instance=self.instances[route.instance_id], n_traj=1, **self.env_config)
        observation, info = env.reset()
        observations = []
        if len(route.actions) > env.unwrapped.max_steps:
            return None
        for index, action in enumerate(route.actions):
            if action < 0 or action >= observation["action_mask"].shape[-1] or not observation["action_mask"][0, action]:
                return None
            observations.append(observation)
            observation, _, terminated, truncated, info = env.step(np.array([action], dtype=np.int64))
            if bool(terminated[0] or truncated[0]) and index != len(route.actions) - 1:
                return None
        if not bool(np.asarray(info.get("success", [False]))[0]):
            return None
        objective = float(info["objective_distance_km"][0])
        if not np.isclose(objective, route.objective, rtol=1e-6, atol=1e-5):
            return None
        return observations

    def add(self, instance_id, actions, objective) -> bool:
        if not self.structure_enabled and self.exploration_capacity == 0:
            return self._legacy_add(instance_id, actions, objective)
        return self.ingest(instance_id, actions, objective)["elite_added"]

    def _legacy_add(self, instance_id, actions, objective) -> bool:
        instance_id = str(instance_id)
        if instance_id not in self.instances:
            return False
        route = PolicyRoute(instance_id, self.fingerprint(instance_id), tuple(map(int, actions)), float(objective))
        old = self.routes.get(instance_id, [])
        if any(route.actions == previous.actions for previous in old):
            return False
        if not np.isfinite(route.objective) or route.objective <= 0:
            return False
        best = min([route.objective] + [previous.objective for previous in old])
        if route.objective > best * (1 + self.max_relative_gap):
            return False
        if self.replay(route) is None:
            return False
        # Re-select from best to worst, so a better near-duplicate replaces the
        # older route; diverse routes survive only within the quality band.
        selected = []
        for candidate in sorted(old + [route], key=lambda value: (value.objective, value.actions)):
            if candidate.objective > best * (1 + self.max_relative_gap):
                continue
            if any(edge_distance(candidate, previous) < self.min_edge_distance or candidate.edges == previous.edges for previous in selected):
                continue
            selected.append(candidate)
            if len(selected) == self.capacity:
                break
        self.routes[instance_id] = selected
        return route in selected

    def _num_customers(self, instance_id):
        return len(self.instances[str(instance_id)].customers)

    def _structure_signature(self):
        return {"structure_enabled": self.structure_enabled,
                "partition_weight": self.partition_weight,
                "min_structure_distance": self.min_structure_distance,
                "exploration_capacity": self.exploration_capacity,
                "exploration_max_relative_gap": self.exploration_max_relative_gap,
                "exploration_stagnation_epochs": self.exploration_stagnation_epochs,
                "capacity": self.capacity, "max_relative_gap": self.max_relative_gap,
                "min_edge_distance": self.min_edge_distance}

    def ingest(self, instance_id, actions, objective, *, epoch=0):
        """Verify once, then independently admit elite and exploration routes.

        ``routes`` is the only imitation archive. ``exploration_routes`` may
        contain worse feasible structures; the host must use these only for
        separate search rollouts, then verify any improved completion anew.
        """
        rejected = {"verified": False, "elite_added": False, "exploration_added": False}
        instance_id = str(instance_id)
        if instance_id not in self.instances or int(epoch) < 0:
            return rejected
        route = PolicyRoute(instance_id, self.fingerprint(instance_id), tuple(map(int, actions)), float(objective))
        if not route.actions or not np.isfinite(route.objective) or route.objective <= 0:
            return rejected
        old = self.routes.get(instance_id, [])
        old_exploration = self.exploration_routes.get(instance_id, [])
        duplicates = [previous for previous in old + old_exploration if previous.actions == route.actions]
        # Archived routes have already passed a fingerprinted environment replay.
        if duplicates:
            if not any(np.isclose(previous.objective, route.objective, rtol=1e-6, atol=1e-5) for previous in duplicates):
                return rejected
        elif self.replay(route) is None:
            return rejected
        previous_best = min(self.best_objectives.get(instance_id, np.inf),
                            min((previous.objective for previous in old + old_exploration), default=np.inf))
        if route.objective < previous_best - max(abs(route.objective), 1.0) * 1e-9:
            self.best_objectives[instance_id] = route.objective
            self.best_improvement_epoch[instance_id] = int(epoch)
        else:
            self.best_objectives.setdefault(instance_id, previous_best)
            self.best_improvement_epoch.setdefault(instance_id, int(epoch))
        self.last_seen_epoch[instance_id] = max(int(epoch), self.last_seen_epoch.get(instance_id, 0))
        best = self.best_objectives[instance_id]
        customers = self._num_customers(instance_id)
        candidates = list({candidate.actions: candidate for candidate in old + [route]}.values())
        if self.structure_enabled:
            indices = diverse_route_indices([candidate.actions for candidate in candidates],
                                             [candidate.objective for candidate in candidates], customers,
                                             capacity=self.capacity, max_relative_gap=self.max_relative_gap,
                                             min_distance=self.min_structure_distance,
                                             partition_weight=self.partition_weight)
            elite = [candidates[index] for index in indices]
        else:
            elite = []
            for candidate in sorted(candidates, key=lambda value: (value.objective, value.actions)):
                if candidate.objective > best * (1 + self.max_relative_gap):
                    continue
                if any(edge_distance(candidate, previous) < self.min_edge_distance or candidate.edges == previous.edges for previous in elite):
                    continue
                elite.append(candidate)
                if len(elite) == self.capacity:
                    break
        self.routes[instance_id] = elite
        if self.exploration_capacity:
            anchors = [route_structure(candidate.actions, customers) for candidate in elite]
            candidates = list({candidate.actions: candidate for candidate in old_exploration + old + [route]}.values())
            candidates = [candidate for candidate in candidates
                          if candidate.objective <= best * (1 + self.exploration_max_relative_gap)
                          and route_structure(candidate.actions, customers) not in anchors]
            indices = diverse_route_indices([candidate.actions for candidate in candidates],
                                             [candidate.objective for candidate in candidates], customers,
                                             capacity=self.exploration_capacity,
                                             max_relative_gap=self.exploration_max_relative_gap,
                                             min_distance=self.min_structure_distance,
                                             partition_weight=self.partition_weight,
                                             anchors=anchors, prioritize_partition=True)
            self.exploration_routes[instance_id] = [candidates[index] for index in indices]
        return {"verified": True,
                "elite_added": route in elite and route not in old,
                "exploration_added": route in self.exploration_routes.get(instance_id, []) and route not in old_exploration}

    def choose_exploration_anchor(self, instance_id, epoch, *, prefix_fraction=0.25,
                                  max_prefix_steps=32, force=False):
        """Return a bounded prefix for a separate feasibility-masked search.

        Prefixes are not on-policy samples and have no imitation advantage.
        The host must reset to the named instance and replay the prefix with
        action masks, then sample and independently verify the completion.
        """
        if not np.isfinite(prefix_fraction) or not 0 < prefix_fraction < 1 or max_prefix_steps < 1:
            raise ValueError("prefix_fraction must be in (0,1); max_prefix_steps positive")
        instance_id = str(instance_id)
        routes = self.exploration_routes.get(instance_id, [])
        stagnation = max(0, int(epoch) - self.best_improvement_epoch.get(instance_id, int(epoch)))
        if not routes or (not force and stagnation < self.exploration_stagnation_epochs):
            return None
        cursor = self.anchor_cursors.get(instance_id, 0)
        route = routes[cursor % len(routes)]
        self.anchor_cursors[instance_id] = cursor + 1
        # Keep at least one decision free, so the stored route is never simply replayed in full.
        prefix_length = min(int(max_prefix_steps), max(1, int(len(route.actions) * prefix_fraction)), len(route.actions) - 1)
        if prefix_length < 1:
            return None
        best = min(self.routes[instance_id], key=lambda value: value.objective)
        customers = self._num_customers(instance_id)
        edge, partition = structure_distances(route_structure(route.actions, customers), route_structure(best.actions, customers))
        return {"instance_id": instance_id, "actions": route.actions[:prefix_length],
                "full_actions": route.actions, "objective": route.objective,
                "source": "exploration_reservoir", "stagnation_epochs": stagnation,
                "edge_distance": edge, "partition_distance": partition}

    def exploration_diagnostics(self, epoch):
        seen = len(self.best_objectives)
        stagnation = [max(0, int(epoch) - value) for value in self.best_improvement_epoch.values()]
        return {"exploration_pool_routes": float(sum(map(len, self.exploration_routes.values()))),
                "exploration_pool_instances": float(sum(bool(routes) for routes in self.exploration_routes.values())),
                "exploration_instance_coverage": seen / max(len(self.instances), 1),
                "exploration_stagnation_mean_epochs": float(np.mean(stagnation)) if stagnation else 0.0,
                "exploration_stagnant_instances": float(sum(value >= self.exploration_stagnation_epochs for value in stagnation)),
                "exploration_intake_cursor": float(self.intake_cursor)}

    def state_dict(self):
        state = {"version": 1, "namespace": self.namespace, "cursor": self.cursor,
                 "routes": [asdict(route) for routes in self.routes.values() for route in routes]}
        if self.structure_enabled or self.exploration_capacity:
            state.update(version=2, structure_signature=self._structure_signature(),
                         exploration_routes=[asdict(route) for routes in self.exploration_routes.values() for route in routes],
                         intake_cursor=self.intake_cursor, best_objectives=dict(self.best_objectives),
                         best_improvement_epoch=dict(self.best_improvement_epoch),
                         last_seen_epoch=dict(self.last_seen_epoch), anchor_cursors=dict(self.anchor_cursors))
        return state

    def load_state_dict(self, state):
        if state.get("version") not in {1, 2} or state.get("namespace") != self.namespace:
            raise ValueError("Policy replay checkpoint belongs to a different training pool")
        if state["version"] == 2 and state.get("structure_signature") != self._structure_signature():
            raise ValueError("Exploration archive configuration changed on resume")
        self.routes.clear()
        self.exploration_routes.clear()
        self.best_objectives.clear()
        self.best_improvement_epoch.clear()
        self.last_seen_epoch.clear()
        self.anchor_cursors.clear()
        self.cursor = int(state.get("cursor", 0))
        if state["version"] == 1:
            for record in state.get("routes", []):
                route = self._validated_record(record)
                self.add(route.instance_id, route.actions, route.objective)
            return
        # Preserve archive roles exactly on resume, rather than re-running an
        # order-dependent admission algorithm and possibly promoting search seeds.
        for name, limit in (("routes", self.capacity), ("exploration_routes", self.exploration_capacity)):
            destination = getattr(self, name)
            for record in state.get(name, []):
                route = self._validated_record(record)
                if self.replay(route) is None:
                    raise ValueError("Invalid persisted exploration/elite route")
                group = destination.setdefault(route.instance_id, [])
                if len(group) >= limit or any(previous.actions == route.actions for previous in group):
                    raise ValueError("Persisted archive exceeds capacity or has duplicate routes")
                group.append(route)
        self.intake_cursor = int(state.get("intake_cursor", 0))
        for name in ("best_objectives", "best_improvement_epoch", "last_seen_epoch", "anchor_cursors"):
            values = state.get(name, {})
            if any(instance_id not in self.instances or not np.isfinite(value) or value < 0 for instance_id, value in values.items()):
                raise ValueError("Invalid exploration archive metadata")
            setattr(self, name, dict(values))
        for instance_id, routes in self.routes.items():
            best = min(route.objective for route in routes)
            if not np.isclose(self.best_objectives.get(instance_id, np.nan), best, rtol=1e-8, atol=1e-8):
                raise ValueError("Persisted best objective disagrees with verified elite routes")
            if any(route.objective > best * (1 + self.max_relative_gap) for route in routes):
                raise ValueError("Persisted elite route exceeds its quality band")
        for instance_id, routes in self.exploration_routes.items():
            best = self.best_objectives.get(instance_id)
            anchors = [route_structure(route.actions, self._num_customers(instance_id))
                       for route in self.routes.get(instance_id, [])]
            if best is None or any(route.objective > best * (1 + self.exploration_max_relative_gap)
                                   or route_structure(route.actions, self._num_customers(instance_id)) in anchors
                                   for route in routes):
                raise ValueError("Persisted search routes violate quality/role isolation")

    def _validated_record(self, record):
        route = PolicyRoute(str(record["instance_id"]), str(record["fingerprint"]),
                            tuple(map(int, record["actions"])), float(record["objective"]))
        if route.instance_id not in self.instances or route.fingerprint != self.fingerprint(route.instance_id):
            raise ValueError("Policy replay checkpoint instance fingerprint mismatch")
        return route
