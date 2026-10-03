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
    def __init__(self, instances, env_config: dict[str, Any], *, capacity=3, max_relative_gap=0.05, min_edge_distance=0.10):
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

    def state_dict(self):
        return {"version": 1, "namespace": self.namespace, "cursor": self.cursor,
                "routes": [asdict(route) for routes in self.routes.values() for route in routes]}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("namespace") != self.namespace:
            raise ValueError("Policy replay checkpoint belongs to a different training pool")
        self.routes.clear()
        self.cursor = int(state.get("cursor", 0))
        for record in state.get("routes", []):
            route = PolicyRoute(str(record["instance_id"]), str(record["fingerprint"]),
                                tuple(map(int, record["actions"])), float(record["objective"]))
            if route.instance_id not in self.instances or route.fingerprint != self.fingerprint(route.instance_id):
                raise ValueError("Policy replay checkpoint instance fingerprint mismatch")
            # Validate persisted actions again rather than trusting checkpoint metadata.
            self.add(route.instance_id, route.actions, route.objective)
