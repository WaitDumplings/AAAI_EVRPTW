"""Storage-only expert observation sharing during the ORIGINAL replay builder.

The original builder calls ``np.asarray(value).copy()`` on each observation field.
During that builder only, a NumPy proxy recognizes static arrays from the current
expert environment observation and lets that copy return an immutable shared
snapshot. Dynamic arrays follow the original NumPy path. No forward encoding,
trajectory selection, action sequence, loss, or floating-point values change.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

STATIC_OBSERVATION_KEYS = frozenset({
    'depot_loc', 'cus_loc', 'rs_loc', 'time_window', 'demand', 'service_time',
    'edge_distance', 'edge_time', 'edge_energy', 'battery_capacity', 'loading_capacity',
})


class _SharedCopyArray(np.ndarray):
    def copy(self, order='C'):
        if order not in ('C', 'K', 'A'):
            return np.asarray(self).copy(order=order)
        self._storage.logical_arrays += 1
        self._storage.logical_bytes += self._snapshot.nbytes
        return self._snapshot


class StaticExpertStorage:
    def __init__(self):
        self.current = {}
        self.snapshots = {}
        self.changed_fields = set()
        self.logical_arrays = self.logical_bytes = self.stored_bytes = 0
        self.original_copies = 0
        self.instances = set()

    def observe(self, instance_id, observations):
        # Hold only the CURRENT originals. Holding every input here would defeat
        # the memory bound even if the replay's stored copies were deduplicated.
        self.current.clear()
        instance_id = str(instance_id)
        self.instances.add(instance_id)
        for key in STATIC_OBSERVATION_KEYS.intersection(observations):
            value = observations[key]
            self.current[id(value)] = (value, instance_id, key)
        return observations

    def asarray(self, value, *args, **kwargs):
        match = self.current.get(id(value)) if not args and not kwargs else None
        if match is None or match[0] is not value:
            return np.asarray(value, *args, **kwargs)
        original = np.asarray(value)
        identity = (match[1], match[2])
        if identity in self.changed_fields:
            return original
        snapshot = self.snapshots.get(identity)
        if snapshot is None:
            snapshot = np.array(original, copy=True, order='C')
            snapshot.setflags(write=False)
            self.snapshots[identity] = snapshot
            self.stored_bytes += snapshot.nbytes
        elif snapshot.shape != original.shape or snapshot.dtype != original.dtype or not np.array_equal(snapshot, original, equal_nan=True):
            # Do not assume a named feature is static. If an original environment
            # changes it, preserve its later steps using the original copy path.
            self.changed_fields.add(identity)
            return original
        result = snapshot.view(_SharedCopyArray)
        result._snapshot = snapshot
        result._storage = self
        return result

    def __getattr__(self, name):
        return getattr(np, name)

    def metrics(self):
        return {
            'enabled': True,
            'storage_scope': 'same-instance equal static expert observation arrays; shared read-only while building each original step; dynamic arrays unchanged',
            'forward_encoding_cache': False,
            'expert_samples_removed': 0,
            'static_keys': sorted(STATIC_OBSERVATION_KEYS),
            'instances': len(self.instances),
            'logical_static_arrays': self.logical_arrays,
            'unique_static_arrays': len(self.snapshots),
            'logical_static_bytes': self.logical_bytes,
            'unique_static_bytes': self.stored_bytes,
            'saved_static_bytes': max(0, self.logical_bytes - self.stored_bytes),
            'avoided_static_copies': max(0, self.logical_arrays - len(self.snapshots)),
            'changed_static_fields_preserved_without_sharing': len(self.changed_fields),
        }

    @contextmanager
    def instrument(self, offline_data):
        original_numpy = offline_data.np
        original_load = offline_data._load_terran_runtime
        storage = self

        def load_runtime():
            original_make, second = original_load()

            class ObservedEnvironment:
                def __init__(self, env, instance_id):
                    self._env, self._instance_id = env, instance_id

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def reset(self, *args, **kwargs):
                    observation, info = self._env.reset(*args, **kwargs)
                    return storage.observe(self._instance_id, observation), info

                def step(self, *args, **kwargs):
                    observation, *rest = self._env.step(*args, **kwargs)
                    return (storage.observe(self._instance_id, observation), *rest)

            def make(*args, **kwargs):
                env = original_make(*args, **kwargs)
                instance = kwargs.get('instance') or env.unwrapped.instance
                return ObservedEnvironment(env, instance.instance_id)
            return make, second

        offline_data.np = self
        offline_data._load_terran_runtime = load_runtime
        try:
            yield self
        finally:
            offline_data.np = original_numpy
            offline_data._load_terran_runtime = original_load
            self.current.clear()
