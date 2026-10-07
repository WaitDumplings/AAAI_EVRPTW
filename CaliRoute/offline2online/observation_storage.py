"""Episode-scoped storage sharing; dynamic observations always own their arrays."""
from __future__ import annotations

import numpy as np

STATIC_OBSERVATION_KEYS = frozenset({
    'cus_loc', 'depot_loc', 'rs_loc', 'demand', 'time_window', 'service_time',
    'edge_distance', 'edge_time', 'edge_energy', 'battery_capacity', 'loading_capacity',
    'full_charge_time', 'fixed_full_charge', 'instance_mask',
    "node_input_context", "graph_input_context",
})


def snapshot_observation(observation, static_cache=None):
    result = {}
    for key, value in observation.items():
        if static_cache is not None and key in STATIC_OBSERVATION_KEYS:
            if key not in static_cache:
                static_cache[key] = np.asarray(value).copy()
            result[key] = static_cache[key]
        else:
            result[key] = np.asarray(value).copy()
    return result
