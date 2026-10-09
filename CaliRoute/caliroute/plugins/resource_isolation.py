"""Explicit task resources, with no changes to environment physics or masks.

The host supplies graph_input_context_v1 flags (energy=6, capacity=7,
time=8). Inactive inputs are selected out before learned arithmetic. Raw
feature normalization excludes inactive channels from its statistics and affine
output; no per-module mutable masks are stored, so PPO replay remains stateless.
"""
from __future__ import annotations

import torch


RESOURCE_COLUMNS = {
    'driver': {6: (1, 2, 11), 7: (0,), 8: (3,)},
    'system': {6: (5, 11, 13, 14), 7: (10,), 8: (12,)},
    'candidate': {6: (4, 11, 19, 24, 25, 26, 29),
                  7: (16, 17, 18), 8: (20, 21, 22, 23)},
}


def resource_flags(states):
    graph = states.get('graph_input_context')
    if graph is None:
        raise KeyError('Resource isolation requires graph_input_context with explicit resource flags')
    if graph.ndim != 2 or graph.shape[-1] != 10:
        raise ValueError('graph_input_context must have shape [B,10]')
    return graph[:, 6:9] > .5  # energy, capacity, time; never infer from zeros.


def sanitize_resource_inputs(states):
    """Return a new shallow mapping; input tensors are never mutated."""
    flags = resource_flags(states)
    energy, capacity, time = flags.unbind(-1)
    out = dict(states)
    fields = {
        'edge_energy': (energy, 0.), 'current_battery': (energy, 0.),
        'remaining_battery': (energy, 0.), 'battery_capacity': (energy, 1.),
        'full_charge_time': (energy & time, 0.), 'fixed_full_charge': (energy, 0.),
        'rs_streak_ratio': (energy, 0.), 'cs_visited_current_route': (energy, 0.),
        'edge_time': (time, 0.), 'time_window': (time, 0.),
        'service_time': (time, 0.), 'current_time': (time, 0.),
        'demand': (capacity, 0.), 'current_load': (capacity, 0.),
        'loading_capacity': (capacity, 1.),
    }
    for key, (active, neutral) in fields.items():
        if key in out:
            value = out[key]
            mask = active.reshape(-1, *([1] * (value.ndim - 1)))
            out[key] = torch.where(mask, value, torch.full_like(value, neutral))
    graph = out['graph_input_context']
    active = torch.stack((time, capacity, energy, time, energy, energy & time,
                          torch.ones_like(time), torch.ones_like(time),
                          torch.ones_like(time), energy), -1)
    out['graph_input_context'] = torch.where(active, graph, 0.)
    if 'node_input_context' in out:
        node = out['node_input_context']
        if node.ndim != 3 or node.shape[-1] != 12:
            raise ValueError('node_input_context must have shape [B,N,12]')
        columns = [torch.ones_like(time)] * 12
        columns[4] = columns[5] = time
        columns[6] = columns[7] = energy
        out['node_input_context'] = torch.where(torch.stack(columns, -1)[:, None], node, 0.)
    return out


def feature_resource_mask(states, features, kind):
    flags = resource_flags(states)
    mask = torch.ones((features.shape[0], features.shape[-1]), dtype=torch.bool,
                      device=features.device)
    for flag, columns in RESOURCE_COLUMNS[kind].items():
        mask[:, list(columns)] = flags[:, flag - 6, None]
    return mask.reshape(features.shape[0], *([1] * (features.ndim - 2)), features.shape[-1])


def masked_raw_layer_norm(layer, features, active):
    """LayerNorm over active raw channels only, preserving parameter names."""
    active = active.to(device=features.device, dtype=torch.bool)
    clean = torch.where(active, features, 0.).float()
    count = active.sum(-1, keepdim=True).clamp_min(1).float()
    mean = clean.sum(-1, keepdim=True) / count
    centered = torch.where(active, clean - mean, 0.)
    variance = centered.square().sum(-1, keepdim=True) / count
    value = centered * torch.rsqrt(variance + layer.eps)
    if layer.elementwise_affine:
        value = value * layer.weight.float()
        if layer.bias is not None:
            value = value + layer.bias.float()
    return torch.where(active, value, 0.).to(features.dtype)


def project_raw_features(sequence, features, active=None):
    if active is None:
        return sequence(features)
    value = masked_raw_layer_norm(sequence[0], features, active)
    for module in list(sequence.children())[1:]:
        value = module(value)
    return value
