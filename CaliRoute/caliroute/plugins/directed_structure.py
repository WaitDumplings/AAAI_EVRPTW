"""Compact deterministic road-structure adapters for the original Transformer.

Use authoritative, already scaled road matrices: no coordinate reconstruction,
per-instance rescaling, random neighbours or low-rank random decompositions.
Distance profiles summarize outgoing/incoming roads separately; attention adds
an edge-conditioned nonlinear residual to content scores. Disabled host flags
keep the original architecture and RNG stream intact.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _matrix(value, name):
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f'{name} must have shape [B,N,N]')
    return value.float()


def _active(value, batch, reference, name):
    if value is None:
        return torch.zeros(batch, 1, 1, device=reference.device, dtype=torch.bool)
    value = torch.as_tensor(value, device=reference.device)
    if value.numel() == 1:
        value = value.expand(batch)
    if value.numel() != batch:
        raise ValueError(f'{name} must be scalar or contain B entries')
    return value.reshape(batch, 1, 1) > .5


def _node_valid(distance, node_mask):
    batch, nodes, _ = distance.shape
    if node_mask is None:
        return torch.ones(batch, nodes, device=distance.device, dtype=torch.bool)
    if node_mask.shape != (batch, nodes):
        raise ValueError('node_mask must have shape [B,N]; True denotes padding')
    return ~node_mask.to(device=distance.device, dtype=torch.bool)


def _valid_roads(distance, edge_valid, node_mask):
    nodes = _node_valid(distance, node_mask)
    valid = torch.isfinite(distance) & (distance >= 0)
    valid = valid & nodes[:, :, None] & nodes[:, None, :]
    if edge_valid is not None:
        if edge_valid.shape != distance.shape:
            raise ValueError('edge_valid must have shape [B,N,N]')
        valid = valid & edge_valid.to(device=distance.device, dtype=torch.bool)
    return valid, nodes


def directed_road_profiles(distance, *, edge_valid=None, node_mask=None):
    """Return [B,N,16] fixed-width in/out log-distance distribution summaries.

    The five interpolated quantiles, mean, standard deviation and reachable
    fraction in each direction exclude self edges and padding. Quantiles operate
    on log1p(D), preserving absolute physical scale instead of dividing by N or
    by the instance maximum. Empty neighbourhoods have zero features. No RNG is
    consumed, including when N is smaller than a usual fixed neighbour count.
    """
    distance = _matrix(distance, 'distance')
    batch, count, _ = distance.shape
    valid, nodes = _valid_roads(distance, edge_valid, node_mask)
    valid = valid & ~torch.eye(count, device=distance.device, dtype=torch.bool)[None]
    log_distance = torch.log1p(torch.where(valid, distance, 0.))
    possible = (nodes.sum(-1, keepdim=True) - 1).clamp_min(1).to(distance.dtype)

    def summarize(values, selected):
        size = selected.sum(-1)
        denom = size.clamp_min(1).to(values.dtype)
        clean = torch.where(selected, values, 0.)
        mean = clean.sum(-1) / denom
        variance = torch.where(selected, (clean - mean[..., None]).square(), 0.).sum(-1) / denom
        # sqrt(eps)-sqrt(eps) gives exact 0 for single/constant neighbourhoods
        # without infinite backward derivatives at variance=0.
        std = (variance + 1e-12).sqrt() - 1e-6
        ordered = values.masked_fill(~selected, float('inf')).sort(dim=-1).values
        quantiles = values.new_tensor([.10, .25, .50, .75, .90])
        position = (size - 1).clamp_min(0).unsqueeze(-1) * quantiles
        lower = position.floor().long()
        upper = position.ceil().long()
        lo = ordered.gather(-1, lower)
        hi = ordered.gather(-1, upper)
        # Select invalid infinities out before arithmetic (not NaN * 0).
        lo = torch.where(size[..., None] > 0, lo, 0.)
        hi = torch.where(size[..., None] > 0, hi, 0.)
        result = lo + (hi - lo) * (position - lower)
        return torch.cat((result, mean[..., None], std[..., None],
                          (size.to(values.dtype) / possible)[..., None]), dim=-1)

    result = torch.cat((summarize(log_distance, valid),
                        summarize(log_distance.transpose(1, 2), valid.transpose(1, 2))), dim=-1)
    return torch.where(nodes[..., None], result, 0.)


class DirectedRoadProfileFusion(nn.Module):
    """Zero-initialized gated residual to the host's existing node embedding."""

    def __init__(self, embedding_dim, hidden_dim=32):
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError('embedding_dim and hidden_dim must be positive')
        self.road = nn.Sequential(nn.Linear(16, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, embedding_dim))
        self.gate = nn.Linear(embedding_dim + 16, 1)
        nn.init.zeros_(self.road[-1].weight)
        nn.init.zeros_(self.road[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, embeddings, distance, *, edge_valid=None, node_mask=None):
        if embeddings.ndim != 3 or embeddings.shape[:2] != distance.shape[:2]:
            raise ValueError('embeddings and distance must agree on [B,N]')
        profiles = directed_road_profiles(distance, edge_valid=edge_valid, node_mask=node_mask).to(embeddings)
        valid_nodes = _node_valid(distance, node_mask)
        clean_nodes = torch.where(valid_nodes[..., None], embeddings, 0.)
        gate = torch.sigmoid(self.gate(torch.cat((clean_nodes, profiles), dim=-1)))
        residual = gate * self.road(profiles)
        output = torch.where(valid_nodes[..., None], clean_nodes + residual, 0.)
        if self.diagnostics_enabled:
            self._diagnostics = {
                'profile_rms': profiles.detach().float().square().mean().sqrt(),
                'profile_gate_mean': gate.detach().float().mean(),
                'profile_residual_rms': residual.detach().float().square().mean().sqrt(),
            }
        return output

    def diagnostics(self):
        return dict(self._diagnostics)


def build_directed_pair_features(distance, *, travel_time=None, energy=None,
                                 time_active=None, energy_active=None,
                                 edge_valid=None, node_mask=None):
    """Return [B,N,N,10] directed physical costs with explicit validity/activity.

    Layout: D_ij,D_ji,T_ij,T_ji,E_ij,E_ji,valid_ij,valid_ji,time_on,energy_on.
    Costs use log1p of the host's physical nondimensional values. Inactive T/E
    may be omitted and cannot affect outputs even if supplied as NaN/inf. The
    masks represent road validity, not a vehicle's dynamic feasible-action set.
    """
    distance = _matrix(distance, 'distance')
    batch, nodes, _ = distance.shape
    valid, real_nodes = _valid_roads(distance, edge_valid, node_mask)
    time_on = _active(time_active, batch, distance, 'time_active')
    energy_on = _active(energy_active, batch, distance, 'energy_active')
    optional = []
    for value, active, name in ((travel_time, time_on, 'travel_time'),
                                (energy, energy_on, 'energy')):
        if value is None:
            if bool(active.any()):
                raise ValueError(f'{name} is required when that resource is active')
            value = torch.zeros_like(distance)
        else:
            value = _matrix(value, name).to(distance.device)
            if value.shape != distance.shape:
                raise ValueError(f'{name} must have the same shape as distance')
        valid = valid & (~active | (torch.isfinite(value) & (value >= 0)))
        optional.append(torch.where(active, value, 0.))
    costs = [torch.log1p(torch.where(valid, value, 0.)) for value in (distance, *optional)]
    features = [item for value in costs for item in (value, value.transpose(1, 2))]
    features += [valid.to(distance.dtype), valid.transpose(1, 2).to(distance.dtype),
                 time_on.expand(-1, nodes, nodes).to(distance.dtype),
                 energy_on.expand(-1, nodes, nodes).to(distance.dtype)]
    features = torch.stack(features, dim=-1)
    real_pairs = real_nodes[:, :, None] & real_nodes[:, None, :]
    return torch.where(real_pairs[..., None], features, 0.)


class DirectedContentScoreMixer(nn.Module):
    """Compact nonlinear content/edge score residual inspired by RADAR.

    Unlike a literal per-head concat MLP, edge features generate amplitude and
    shift: residual = amplitude(edge) * tanh(content + shift(edge)). A shared
    hidden width keeps buffers O(B*N*N*(R+H)), with no H*N*N*R or N*N*D values.
    The last layer starts at zero, preserving the host's initial attention.
    Input content is [H,B,N+1,N+1]; physical features omit the graph token.
    """
    feature_dim = 10

    def __init__(self, n_heads, hidden_dim=8):
        super().__init__()
        if n_heads < 1 or hidden_dim < 1:
            raise ValueError('n_heads and hidden_dim must be positive')
        self.n_heads = n_heads
        self.net = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 2 * n_heads))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, content, pair_features):
        heads, batch, query_nodes, key_nodes = content.shape
        nodes = query_nodes - 1
        if heads != self.n_heads or query_nodes != key_nodes:
            raise ValueError('content must have shape [H,B,N+1,N+1]')
        if pair_features.shape != (batch, nodes, nodes, self.feature_dim):
            raise ValueError('directed_pair_features must have shape [B,N,N,10]')
        # Explicit validity suppresses a learned bias on missing/padded roads.
        valid = pair_features[..., 6] > .5
        clean = torch.where(valid[..., None], pair_features, 0.)
        coefficients = self.net(clean.to(self.net[0].weight.dtype)).float()
        amplitude, shift = coefficients.chunk(2, dim=-1)
        amplitude = amplitude.permute(3, 0, 1, 2)
        shift = shift.permute(3, 0, 1, 2)
        residual = amplitude * torch.tanh(content[:, :, 1:, 1:].float() + shift)
        residual = torch.where(valid[None], residual, 0.)
        if self.diagnostics_enabled:
            self._diagnostics['directed_score_rms'] = residual.detach().square().mean().sqrt()
        return content + F.pad(residual.to(content.dtype), (1, 0, 1, 0))

    def diagnostics(self):
        return dict(self._diagnostics)
