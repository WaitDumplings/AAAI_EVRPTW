"""Physical static fusion and compact directed relations, independent of a host env.

Inputs have already been nondimensionalized by the observation contract. These
modules never reconstruct road costs from coordinates or alter physical masks.
All host-facing residual heads start at zero. Shared edge latents cost O(N²R),
not O(trajectories*N²D); the optional value path aggregates in R before projection.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _zero_output(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def _branch(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), _zero_output(nn.Linear(hidden, outputs)))


def _contexts(states, batch, nodes, reference):
    for name in ('node_input_context', 'graph_input_context'):
        if name not in states:
            raise KeyError(f"Stage-2 physical integration requires {name!r}")
    node = states['node_input_context'].to(reference)
    graph = states['graph_input_context'].to(reference)
    if node.shape != (batch, nodes, 12) or graph.shape != (batch, 10):
        raise ValueError('Stage-2 contexts must have shapes [B,N,12] and [B,10]')
    if not bool(torch.isfinite(node).all() & torch.isfinite(graph).all()):
        raise ValueError('Stage-2 contexts must contain finite values')
    return node, graph


class TypedStaticFusion(nn.Module):
    """Semantic branches + resource-conditioned FiLM and graph-token residual."""

    def __init__(self, embedding_dim, hidden_dim=32):
        super().__init__()
        _positive_int(embedding_dim, 'embedding_dim')
        _positive_int(hidden_dim, 'hidden_dim')
        self.geometry = _branch(2, hidden_dim, embedding_dim)
        self.time = _branch(3, hidden_dim, embedding_dim)
        self.load = _branch(1, hidden_dim, embedding_dim)
        self.road = _branch(12, hidden_dim, embedding_dim)
        self.node_type = _branch(3, hidden_dim, embedding_dim)
        self.resource_film = _branch(10, hidden_dim, 2 * embedding_dim)
        self.graph_token = _branch(10, hidden_dim, embedding_dim)
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, embeddings, observations, states, node_type):
        batch, nodes, _ = embeddings.shape
        node, graph = _contexts(states, batch, nodes, embeddings)
        depot = observations['depot_loc']
        if depot.ndim == 2:
            depot = depot.unsqueeze(1)
        xy = torch.cat((depot, observations['cus_loc'], observations['rs_loc']), dim=1).to(embeddings)
        tw = observations['time_window'].to(embeddings)
        service = observations['service_time'].to(embeddings).reshape(batch, nodes, 1)
        demand = observations['demand'].to(embeddings).reshape(batch, nodes, 1)
        # Time and capacity can be physically inactive. Select dummy inputs
        # out before arithmetic so even nonfinite inactive values cannot leak.
        time_features = torch.where(graph[:, None, 8:9] > .5, torch.cat((tw, service), dim=-1), 0.)
        demand = torch.where(graph[:, None, 7:8] > .5, demand, 0.)
        if not bool(torch.isfinite(xy).all() & torch.isfinite(time_features).all()
                    & torch.isfinite(demand).all()):
            raise ValueError('Typed static features must contain finite active values')
        typed = (self.geometry(xy) + self.time(time_features) + self.load(demand)
                 + self.road(node) + self.node_type(F.one_hot(node_type, 3).to(embeddings)))
        gamma, beta = self.resource_film(graph).chunk(2, dim=-1)
        # Bound multiplicative modulation while keeping exact identity at zero.
        residual = typed + torch.tanh(gamma).unsqueeze(1) * embeddings + beta.unsqueeze(1)
        token = self.graph_token(graph).unsqueeze(1)
        if self.diagnostics_enabled:
            self._diagnostics = {
                'typed_residual_rms': typed.detach().float().square().mean().sqrt(),
                'film_scale_abs_mean': torch.tanh(gamma).detach().float().abs().mean(),
                'residual_rms': residual.detach().float().square().mean().sqrt(),
                'graph_token_rms': token.detach().float().square().mean().sqrt(),
            }
        return embeddings + residual, token

    def diagnostics(self):
        return dict(self._diagnostics)


class DirectedPhysicalRelationEncoder(nn.Module):
    """Fixed-width, asymmetric D/T/E relation with explicit validity flags.

    Forward and reverse normalized costs are independent inputs. Log1p is an
    invertible compression for nonnegative values, not per-instance scaling.
    Invalid road transitions are represented by zeros AND a separate flag.
    Battery feasibility is encoded separately from physical road reachability.
    """
    feature_dim = 26

    def __init__(self, relation_dim=16):
        super().__init__()
        _positive_int(relation_dim, 'edge_relation_dim')
        self.relation_dim = relation_dim
        self.net = nn.Sequential(nn.Linear(self.feature_dim, 2 * relation_dim), nn.SiLU(),
                                 nn.Linear(2 * relation_dim, relation_dim), nn.Tanh())
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, states, node_type):
        parameter = self.net[0].weight
        values = []
        for key in ('edge_distance', 'edge_time', 'edge_energy'):
            if key not in states:
                raise KeyError(f"Directed relations require {key!r}")
            # Keep physical arithmetic in FP32 even for an explicitly half model.
            matrix = states[key].to(device=parameter.device, dtype=torch.float32)
            if matrix.ndim == 2:
                matrix = matrix.unsqueeze(0)
            values.append(matrix)
        distance, travel, energy = values
        batch, nodes, _ = distance.shape
        if any(value.shape != (batch, nodes, nodes) for value in values):
            raise ValueError('Directed relation matrices must have identical [B,N,N] shapes')
        if node_type.shape != (batch, nodes):
            raise ValueError('Directed relation node types must have shape [B,N]')
        _, graph = _contexts(states, batch, nodes, parameter.float())
        time_active = graph[:, None, None, 8] > .5
        battery_active = graph[:, None, None, 6] > .5
        valid = torch.isfinite(distance) & (distance >= 0)
        valid = valid & ((torch.isfinite(travel) & (travel >= 0)) | ~time_active)
        valid = valid & ((torch.isfinite(energy) & (energy >= 0)) | ~battery_active)
        # Inactive dummy quantities must neither change reachability nor yield
        # NaN*0: select them out before log compression or any arithmetic.
        clean = [torch.where(valid, distance, 0),
                 torch.where(valid & time_active, travel, 0),
                 torch.where(valid & battery_active, energy, 0)]
        costs = torch.stack([torch.log1p(value) for value in clean], dim=-1)
        battery = states['battery_capacity'].to(device=parameter.device, dtype=torch.float32).reshape(batch, -1)[:, 0]
        battery_reachable = valid & ((energy <= battery[:, None, None] + 1e-6)
                                     | ~battery_active)
        types = F.one_hot(node_type, 3).to(parameter)
        features = torch.cat((costs, costs.transpose(1, 2), valid.unsqueeze(-1),
                              valid.transpose(1, 2).unsqueeze(-1),
                              battery_reachable.unsqueeze(-1),
                              battery_reachable.transpose(1, 2).unsqueeze(-1),
                              types[:, :, None, :].expand(-1, -1, nodes, -1),
                              types[:, None, :, :].expand(-1, nodes, -1, -1),
                              graph[:, None, None, :].expand(-1, nodes, nodes, -1)), dim=-1)
        relations = self.net(features.to(dtype=parameter.dtype))
        if self.diagnostics_enabled:
            self._diagnostics = {
                'latent_rms': relations.detach().float().square().mean().sqrt(),
                'directed_difference_rms': (relations - relations.transpose(1, 2)).detach().float().square().mean().sqrt(),
                'road_valid_fraction': valid.detach().float().mean(),
            }
        return relations, valid

    def diagnostics(self):
        return dict(self._diagnostics)


class EdgeRelationLayerAdapter(nn.Module):
    """Per-layer light bias/gating; optional R-space values and edge updates.

    The value option explicitly materializes scalar attention weights, even if
    the normal attention path uses SDPA, and therefore has an extra O(H*N²)
    time/memory cost. It never materializes per-edge embedding-dimensional values.
    """
    def __init__(self, relation_dim, embedding_dim, n_heads, use_values=False, use_updates=False):
        super().__init__()
        self.bias = _zero_output(nn.Linear(relation_dim, n_heads))
        self.gate = _zero_output(nn.Linear(relation_dim, n_heads))
        self.value_out = _zero_output(nn.Linear(relation_dim, embedding_dim, bias=False)) if use_values else None
        if use_updates:
            self.endpoint = nn.Linear(embedding_dim, relation_dim, bias=False)
            self.update = _branch(4 * relation_dim, 2 * relation_dim, relation_dim)
        else:
            self.endpoint = self.update = None
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def attention_bias(self, relations):
        raw = self.gate(relations)
        # log(2*sigmoid(raw)), exactly zero at initialization.
        gate = math.log(2.0) - F.softplus(-raw)
        values = self.bias(relations) + gate
        if self.diagnostics_enabled:
            self._diagnostics['bias_rms'] = values.detach().float().square().mean().sqrt()
        return F.pad(values.permute(0, 3, 1, 2), (1, 0, 1, 0))

    def value_message(self, relations, normalized_nodes, attention, attn_bias):
        if self.value_out is None:
            return None
        query = attention.MHA._make_heads(attention.queryEncoder(normalized_nodes))
        key = attention.MHA._make_heads(attention.keyEncoder(normalized_nodes))
        scores = attention.MHA.attentionScore(query, key, mask=None, attn_bias=attn_bias)
        # Include the graph token in normalization, but it has no road-edge value.
        weights = scores.softmax(-1).mean(0)[:, 1:, 1:]
        aggregate = torch.einsum('bij,bijr->bir', weights, relations)
        message = self.value_out(aggregate)
        if self.diagnostics_enabled:
            self._diagnostics['value_message_rms'] = message.detach().float().square().mean().sqrt()
        return F.pad(message, (0, 0, 1, 0))

    def update_relations(self, relations, nodes):
        if self.update is None:
            return relations
        endpoints = self.endpoint(nodes[:, 1:, :])
        count = endpoints.size(1)
        features = torch.cat((relations, relations.transpose(1, 2),
                              endpoints[:, :, None, :].expand(-1, -1, count, -1),
                              endpoints[:, None, :, :].expand(-1, count, -1, -1)), dim=-1)
        delta = self.update(features)
        if self.diagnostics_enabled:
            self._diagnostics['edge_update_rms'] = delta.detach().float().square().mean().sqrt()
        return relations + delta

    def diagnostics(self):
        return dict(self._diagnostics)
