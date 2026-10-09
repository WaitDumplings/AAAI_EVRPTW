"""Compact directed node-edge Transformer for static routing graphs.

Inspired by GRIT pair updates, EGT edge gates and UniteFormer's joint encoding.
This is a new implementation, not a reproduction or a claim of benchmark SOTA.
Physical D/T/E tensors and feasibility masks remain owned by the environment.
Latent road edges use O(B*N*N*R), independent of the number of trajectories;
edge values aggregate at width R before their per-head projection to node width.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


class JointGraphLayer(nn.Module):
    def __init__(self, embedding_dim, edge_dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, embedding_dim // n_heads
        self.node_norm = nn.LayerNorm(embedding_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.edge_bias = nn.Linear(edge_dim, n_heads)
        self.edge_gate = nn.Linear(edge_dim, n_heads)
        self.node_out = nn.Linear(embedding_dim, embedding_dim)
        self.edge_value = nn.Parameter(torch.empty(n_heads, edge_dim, self.head_dim))
        nn.init.xavier_uniform_(self.edge_value)
        self.edge_value_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.ff_norm = nn.LayerNorm(embedding_dim)
        self.ff_in = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.ff_out = nn.Linear(2 * embedding_dim, embedding_dim)
        self.edge_source = nn.Linear(embedding_dim, edge_dim, bias=False)
        self.edge_target = nn.Linear(embedding_dim, edge_dim, bias=False)
        self.edge_self = nn.Linear(edge_dim, edge_dim)
        self.edge_reverse = nn.Linear(edge_dim, edge_dim, bias=False)
        self.edge_score = nn.Linear(n_heads, edge_dim, bias=False)
        self.edge_global = nn.Linear(embedding_dim, edge_dim, bias=False)
        self.edge_update = nn.Linear(edge_dim, edge_dim)
        self.edge_update_gate = nn.Linear(edge_dim, edge_dim)
        # Nonzero residual scales: all new branches can learn from scratch.
        self.attention_scale = nn.Parameter(torch.full((embedding_dim,), .1))
        self.ff_scale = nn.Parameter(torch.full((embedding_dim,), .1))
        self.edge_scale = nn.Parameter(torch.full((edge_dim,), .1))
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, nodes, edges, valid, node_valid, bias):
        batch, total, width = nodes.shape
        h = self.node_norm(nodes)
        e = torch.where(valid[..., None], self.edge_norm(edges), 0.)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def heads(x):
            return x.reshape(batch, total, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = (heads(x) for x in (q, k, v))
        # Keep logits/softmax in FP32 under AMP, including long/unreachable edges.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            content = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        edge_bias = self.edge_bias(e).permute(0, 3, 1, 2)
        scores = content + F.pad(edge_bias.float(), (1, 0, 1, 0))
        if bias is not None:
            scores = scores + bias.float()
        scores = scores.masked_fill(~node_valid[:, None, None, :], -torch.inf)
        weights = scores.softmax(-1)
        # Post-softmax gates preserve edge-strength information (EGT style).
        # Graph-token edges have unit gates, because they are not physical roads.
        gate = 2. * torch.sigmoid(self.edge_gate(e).float()).permute(0, 3, 1, 2)
        gate = torch.where(valid[:, None], gate, 1.)
        gated = weights * F.pad(gate, (1, 0, 1, 0), value=1.)
        node_message = torch.matmul(gated.to(v.dtype), v).transpose(1, 2).reshape(batch, total, width)
        road_weights = gated[:, :, 1:, 1:] * valid[:, None]
        # Never construct [B,H,N,N,D_head] or [B,N,N,D_model] values.
        aggregate = torch.einsum('bhij,bijr->bhir', road_weights.to(e.dtype), e)
        edge_message = torch.einsum('bhir,hrd->bihd', aggregate, self.edge_value.to(aggregate.dtype))
        edge_message = self.edge_value_out(edge_message.reshape(batch, total - 1, width))
        message = self.node_out(node_message) + F.pad(edge_message, (0, 0, 1, 0))
        nodes = nodes + self.attention_scale * message
        value, gate_ff = self.ff_in(self.ff_norm(nodes)).chunk(2, dim=-1)
        nodes = nodes + self.ff_scale * self.ff_out(value * F.silu(gate_ff))
        nodes = torch.where(node_valid[..., None], nodes, 0.)

        # Separate endpoint projections and reverse edges retain directionality.
        updated_h = self.node_norm(nodes)
        source = self.edge_source(updated_h[:, 1:])[:, :, None, :]
        target = self.edge_target(updated_h[:, 1:])[:, None, :, :]
        pair = (self.edge_self(e) + self.edge_reverse(e.transpose(1, 2)) + source + target
                + self.edge_global(updated_h[:, :1])[:, :, None, :]
                + self.edge_score(torch.tanh(content[:, :, 1:, 1:].permute(0, 2, 3, 1)).to(e.dtype)))
        delta = self.edge_update(F.silu(pair)) * torch.sigmoid(self.edge_update_gate(e))
        edges = torch.where(valid[..., None], edges + self.edge_scale * delta, 0.)
        if self.diagnostics_enabled:
            self._diagnostics = {
                'attention_entropy': -(weights.detach() * weights.detach().clamp_min(1e-20).log()).sum(-1).mean(),
                'edge_gate_mean': gate.detach().mean(),
                'edge_value_rms': edge_message.detach().float().square().mean().sqrt(),
                'edge_update_rms': (self.edge_scale * delta).detach().float().square().mean().sqrt(),
            }
        return nodes, edges

    def diagnostics(self):
        return dict(self._diagnostics)


class JointGraphEncoder(nn.Module):
    """Graph-aware input fusion followed by evolving directed pair attention.

    Node order is preserved; returned index zero is the global graph token.
    ``node_mask=True`` denotes padding, NOT the decoder's dynamic action mask.
    Dense node attention is structural communication, not a feasible move.
    """
    def __init__(self, embedding_dim=256, edge_dim=32, n_heads=16, n_layers=2, dropout=0.):
        super().__init__()
        for name, value in [('embedding_dim', embedding_dim), ('edge_dim', edge_dim),
                            ('n_heads', n_heads), ('n_layers', n_layers)]:
            _positive(value, name)
        if embedding_dim % n_heads:
            raise ValueError('embedding_dim must be divisible by n_heads')
        if isinstance(dropout, bool) or dropout != 0.:
            raise ValueError('joint_graph_dropout must be zero for deterministic static caching and PPO replay')
        self.embedding_dim, self.edge_dim, self.n_heads = embedding_dim, edge_dim, n_heads
        self.graph_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.xavier_uniform_(self.graph_token)
        self.global_projection = nn.Sequential(nn.Linear(10, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
        self.input_edge_norm = nn.LayerNorm(edge_dim)
        self.input_road_projection = nn.Sequential(nn.Linear(2 * edge_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
        self.input_scale = nn.Parameter(torch.full((embedding_dim,), .1))
        self.layers = nn.ModuleList([JointGraphLayer(embedding_dim, edge_dim, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.final_edge_norm = nn.LayerNorm(edge_dim)
        self.diagnostics_enabled = False
        self._diagnostics = {}

    def forward(self, node_embeddings, edge_relations, edge_valid, graph_features, *,
                graph_context=None, attn_bias=None, node_mask=None):
        batch, count, width = node_embeddings.shape
        if width != self.embedding_dim or edge_relations.shape != (batch, count, count, self.edge_dim):
            raise ValueError('Joint graph node/edge dimensions do not match encoder configuration')
        if edge_valid.shape != (batch, count, count) or graph_features.shape != (batch, 10):
            raise ValueError('Expected edge_valid [B,N,N] and graph_features [B,10]')
        if graph_context is not None and graph_context.shape != (batch, 1, width):
            raise ValueError('graph_context must have shape [B,1,D]')
        node_valid = torch.ones(batch, count, dtype=torch.bool, device=node_embeddings.device)
        if node_mask is not None:
            if node_mask.shape != (batch, count):
                raise ValueError('node_mask must have shape [B,N]')
            node_valid = ~node_mask.bool()
        valid = edge_valid.bool() & node_valid[:, :, None] & node_valid[:, None, :]
        edges = torch.where(valid[..., None], edge_relations, 0.)
        normalized = torch.where(valid[..., None], self.input_edge_norm(edges), 0.)
        # Exclude self loops from road-neighborhood summaries; isolated nodes get zeros.
        neighbors = valid & ~torch.eye(count, device=valid.device, dtype=torch.bool)[None]
        weighted = torch.where(neighbors[..., None], normalized, 0.)
        outgoing = weighted.sum(2) / neighbors.sum(2).clamp_min(1)[..., None]
        incoming = weighted.sum(1) / neighbors.sum(1).clamp_min(1)[..., None]
        road_input = self.input_road_projection(torch.cat((outgoing, incoming), dim=-1))
        nodes = torch.where(node_valid[..., None], node_embeddings + self.input_scale * road_input, 0.)
        graph = self.graph_token.expand(batch, -1, -1) + self.global_projection(graph_features.to(nodes)).unsqueeze(1)
        if graph_context is not None:
            graph = graph + graph_context
        nodes = torch.cat((graph, nodes), dim=1)
        node_valid = F.pad(node_valid, (1, 0), value=True)
        bias = None
        if attn_bias is not None:
            if attn_bias.ndim == 3:
                attn_bias = attn_bias.unsqueeze(1)
            if (attn_bias.ndim != 4 or attn_bias.shape[0] != batch
                    or attn_bias.shape[-2:] != (count, count) or attn_bias.shape[1] not in (1, self.n_heads)):
                raise ValueError('attn_bias must be [B,N,N] or [B,1|H,N,N]')
            # FP32 finite sentinel avoids -inf/+inf NaNs while allowing global token communication.
            bias = F.pad(torch.nan_to_num(attn_bias.float(), nan=-1e9, posinf=1e9, neginf=-1e9), (1, 0, 1, 0))
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, valid, node_valid, bias)
        nodes = torch.where(node_valid[..., None], self.final_norm(nodes), 0.)
        edges = torch.where(valid[..., None], self.final_edge_norm(edges), 0.)
        if self.diagnostics_enabled:
            self._diagnostics = {
                'input_road_rms': road_input.detach().float().square().mean().sqrt(),
                'edge_latent_rms': edges.detach().float().square().mean().sqrt(),
                'directed_difference_rms': (edges - edges.transpose(1, 2)).detach().float().square().mean().sqrt(),
                'valid_edge_fraction': valid.float().mean(),
            }
        return nodes, edges

    def diagnostics(self):
        return dict(self._diagnostics)
