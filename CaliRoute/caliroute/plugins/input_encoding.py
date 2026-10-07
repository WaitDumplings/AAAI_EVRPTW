"""Tensor-only residual encoder for physical node and instance context.

The host converts physical units before calling this module. Inputs are static
within a routing episode: node context [B,N,12], graph context [B,10], in the
host's [depot, customers, stations] order. No customer-count parameter or city
identifier is learned. The v1 dimensions follow caliroute.input_normalization;
this module deliberately has no dependency on an environment or dataset.
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicalInputContextAdapter(nn.Module):
    """Encode directed node relations and physical resource scales separately.

    Returns an additive node-embedding residual [B,N,D]. Both output projections
    start at zero, so adding the module preserves the host's output for the SAME
    observation. Changing the host's coordinate normalization is a separate
    change and is not covered by this identity guarantee.
    """

    node_feature_dim = 12
    graph_feature_dim = 10

    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        for name, value in (("embedding_dim", embedding_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.embedding_dim = embedding_dim
        self.diagnostics_enabled = False
        self._diagnostics = {}
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(self.graph_feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        for branch in (self.node_mlp, self.graph_mlp):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, node_input_context: torch.Tensor,
                graph_input_context: torch.Tensor) -> torch.Tensor:
        if not isinstance(node_input_context, torch.Tensor) or not isinstance(graph_input_context, torch.Tensor):
            raise TypeError("physical input contexts must be tensors")
        node_shape = tuple(node_input_context.shape)
        graph_shape = tuple(graph_input_context.shape)
        if node_input_context.ndim != 3 or node_shape[-1] != self.node_feature_dim:
            raise ValueError(f"node_input_context must have shape [B,N,{self.node_feature_dim}], got {node_shape}")
        if node_shape[0] < 1 or node_shape[1] < 1:
            raise ValueError("node_input_context must contain at least one instance and one node")
        if graph_shape != (node_shape[0], self.graph_feature_dim):
            raise ValueError(
                f"graph_input_context must have shape [{node_shape[0]},{self.graph_feature_dim}], got {graph_shape}"
            )
        if node_input_context.is_complex() or graph_input_context.is_complex():
            raise TypeError("physical input contexts must contain real values")
        parameter = self.node_mlp[0].weight
        nodes = node_input_context.to(device=parameter.device, dtype=parameter.dtype)
        graph = graph_input_context.to(device=parameter.device, dtype=parameter.dtype)
        # One device synchronization per static encode, not per decoder action.
        if not bool(torch.isfinite(nodes).all() & torch.isfinite(graph).all()):
            raise ValueError("physical input contexts must contain only finite values")
        node_residual = self.node_mlp(nodes)
        graph_residual = self.graph_mlp(graph).unsqueeze(1)
        residual = node_residual + graph_residual
        if self.diagnostics_enabled:
            with torch.no_grad():
                self._diagnostics = {
                    "node_features_abs_max": nodes.detach().float().abs().max(),
                    "graph_features_abs_max": graph.detach().float().abs().max(),
                    "node_residual_rms": node_residual.detach().float().square().mean().sqrt(),
                    "graph_residual_rms": graph_residual.detach().float().square().mean().sqrt(),
                    "residual_rms": residual.detach().float().square().mean().sqrt(),
                }
        return residual

    def diagnostics(self):
        return dict(self._diagnostics)
