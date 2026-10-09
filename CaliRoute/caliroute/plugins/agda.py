"""Tensor-only candidate-conditioned gates for dynamic graph residuals.

Contract: candidate features [...,N,F] plus (key, value, action-key, action-bias)
residuals -> a tuple with identical shapes. Embedding residuals have a trailing
embedding dimension and bias has no embedding dimension. Numeric zero/None
represent disabled branches. The host owns feature construction and hard masks.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .resource_isolation import project_raw_features


class AdaptiveGraphDecisionAdapter(nn.Module):
    """Identity-initialized, bounded per-candidate residual modulation.

    Existing DDE has one learned scalar per branch. These gates can suppress or
    strengthen a branch according to current candidate state, without changing
    a pretrained policy at initialization. Bounds prevent an initially small
    DDE correction from acquiring arbitrarily large multiplicative gain.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, max_modulation: float = 0.5):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or not 0 < max_modulation < 1:
            raise ValueError("positive dimensions and 0 < max_modulation < 1 are required")
        self.feature_dim = int(feature_dim)
        self.max_modulation = float(max_modulation)
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.diagnostics_enabled = False
        self._diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, features: torch.Tensor, residuals: tuple, feature_mask=None) -> tuple:
        if features.shape[-1] != self.feature_dim or len(residuals) != 4:
            raise ValueError("expected configured feature width and four residual branches")
        clean = torch.nan_to_num(features, nan=0., posinf=10., neginf=-10.).clamp(-10., 10.)
        gates = 1.0 + self.max_modulation * torch.tanh(project_raw_features(self.net, clean, feature_mask))
        output = tuple(
            value * (gates[..., index, None] if index < 3 else gates[..., index])
            if torch.is_tensor(value) else value
            for index, value in enumerate(residuals)
        )
        if self.diagnostics_enabled:
            with torch.no_grad():
                self._diagnostics = {
                    'feature_rms': clean.float().square().mean().sqrt().detach(),
                    'feature_abs_max': clean.abs().amax().detach(),
                    'gate_mean': gates.mean().detach(), 'gate_std': gates.std(unbiased=False).detach(),
                    'gate_deviation': (gates - 1.).abs().mean().detach(),
                    'gate_saturation': ((gates - 1.).abs() > .95 * self.max_modulation).float().mean().detach(),
                }
                for name, original, value in zip(('key', 'value', 'action_key', 'action_bias'), residuals, output):
                    if torch.is_tensor(value):
                        self._diagnostics[f'{name}_residual_rms'] = value.float().square().mean().sqrt().detach()
                        self._diagnostics[f'{name}_modulation_to_residual'] = (value - original).float().square().mean().sqrt() / original.float().square().mean().sqrt().clamp_min(1e-6)
        return output

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Detached device tensors, with no CPU synchronization in forward."""
        return dict(self._diagnostics)


class AdaptiveGraphAttention(nn.Module):
    """Complete tensor-only AGDA fusion core, reusable with another backbone.

    The routing wrapper constructs problem-specific features and summary tokens;
    this module owns attention, candidate fusion and all residual projections.
    Attribute names intentionally match the earlier CaliRoute DDE checkpoints.
    """

    def __init__(
        self,
        embedding_dim,
        n_heads=4,
        enabled=False,
        enable_delta_k=True,
        enable_delta_v=True,
        enable_delta_action_key=True,
        enable_action_bias=True,
        optimize_dynamic_projections=False,
        use_agda_v2=False,
        agda_hidden_dim=32,
        candidate_feature_dim=30,
        system_feature_dim=15,
        num_tokens=9,
        use_resource_isolation=False,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.use_resource_isolation = bool(use_resource_isolation)
        self.enabled = bool(enabled)
        self.enable_delta_k = bool(enable_delta_k)
        self.enable_delta_v = bool(enable_delta_v)
        self.enable_delta_action_key = bool(enable_delta_action_key)
        self.enable_action_bias = bool(enable_action_bias)
        self.optimize_dynamic_projections = bool(optimize_dynamic_projections)
        self.routing_system_feature_dim = 10
        self.problem_system_feature_dim = 5
        self.system_feature_dim = int(system_feature_dim)
        self.routing_candidate_feature_dim = 16
        self.problem_candidate_feature_dim = 14
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.num_tokens = int(num_tokens)

        n_heads = max(1, int(n_heads))
        if self.embedding_dim % n_heads != 0:
            n_heads = 1

        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.system_feature_dim),
            nn.Linear(self.system_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(1, self.num_tokens, embedding_dim))
        self.token_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.token_ff = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.token_ff_norm = nn.LayerNorm(embedding_dim)
        self.route_pos_proj = nn.Sequential(
            nn.LayerNorm(4 * embedding_dim),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.node_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.decision_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.step_state_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.candidate_feature_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 3 * embedding_dim),
        )
        self.candidate_delta_base = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
        )
        self.candidate_key_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.candidate_value_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.candidate_action_key_delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.action_bias_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.key_scale = nn.Parameter(torch.tensor(0.1))
        self.value_scale = nn.Parameter(torch.tensor(0.1))
        self.action_key_scale = nn.Parameter(torch.tensor(0.1))
        self.action_bias_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.token_type, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.state_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[1].bias)
        nn.init.xavier_uniform_(self.state_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.state_proj[3].bias)
        nn.init.xavier_uniform_(self.route_pos_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.route_pos_proj[1].bias)
        nn.init.zeros_(self.route_pos_proj[3].weight)
        nn.init.zeros_(self.route_pos_proj[3].bias)
        nn.init.zeros_(self.node_state_proj.weight)
        nn.init.zeros_(self.decision_state_proj.weight)
        nn.init.zeros_(self.step_state_proj.weight)
        nn.init.xavier_uniform_(self.candidate_feature_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.candidate_feature_proj[1].bias)
        nn.init.zeros_(self.candidate_feature_proj[3].weight)
        nn.init.zeros_(self.candidate_feature_proj[3].bias)
        nn.init.xavier_uniform_(self.candidate_delta_base[1].weight, gain=0.5)
        nn.init.zeros_(self.candidate_delta_base[1].bias)
        nn.init.zeros_(self.candidate_key_delta_proj.weight)
        nn.init.zeros_(self.candidate_key_delta_proj.bias)
        nn.init.zeros_(self.candidate_value_delta_proj.weight)
        nn.init.zeros_(self.candidate_value_delta_proj.bias)
        nn.init.zeros_(self.candidate_action_key_delta_proj.weight)
        nn.init.zeros_(self.candidate_action_key_delta_proj.bias)
        nn.init.xavier_uniform_(self.action_bias_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.action_bias_proj[1].bias)
        nn.init.zeros_(self.action_bias_proj[3].weight)
        nn.init.zeros_(self.action_bias_proj[3].bias)

        self.agda_adapter = None
        if use_agda_v2:
            with torch.random.fork_rng(devices=[]):
                self.agda_adapter = AdaptiveGraphDecisionAdapter(self.candidate_feature_dim, agda_hidden_dim)

    def project_enabled(self, layer, value):
        """Keep original parameter tensors, but compute only enabled output rows."""
        flags = (self.enable_delta_k, self.enable_delta_v, self.enable_delta_action_key)
        if not self.optimize_dynamic_projections or all(flags):
            return layer(value).chunk(3, dim=-1)
        width = self.embedding_dim
        return tuple(
            F.linear(value, layer.weight[i * width:(i + 1) * width]) if enabled else None
            for i, enabled in enumerate(flags)
        )

    def precompute_node_projections(self, node_embeddings):
        if not self.enabled or not self.optimize_dynamic_projections:
            return None
        if not (self.enable_delta_k or self.enable_delta_v or self.enable_delta_action_key):
            return None
        return self.project_enabled(self.node_state_proj, node_embeddings)

    def forward(self, node_embeddings, decision_tokens, candidate_features,
                system_features=None, state_token=None, node_projections=None,
                candidate_feature_mask=None, system_feature_mask=None, token_mask=None):
        """Return dynamic K/V/action-key/bias residuals.

        node_embeddings: [B,N,D], decision_tokens: [B,T,S,D], S <= num_tokens.
        The first token is the decision token. candidate_features: [B,T,N,F].
        Pass state_token [B,T,D], or system_features [B,T,G] to embed here.
        In bias-only mode decision_tokens/state features may be None.
        """
        if not self.enabled:
            return 0, 0, 0, 0
        if candidate_features.ndim != 4 or candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate_features must be [B,T,N,configured_feature_dim]")
        if not (self.enable_delta_k or self.enable_delta_v or self.enable_delta_action_key):
            bias = project_raw_features(self.action_bias_proj, candidate_features, candidate_feature_mask).squeeze(-1) if self.enable_action_bias else 0
            residuals = (0, 0, 0, torch.tanh(self.action_bias_scale) * bias)
            return self.agda_adapter(candidate_features, residuals, feature_mask=candidate_feature_mask) if self.agda_adapter is not None else residuals
        if state_token is None:
            if system_features is None:
                raise ValueError("state_token or system_features is required for embedding residuals")
            state_token = project_raw_features(self.state_proj, system_features, system_feature_mask)
        if decision_tokens is None or decision_tokens.ndim != 4 or decision_tokens.shape[2] > self.num_tokens:
            raise ValueError("decision_tokens must be [B,T,S,D], S <= num_tokens")
        tokens = decision_tokens
        B, T, S, D = tokens.shape
        flat_tokens = tokens.reshape(B * T, S, D)
        flat_tokens = flat_tokens + self.token_type[:, :S, :].to(
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        )
        attended_tokens, _ = self.token_attn(
            flat_tokens,
            flat_tokens,
            flat_tokens,
            need_weights=False,
            key_padding_mask=token_mask.reshape(B * T, S) if token_mask is not None else None,
        )
        flat_tokens = self.token_norm(flat_tokens + attended_tokens)
        flat_tokens = self.token_ff_norm(flat_tokens + self.token_ff(flat_tokens))
        decision_token = flat_tokens[:, 0, :].reshape(B, T, D)

        key_delta = 0
        value_delta = 0
        action_key_delta = 0
        if self.enable_delta_k or self.enable_delta_v or self.enable_delta_action_key:
            candidate_base = project_raw_features(self.candidate_delta_base, candidate_features, candidate_feature_mask)
            if node_projections is None:
                node_projections = self.project_enabled(self.node_state_proj, node_embeddings)
            node_key, node_value, node_action_key = node_projections
            decision_key, decision_value, decision_action_key = self.project_enabled(self.decision_state_proj, decision_token)
            step_key, step_value, step_action_key = self.project_enabled(self.step_state_proj, state_token)
            if self.enable_delta_k:
                key_delta = self.candidate_key_delta_proj(candidate_base)
                key_delta = key_delta + node_key.unsqueeze(1)
                key_delta = key_delta + decision_key.unsqueeze(2)
                key_delta = key_delta + step_key.unsqueeze(2)
                key_delta = torch.tanh(self.key_scale) * key_delta
            if self.enable_delta_v:
                value_delta = self.candidate_value_delta_proj(candidate_base)
                value_delta = value_delta + node_value.unsqueeze(1)
                value_delta = value_delta + decision_value.unsqueeze(2)
                value_delta = value_delta + step_value.unsqueeze(2)
                value_delta = torch.tanh(self.value_scale) * value_delta
            if self.enable_delta_action_key:
                action_key_delta = self.candidate_action_key_delta_proj(candidate_base)
                action_key_delta = action_key_delta + node_action_key.unsqueeze(1)
                action_key_delta = action_key_delta + decision_action_key.unsqueeze(2)
                action_key_delta = action_key_delta + step_action_key.unsqueeze(2)
                action_key_delta = torch.tanh(self.action_key_scale) * action_key_delta
        if self.enable_action_bias:
            action_bias = project_raw_features(self.action_bias_proj, candidate_features, candidate_feature_mask).squeeze(-1)
            action_bias = torch.tanh(self.action_bias_scale) * action_bias
        else:
            action_bias = 0
        residuals = (key_delta, value_delta, action_key_delta, action_bias)
        return self.agda_adapter(candidate_features, residuals, feature_mask=candidate_feature_mask) if self.agda_adapter is not None else residuals
