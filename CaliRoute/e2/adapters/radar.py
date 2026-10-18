"""RADAR decoder boundary for E1 AGDA action-key/bias residuals.

The native SVD/Sinkhorn encoder and native glimpse are retained. Host feature
packing and the native training loop are intentionally outside this boundary.
The adapter must be installed BEFORE optimizer construction. No long-run
launcher is enabled until environment/feature/replay acceptance is complete.
"""
import torch
from torch import nn
from caliroute.plugins.agda import AdaptiveGraphAttention


class RadarAGDADecoder(nn.Module):
    def __init__(self, native_decoder, *, enabled=False, candidate_feature_dim=30,
                 system_feature_dim=15):
        super().__init__()
        self.native = native_decoder
        self.enabled = bool(enabled)
        width = native_decoder.model_params['embedding_dim']
        self.agda = AdaptiveGraphAttention(width, enabled=True,
            enable_delta_k=False, enable_delta_v=False,
            enable_delta_action_key=True, enable_action_bias=True,
            optimize_dynamic_projections=True, use_agda_v2=False,
            candidate_feature_dim=candidate_feature_dim,
            system_feature_dim=system_feature_dim)

    def forward(self, encoded_q0, load, ninf_mask, *, node_embeddings=None,
                decision_tokens=None, candidate_features=None, system_features=None):
        if not self.enabled:
            return self.native(encoded_q0, load, ninf_mask)
        if any(x is None for x in (node_embeddings, decision_tokens, candidate_features, system_features)):
            raise ValueError('Enabled AGDA requires explicit host features and tokens')
        batch, trajectories, nodes = ninf_mask.shape
        if candidate_features.shape[:3] != (batch, trajectories, nodes):
            raise ValueError('Candidate features must match native batch/trajectory/node order')
        if node_embeddings.shape[:2] != (batch, nodes):
            raise ValueError('Node embedding order/shape mismatch')
        _, _, action_key, bias = self.agda(node_embeddings, decision_tokens,
            candidate_features, system_features=system_features)
        native = self.native
        heads = native.model_params['head_num']
        query = native.Wq_0(torch.cat((encoded_q0, load[..., None]), -1))
        query = query.reshape(batch, trajectories, heads, -1).transpose(1, 2)
        glimpse = native.multi_head_combine(native._multi_head_attention(
            query, native.k, native.v, rank3_ninf_mask=ninf_mask))
        # Retain the native action key and add only a candidate-conditioned delta.
        scores = torch.matmul(glimpse, native.single_head_key)
        scores = scores + torch.einsum('btd,btnd->btn', glimpse, action_key)
        scaled = scores / native.model_params['sqrt_embedding_dim'] + bias
        clipped = native.model_params['logit_clipping'] * torch.tanh(scaled)
        return torch.softmax(clipped + ninf_mask, dim=-1)
