import torch
from torch import nn
from caliroute.plugins.physical_static import EdgeRelationLayerAdapter
from ...nets.graph_model.multi_head_attention import MultiHeadAttentionProj

class GraphBiasBuilder(nn.Module):
    def __init__(self, num_node_types: int = 3):
        super().__init__()
        # 0: depot, 1: RS, 2: customer
        self.type_pair_bias = nn.Embedding(num_node_types * num_node_types, 1)
        nn.init.zeros_(self.type_pair_bias.weight)

        self.dist_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dist_mat, node_type):
        """
        dist_mat:  [B, N, N]
        node_type: [B, N]   (0 depot, 1 RS, 2 customer)

        return:
            attn_bias: [B, N, N]
        """
        # type-pair ids
        type_i = node_type.unsqueeze(2)   # [B,N,1]
        type_j = node_type.unsqueeze(1)   # [B,1,N]
        pair_id = type_i * 3 + type_j     # [B,N,N]

        type_bias = self.type_pair_bias(pair_id).squeeze(-1)  # [B,N,N]

        # simple distance bias: closer = larger
        dist_bias = -self.dist_scale * dist_mat

        attn_bias = type_bias + dist_bias
        return attn_bias

class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        value = self.value_proj(x)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        return self.out_proj(value * gate)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        feed_forward_hidden: int = 512,
        use_sdpa: bool = False,
        use_edge_relation_encoder: bool = False,
        edge_relation_dim: int = 16,
        use_edge_value_messages: bool = False,
        use_edge_state_updates: bool = False,
        use_directed_score_mixer: bool = False,
        directed_score_hidden: int = 8,
    ):
        super().__init__()

        self.attn = MultiHeadAttentionProj(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            use_sdpa=use_sdpa,
            use_directed_score_mixer=use_directed_score_mixer,
            directed_score_hidden=directed_score_hidden,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = SwiGLUFFN(
            embed_dim=embedding_dim,
            hidden_dim=feed_forward_hidden,
        )

        if use_directed_score_mixer and use_edge_value_messages:
            raise ValueError('Directed score mixing and legacy edge value messages cannot be combined')
        self.edge_relation_adapter = None
        if use_edge_relation_encoder:
            # Preserve initialization of every preexisting host parameter.
            with torch.random.fork_rng(devices=[]):
                self.edge_relation_adapter = EdgeRelationLayerAdapter(
                    edge_relation_dim, embedding_dim, n_heads,
                    use_values=use_edge_value_messages, use_updates=use_edge_state_updates,
                )

    def forward(self, x, attn_bias=None, edge_relations=None,
                directed_pair_features=None, node_mask=None):
        # Attention block (Pre-LN); old path executes unchanged when disabled.
        h = self.norm1(x)
        effective_bias = attn_bias
        adapter = self.edge_relation_adapter
        if adapter is not None:
            if edge_relations is None:
                raise ValueError('Edge relation encoder requires edge_relations')
            relation_bias = adapter.attention_bias(edge_relations)
            if effective_bias is not None and effective_bias.dim() == 3:
                effective_bias = effective_bias.unsqueeze(1)
            effective_bias = relation_bias if effective_bias is None else effective_bias + relation_bias
        attention_output = self.attn(h, mask=node_mask, attn_bias=effective_bias,
                                     directed_pair_features=directed_pair_features)
        if adapter is not None and adapter.value_out is not None:
            attention_output = attention_output + adapter.value_message(edge_relations, h, self.attn, effective_bias)
        x = x + attention_output

        # FFN block (Pre-LN)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h

        return x


class GraphAttentionEncoder(nn.Module):
    """
    v1 graph encoder:
    - graph token is always used
    - wrapper passes raw node embeddings [B, N, D]
    - output becomes [B, N+1, D], with graph token at index 0
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        use_sdpa: bool = False,
        use_edge_relation_encoder: bool = False,
        edge_relation_dim: int = 16,
        use_edge_value_messages: bool = False,
        use_edge_state_updates: bool = False,
        use_directed_score_mixer: bool = False,
        directed_score_hidden: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_directed_score_mixer = bool(use_directed_score_mixer)

        self.graph_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads=n_heads,
                    embedding_dim=embed_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    use_sdpa=use_sdpa,
                    use_edge_relation_encoder=use_edge_relation_encoder,
                    edge_relation_dim=edge_relation_dim,
                    use_edge_value_messages=use_edge_value_messages,
                    use_edge_state_updates=use_edge_state_updates,
                    use_directed_score_mixer=use_directed_score_mixer,
                    directed_score_hidden=directed_score_hidden,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def _prepend_graph_token(self, x, attn_bias=None, graph_context=None):
        """
        x: [B, N, D]
        attn_bias: [B, N, N] or [B, 1, N, N] or None
        """
        B, _, D = x.shape
        graph_token = self.graph_token.expand(B, 1, D)   # [B,1,D]
        if graph_context is not None:
            if graph_context.shape != (B, 1, D):
                raise ValueError('graph_context must have shape [B,1,D]')
            graph_token = graph_token + graph_context
        x = torch.cat([graph_token, x], dim=1)           # [B,N+1,D]

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # [B, N, N] -> [B, N+1, N+1]
                B2, N, _ = attn_bias.shape
                new_bias = torch.zeros(
                    B2, N + 1, N + 1,
                    dtype=attn_bias.dtype,
                    device=attn_bias.device
                )
                new_bias[:, 1:, 1:] = attn_bias
                attn_bias = new_bias

            elif attn_bias.dim() == 4:
                # [B, H, N, N] or [B, 1, N, N] -> [B, H, N+1, N+1]
                B2, H, N, _ = attn_bias.shape
                new_bias = torch.zeros(
                    B2, H, N + 1, N + 1,
                    dtype=attn_bias.dtype,
                    device=attn_bias.device
                )
                new_bias[:, :, 1:, 1:] = attn_bias
                attn_bias = new_bias

            else:
                raise ValueError(f"Unsupported attn_bias shape: {attn_bias.shape}")

        return x, attn_bias

    def forward(self, x, mask=None, attn_bias=None, graph_context=None,
                edge_relations=None, return_edge_relations=False,
                directed_pair_features=None):
        """
        x: [B, N, D]
        mask: legacy path ignores; directed mixer interprets True as padded node
        attn_bias: [B, N, N] or [B, H, N, N]
        """
        node_mask = None
        if self.use_directed_score_mixer and mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError('mask must have shape [B,N] with True for padding')
            x = torch.where(mask[..., None].bool(), 0., x)
            node_mask = torch.nn.functional.pad(mask.bool(), (1, 0), value=False)
        x, attn_bias = self._prepend_graph_token(x, attn_bias=attn_bias, graph_context=graph_context)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, edge_relations=edge_relations,
                      directed_pair_features=directed_pair_features, node_mask=node_mask)
            if layer.edge_relation_adapter is not None:
                edge_relations = layer.edge_relation_adapter.update_relations(edge_relations, x)

        x = self.final_norm(x)   # [B, N+1, D]
        if node_mask is not None:
            x = torch.where(node_mask[..., None], 0., x)
        return (x, edge_relations) if return_edge_relations else x

    @staticmethod
    def _mean_without_graph_token(x):
        """
        x: [B, N+1, D], graph token at index 0
        """
        return x[:, 1:, :].mean(dim=1)
