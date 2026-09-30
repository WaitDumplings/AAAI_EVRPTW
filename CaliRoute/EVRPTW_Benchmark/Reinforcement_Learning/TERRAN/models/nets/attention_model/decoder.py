import torch
import torch.nn as nn

from ...nets.attention_model.context import AutoContext
from ...nets.attention_model.dynamic_embedding import AutoDynamicEmbedding
from ...nets.attention_model.multi_head_attention import AttentionScore, MultiHeadAttention


class Decoder(nn.Module):
    """TERRAN pointer decoder adapted for step-wise Gymnasium rollouts."""

    def __init__(self, embedding_dim, step_context_dim, n_heads, problem, tanh_clipping):
        super().__init__()
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)

        self.context = AutoContext(problem.NAME, {"context_dim": step_context_dim})
        self.dynamic_embedding = AutoDynamicEmbedding(
            problem.NAME, {"embedding_dim": embedding_dim}
        )
        self.glimpse = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.pointer = AttentionScore(
            use_tanh=True,
            C=tanh_clipping,
            learn_scale=True,
            learn_C=True,
        )

        self.decode_type = None
        self.problem = problem

    def forward(self, input, embeddings):
        outputs = []
        sequences = []
        state = self.problem.make_state(input)
        cached_embeddings = self._precompute(embeddings)
        while not state.all_finished():
            log_p, mask = self.advance(cached_embeddings, state)
            action = self.decode(log_p.exp(), mask)
            state = state.update(action)
            outputs.append(log_p)
            sequences.append(action)
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def set_decode_type(self, decode_type):
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def decode(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), (
                "Decode greedy: infeasible action has maximum probability"
            )
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            raise ValueError(f"Unknown decode type: {self.decode_type}")
        return selected

    def _precompute(self, embeddings, mask=None):
        if mask is None:
            graph_embed = embeddings.mean(1)
        else:
            valid_mask = (~mask).to(embeddings.device).unsqueeze(-1).type_as(embeddings)
            graph_embed = (embeddings * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)

        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(embeddings).chunk(
            3, dim=-1
        )
        return embeddings, graph_context, glimpse_key, glimpse_val, logit_key

    def advance(self, cached_embeddings, state, node_mask=None):
        node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K = cached_embeddings

        context = self.context(node_embeddings, state)
        step_context = self.project_step_context(context)
        query = graph_context + step_context

        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.dynamic_embedding(state)
        glimpse_K = glimpse_K + glimpse_key_dynamic
        glimpse_V = glimpse_V + glimpse_val_dynamic
        logit_K = logit_K + logit_key_dynamic

        mask = state.get_mask()
        if node_mask is not None:
            if node_mask.dim() == 2 and mask.dim() == 3:
                node_mask = node_mask.unsqueeze(1)
            mask = mask | node_mask.to(mask.device)

        logits, glimpse = self.calc_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        return logits, glimpse

    def calc_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        glimpse = self.glimpse(query, glimpse_K, glimpse_V, mask)
        logits = self.pointer(glimpse, logit_K, mask)
        return logits, glimpse
