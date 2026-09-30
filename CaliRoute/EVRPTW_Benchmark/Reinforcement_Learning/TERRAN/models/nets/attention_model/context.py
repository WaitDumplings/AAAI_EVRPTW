"""Problem-specific decoder context used by the TERRAN backbone."""

import torch
from torch import nn


def AutoContext(problem_name, config):
    mapping = {
        "evrptw": VRPContext,
    }
    return mapping[problem_name](**config)


def _gather_by_index(source, index):
    return torch.gather(source, 1, index.unsqueeze(-1).expand(-1, -1, source.size(-1)))


class PrevNodeContext(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim

    def _prev_node_embedding(self, embeddings, state):
        return _gather_by_index(embeddings, state.get_current_node())

    def _state_embedding(self, embeddings, state):
        raise NotImplementedError

    def forward(self, embeddings, state):
        prev_node_embedding = self._prev_node_embedding(embeddings, state)
        state_embedding = self._state_embedding(embeddings, state)
        return torch.cat((prev_node_embedding, state_embedding), -1)


class VRPContext(PrevNodeContext):
    """Original TERRAN EVRPTW context: previous node + load/battery/time."""

    def __init__(self, context_dim):
        super().__init__(context_dim)

    def _state_embedding(self, embeddings, state):
        state_embedding_capacity = state.used_capacity.unsqueeze(-1)
        state_embedding_battery = state.used_battery.unsqueeze(-1)
        state_embedding_time = state.current_time.unsqueeze(-1)
        return torch.cat(
            (state_embedding_capacity, state_embedding_battery, state_embedding_time), dim=-1
        )
