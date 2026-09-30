import torch
import torch.nn as nn


def AutoEmbedding(problem_name, config):
    """Select the embedding module for the requested problem."""
    mapping = {
        "evrptw": EVRPTWEmbedding,
    }
    embedding_class = mapping[problem_name]
    return embedding_class(**config)


class EVRPTWEmbedding(nn.Module):
    """TERRAN-compatible EVRPTW embedding.

    The internal node order follows the original TERRAN convention:
    ``[depot, recharging stations, customers]``. EVRPTW-DB observations use
    ``[depot, customers, stations]`` externally and are reordered in
    ``attention_model_wrapper.stateWrapper`` before reaching this module.
    """

    def __init__(self, embedding_dim, extra_dim=3):
        super().__init__()
        self.depot_embedding = nn.Linear(2, embedding_dim)
        self.nodes_embedding = nn.Linear(2 + extra_dim, embedding_dim)
        self.rs_embedding = nn.Linear(2, embedding_dim)
        self.context_dim = embedding_dim + 3

    def forward(self, x):
        depot_loc = x["depot_loc"]
        if depot_loc.dim() == 3 and depot_loc.size(1) == 1:
            depot_loc = depot_loc[:, 0, :]

        demand = x["demand"]
        if demand.dim() == 3 and demand.size(-1) == 1:
            demand = demand.squeeze(-1)

        n_rs = x["rs_loc"].size(1)
        cus_demand = demand[:, 1 + n_rs :].unsqueeze(-1)
        cus_time_window = x["time_window"][:, 1 + n_rs :]

        cus_nodes = self.nodes_embedding(
            torch.cat((x["cus_loc"], cus_demand, cus_time_window), dim=-1)
        )
        rs_nodes = self.rs_embedding(x["rs_loc"])
        depot_node = self.depot_embedding(depot_loc.unsqueeze(1))

        return torch.cat((depot_node, rs_nodes, cus_nodes), dim=-2)
