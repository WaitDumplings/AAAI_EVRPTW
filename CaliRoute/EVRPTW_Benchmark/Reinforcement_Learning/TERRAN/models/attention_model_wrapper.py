from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .nets.attention_model.decoder import Decoder
from .nets.attention_model.embedding import AutoEmbedding
from .nets.attention_model.encoder import GraphAttentionEncoder


class Problem:
    def __init__(self, name: str):
        self.NAME = name


def _to_tensor(value: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def prepare_observation_batch(obs: dict[str, Any]) -> dict[str, Any]:
    """Add the outer environment batch dimension expected by TERRAN."""
    out: dict[str, Any] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if key in {"cus_loc", "rs_loc"}:
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key == "depot_loc":
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key in {
            "demand",
            "service_time",
            "last_node_idx",
            "current_load",
            "current_battery",
            "remaining_battery",
            "current_time",
        }:
            out[key] = arr[None, ...] if arr.ndim == 1 else value
        elif key == "time_window":
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key == "action_mask":
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key in {
            "visited_customers_ratio",
            "visited_customers_raio",
            "remain_feasible_customers_ratio",
            "remain_feasible_customers_raio",
        }:
            out[key] = arr[None, ...] if arr.ndim == 2 else value
        elif key in {"battery_capacity", "loading_capacity"}:
            out[key] = arr[None, ...] if arr.ndim == 1 else value
        else:
            out[key] = value
    if "visited_customers_raio" not in out and "visited_customers_ratio" in out:
        out["visited_customers_raio"] = out["visited_customers_ratio"]
    if "remain_feasible_customers_raio" not in out and "remain_feasible_customers_ratio" in out:
        out["remain_feasible_customers_raio"] = out["remain_feasible_customers_ratio"]
    return out


class Backbone(nn.Module):
    """Original TERRAN backbone with EVRPTW-DB observation adapters.

    EVRPTW-DB exposes nodes as ``[depot, customers, stations]``. The original
    TERRAN model embeds/decodes nodes as ``[depot, stations, customers]``.
    ``stateWrapper`` performs that internal permutation and logits are converted
    back before they leave the backbone.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        problem_name: str = "evrptw",
        n_encode_layers: int = 3,
        tanh_clipping: float = 15.0,
        n_heads: int = 16,
        device: str | torch.device = "cpu",
        use_graph_token: bool = False,
        use_dynamic_embedding: bool = False,
    ):
        super().__init__()
        del use_graph_token, use_dynamic_embedding
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )
        self.decoder = Decoder(
            embedding_dim,
            self.embedding.context_dim,
            n_heads,
            self.problem,
            tanh_clipping,
        )

    def forward(self, obs, use_mask: bool = False):
        obs = prepare_observation_batch(obs)
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        mask = state.external_mask_to_internal(obs["instance_mask"]) if use_mask and "instance_mask" in obs else None
        embedding = self.embedding(state.states["observations"])
        encoded_inputs, _ = self.encoder(embedding, mask=mask)
        cached_embeddings = self.decoder._precompute(encoded_inputs, mask=mask)
        logits, glimpse = self.decoder.advance(cached_embeddings, state, node_mask=mask)
        return state.logits_to_external(logits), glimpse

    def encode(self, obs, use_mask: bool = False):
        obs = prepare_observation_batch(obs)
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        mask = state.external_mask_to_internal(obs["instance_mask"]) if use_mask and "instance_mask" in obs else None
        embedding = self.embedding(state.states["observations"])
        encoded_inputs, _ = self.encoder(embedding, mask=mask)
        return self.decoder._precompute(encoded_inputs, mask=mask)

    def decode(self, obs, cached_embeddings):
        obs = prepare_observation_batch(obs)
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)
        return state.logits_to_external(logits), glimpse


class Actor(nn.Module):
    def forward(self, x):
        return x[0]


def orthogonal_init(layer, gain: float = 1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Critic(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

    def forward(self, x):
        return self.mlp(x[1])


class Agent(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        tanh_clipping: float = 15,
        n_encode_layers: int = 3,
        device: str | torch.device = "cpu",
        name: str = "evrptw",
        use_graph_token: bool = False,
        use_dynamic_embedding: bool = False,
    ):
        super().__init__()
        self.backbone = Backbone(
            embedding_dim=embedding_dim,
            device=device,
            tanh_clipping=tanh_clipping,
            n_encode_layers=n_encode_layers,
            problem_name=name,
            use_graph_token=use_graph_token,
            use_dynamic_embedding=use_dynamic_embedding,
        )
        self.critic = Critic(hidden_size=embedding_dim)
        self.actor = Actor()

    def forward(self, x, use_mask: bool = False, return_logits: bool = True):
        x = self.backbone(x, use_mask)
        logits = self.actor(x)
        action = logits.max(2)[1]
        if not return_logits:
            return action
        return action, logits

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.backbone(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_acction_and_value(self, x, action=None):
        return self.get_action_and_value(x, action=action)

    def get_value_cached(self, x, state):
        x = self.backbone.decode(x, state)
        return self.critic(x)

    def get_action_and_value_cached(self, x, action=None, state=None, print_probs: bool = False):
        del print_probs
        if state is None:
            state = self.backbone.encode(x)
        x = self.backbone.decode(x, state)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic((x[0], x[1]))
        return action, probs.log_prob(action), probs.entropy(), value, state


class stateWrapper:
    def __init__(self, states, device, problem: str = "evrptw"):
        self.device = device
        states = prepare_observation_batch(states)
        self.states = {k: _to_tensor(v, device=self.device) for k, v in states.items()}
        if problem != "evrptw":
            return

        n_cus = int(self.states["cus_loc"].size(1))
        n_rs = int(self.states["rs_loc"].size(1))
        n_nodes = 1 + n_cus + n_rs
        device_t = self.states["cus_loc"].device

        depot = torch.zeros(1, dtype=torch.long, device=device_t)
        customer_external = torch.arange(1, 1 + n_cus, dtype=torch.long, device=device_t)
        rs_external = torch.arange(1 + n_cus, n_nodes, dtype=torch.long, device=device_t)
        self.internal_to_external = torch.cat((depot, rs_external, customer_external), dim=0)
        self.external_to_internal = torch.empty(n_nodes, dtype=torch.long, device=device_t)
        self.external_to_internal[self.internal_to_external] = torch.arange(
            n_nodes, dtype=torch.long, device=device_t
        )

        depot_loc = self.states["depot_loc"].float()
        if depot_loc.dim() == 3 and depot_loc.size(1) == 1:
            depot_loc = depot_loc[:, 0, :]

        demand = self.states["demand"].float()
        if demand.dim() == 3 and demand.size(-1) == 1:
            demand = demand.squeeze(-1)
        demand_internal = demand.index_select(1, self.internal_to_external)

        time_window = self.states["time_window"].float()
        time_window_internal = time_window.index_select(1, self.internal_to_external)

        self.states["observations"] = {
            "depot_loc": depot_loc,
            "cus_loc": self.states["cus_loc"].float(),
            "rs_loc": self.states["rs_loc"].float(),
            "time_window": time_window_internal,
            "demand": demand_internal,
        }
        self.VEHICLE_CAPACITY = self.states["loading_capacity"].float()
        self.VEHICLE_BATTERY = self.states["battery_capacity"].float()
        self.used_capacity = self.states["current_load"].float()
        self.used_battery = self.states["current_battery"].float()
        self.current_time = self.states["current_time"].float()

    def external_mask_to_internal(self, mask):
        mask_t = _to_tensor(mask, self.device)
        return mask_t.index_select(-1, self.internal_to_external)

    def logits_to_external(self, logits):
        return logits.index_select(-1, self.external_to_internal)

    def get_current_node(self):
        current_external = self.states["last_node_idx"].long()
        return self.external_to_internal[current_external]

    def get_mask(self):
        action_mask = self.states["action_mask"].long()
        internal_action_mask = action_mask.index_select(-1, self.internal_to_external)
        return (1 - internal_action_mask).to(torch.bool)
