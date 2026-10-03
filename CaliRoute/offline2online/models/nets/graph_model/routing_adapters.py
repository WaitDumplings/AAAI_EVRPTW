"""Optional residual routing features; zero output preserves a loaded base policy."""
from __future__ import annotations

import torch
from torch import nn


class DirectedEdgeBias(nn.Module):
    """Per-head bias from normalized directed road metrics and time compatibility."""

    def __init__(self, n_heads: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_heads))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states):
        distance = states["edge_distance"].float()
        travel = states["edge_time"].float()
        energy = states["edge_energy"].float()
        tw = states["time_window"].float()
        service = states["service_time"].float()
        if service.dim() == 3:
            service = service.squeeze(-1)
        earliest_arrival = tw[..., 0].unsqueeze(-1) + service.unsqueeze(-1) + travel
        latest_departure = tw[..., 1].unsqueeze(-2) - travel - service.unsqueeze(-1)
        features = torch.stack([
            distance, travel, energy,
            tw[..., 1].unsqueeze(-2) - earliest_arrival,
            torch.relu(tw[..., 0].unsqueeze(-2) - earliest_arrival),
            latest_departure - tw[..., 0].unsqueeze(-1),
        ], dim=-1)
        # Road data can contain unreachable edges. Their mask is applied by the
        # caller; use finite inputs here to avoid inf * zero during initialization.
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.net(features).permute(0, 3, 1, 2)


class PostChargeAdapter(nn.Module):
    """Action bias from post-action vehicle state and one-hop escape margins.

    These are soft planning features, not replacements for the environment's
    authoritative feasibility mask or a guarantee of multi-hop return feasibility.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _scalar(value, batch_size, steps, like):
        value = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if value.dim() == 0:
            value = value.view(1, 1, 1)
        elif value.dim() == 1:
            value = value[:, None, None]
        elif value.dim() == 2:
            value = value.unsqueeze(-1)
        return value.expand(batch_size, steps, 1)

    def features(self, state, like):
        values = state.states
        B, N, _ = like.shape
        current = state.get_current_node()
        if current.dim() == 1:
            current = current.unsqueeze(1)
        T = current.size(1)
        n_cus = int(values["cus_loc"].size(1))
        node_ids = torch.arange(N, device=like.device)
        is_cs = node_ids >= 1 + n_cus
        is_depot = node_ids == 0
        batch_ids = torch.arange(B, device=like.device)[:, None]
        edge_time = values["edge_time"].to(like)
        edge_energy = values["edge_energy"].to(like)
        arrival = self._scalar(values["current_time"], B, T, like) + edge_time[batch_ids, current, :]
        consumed = self._scalar(values["current_battery"], B, T, like) + edge_energy[batch_ids, current, :]
        if "full_charge_time" not in values or "fixed_full_charge" not in values:
            raise KeyError("post-charge adapter requires full_charge_time and fixed_full_charge observations")
        full_charge = self._scalar(values["full_charge_time"], B, T, like)
        fixed = self._scalar(values["fixed_full_charge"], B, T, like).bool()
        charge = torch.where(fixed, full_charge.expand_as(consumed), consumed.clamp(0, 1) * full_charge)
        charge = charge * is_cs.to(like.dtype)
        service = values["service_time"].to(like)
        if service.dim() == 3:
            service = service.squeeze(-1)
        start = torch.maximum(arrival, values["time_window"].to(like)[..., 0].unsqueeze(1))
        departure = start + service.unsqueeze(1) + charge
        departure = torch.where(is_depot, torch.zeros_like(departure), departure)
        remaining = torch.where(is_cs | is_depot, torch.ones_like(consumed), 1.0 - consumed)

        # Future stops: depot and unvisited physical CS, excluding the candidate
        # itself. Each entry refers to a next stop, not to the current action mask.
        stop_ids = node_ids[is_cs | is_depot]
        energy_to_stop = edge_energy[:, :, stop_ids].unsqueeze(1)
        time_to_stop = edge_time[:, :, stop_ids].unsqueeze(1)
        allowed = (node_ids[:, None] != stop_ids[None, :]).view(1, 1, N, -1).expand(B, T, -1, -1)
        visited = values.get("cs_visited_current_route")
        if visited is not None:
            visited = visited.to(device=like.device).bool()
            if visited.dim() == 2:
                visited = visited.unsqueeze(1)
            available = ~visited[..., stop_ids]
            available[..., 0] = True  # depot is always an available stop
            # A depot action starts a fresh vehicle route, clearing CS visits.
            allowed = allowed & (available.unsqueeze(2) | is_depot.view(1, 1, N, 1))
        escape_margin = remaining.unsqueeze(-1) - energy_to_stop
        time_margin = 1.0 - departure.unsqueeze(-1) - time_to_stop
        reachable = allowed & (escape_margin >= 0) & (time_margin >= 0)
        best_energy_margin = escape_margin.masked_fill(~allowed, -10.0).amax(-1)
        best_time_margin = time_margin.masked_fill(~reachable, -10.0).amax(-1)
        reachable_fraction = reachable.to(like.dtype).sum(-1) / allowed.sum(-1).clamp_min(1)
        features = torch.stack([
            departure, remaining, charge, best_energy_margin, best_time_margin,
            reachable_fraction, remaining - edge_energy[:, None, :, 0],
            1.0 - departure - edge_time[:, None, :, 0],
            is_cs.to(like.dtype).view(1, 1, N).expand(B, T, N),
            departure - arrival,
        ], dim=-1)
        return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    def forward(self, state, node_embeddings):
        if int(state.states["rs_loc"].size(1)) == 0:
            # CVRP/VRPTW do not use a charging adapter.
            return 0
        return self.net(self.features(state, node_embeddings)).squeeze(-1)
