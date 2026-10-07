"""Typed candidate transitions and resource-conditioned, stateless decoder residuals.

All transition arithmetic uses the environment's normalized physical matrices in
float32. Learned feature compression happens only afterwards. Direct-return
margins are descriptive features, never substitutes for the environment mask
(which also permits feasible returns through charging stations).
"""
from __future__ import annotations

import math
import torch
from torch import nn

CANDIDATE_FEATURE_GROUPS = {
    'time': ('arrival', 'wait', 'service_start', 'departure', 'start_due_slack',
             'work_end_slack', 'direct_return_time_margin', 'next_time', 'charge_time'),
    'energy': ('battery_on_arrival', 'battery_after', 'battery_margin',
               'direct_return_battery_margin', 'travel_energy'),
    'capacity': ('load_on_arrival', 'load_after', 'capacity_margin', 'demand'),
    'relation': ('travel_distance', 'return_distance', 'depot_detour'),
    'node_type': ('is_depot', 'is_customer', 'is_station', 'terminal', 'next_route',
             'direct_return_finite', 'action_feasible'),
}


def _step(x, batch, steps, nodes=None):
    if nodes is None:
        if x.ndim == 0:
            x = x.reshape(1, 1, 1)
        elif x.ndim == 1:
            x = x[:, None, None]
        elif x.ndim == 2:
            x = x[..., None]
        return x.expand(batch, steps, 1)
    if x.ndim == 2:
        x = x[:, None, :]
    return x.expand(batch, steps, nodes)


def candidate_transitions(state, node_embeddings, *, node_mask=None):
    """Return exact one-action post-state predictions, plus informative margins.

    Customer windows constrain service *start*, not finish. A station's service
    window is not a charging constraint in this environment. Depot arrival and
    route reset are separate: the actual next state resets time/load/battery on
    both terminal and nonterminal depot visits. No Euclidean fallback is allowed.
    """
    s = state.states
    B, N, _ = node_embeddings.shape
    device = node_embeddings.device
    action = s['action_mask'].to(device=device, dtype=torch.bool)
    if action.ndim == 2:
        action = action[:, None]
    T = action.size(1)
    current = state.get_current_node().to(device=device, dtype=torch.long)
    if current.ndim == 1:
        current = current[:, None]
    current = current.expand(B, T)
    bi = torch.arange(B, device=device)[:, None]
    edges = {}
    for name in ('distance', 'time', 'energy'):
        key = 'edge_' + name
        if key not in s:
            raise KeyError(f'Physical decoder requires {key}; no geometric fallback is valid')
        edge = s[key].to(device=device, dtype=torch.float32)
        if edge.ndim == 2:
            edge = edge[None]
        edge = edge.expand(B, N, N)
        edges[name] = (edge[bi, current], edge[:, None, :, 0].expand(B, T, N),
                       edge[bi, current, 0, None])
    valid = torch.ones((B, T, N), dtype=torch.bool, device=device)
    return_finite = valid.clone()
    graph_context = s.get('graph_input_context')
    for name, (travel, back, _) in edges.items():
        active = torch.ones((B, 1, 1), dtype=torch.bool, device=device)
        if graph_context is not None and name in {'time', 'energy'}:
            flag = 8 if name == 'time' else 6
            active = graph_context[:, flag, None, None].to(device=device) > .5
        valid = valid & (~active | (torch.isfinite(travel) & (travel >= 0)))
        return_finite = return_finite & (~active | (torch.isfinite(back) & (back >= 0)))
    def clean(x):
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    edges = {key: tuple(clean(x) for x in value) for key, value in edges.items()}
    distance, return_distance, current_to_depot = edges['distance']
    travel_time, return_time, _ = edges['time']
    travel_energy, return_energy, _ = edges['energy']
    n_cus = s['cus_loc'].size(1)
    ids = torch.arange(N, device=device)[None, None]
    depot = (ids == 0).expand(B, T, N)
    customer = ((ids >= 1) & (ids <= n_cus)).expand(B, T, N)
    station = (ids > n_cus).expand(B, T, N)
    for pad in (s.get('instance_mask'), node_mask):
        if pad is not None:
            pad = pad.to(device=device, dtype=torch.bool)
            valid = valid & ~_step(pad, B, T, N)
            customer = customer & ~_step(pad, B, T, N)
            station = station & ~_step(pad, B, T, N)
            depot = depot & ~_step(pad, B, T, N)
    def scalar(key):
        return _step(s[key].to(device=device, dtype=torch.float32), B, T)
    def node(key):
        x = s[key].to(device=device, dtype=torch.float32)
        if x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return x[:, None].expand(B, T, N)
    tw = s['time_window'].to(device=device, dtype=torch.float32)
    demand, service = node('demand'), node('service_time')
    if graph_context is None:
        time_active = capacity_active = torch.ones((B, 1, 1), dtype=torch.bool, device=device)
    else:
        time_active = graph_context[:, 8, None, None] > .5
        capacity_active = graph_context[:, 7, None, None] > .5
    valid = valid & (~capacity_active | ~customer | torch.isfinite(demand))
    valid = valid & (~time_active | ~customer | (
        torch.isfinite(tw).all(-1)[:, None] & torch.isfinite(service)))
    arrival = scalar('current_time') + travel_time
    service_start = torch.where(customer, torch.maximum(arrival, tw[:, None, :, 0]), arrival)
    wait = service_start - arrival
    battery_on_arrival = scalar('current_battery') + travel_energy
    full_charge = scalar('full_charge_time')
    fixed = scalar('fixed_full_charge') > .5
    # Clipping here is the physical partial-charge rule, not feature clipping.
    charge = torch.where(fixed, full_charge, full_charge * battery_on_arrival.clamp(0, 1))
    charge = torch.where(station, charge, 0.)
    departure = service_start + torch.where(customer, service, 0.) + charge
    next_time = torch.where(depot, 0., departure)
    battery_after = torch.where(station | depot, 0., battery_on_arrival)
    load_on_arrival = scalar('current_load').expand(B, T, N)
    load_after = torch.where(depot, 0., load_on_arrival + torch.where(customer, demand, 0.))
    visited = s.get('customer_visited', s.get('node_visit_count'))
    if visited is None:
        raise KeyError('Physical decoder requires customer_visited or node_visit_count')
    visited = _step(visited.to(device=device), B, T, N) > 0
    unserved = customer & ~visited
    all_served = ~unserved.any(-1, keepdim=True)
    terminal = depot & all_served
    next_route = depot & ~all_served
    cs_seen = s.get('cs_visited_current_route', s.get('node_visit_count'))
    cs_seen = _step(cs_seen.to(device=device), B, T, N) > 0
    # CS visits are route-local. A depot reset clears cs_seen in the env.
    future = (unserved | depot | (station & ~cs_seen & (ids != current[..., None]))) & valid
    action = action.expand(B, T, N)
    out = dict(arrival=arrival, wait=wait, service_start=service_start,
               departure=departure, start_due_slack=tw[:, None, :, 1] - service_start,
               work_end_slack=1. - departure,
               direct_return_time_margin=1. - departure - return_time,
               next_time=next_time, charge_time=charge,
               battery_on_arrival=battery_on_arrival, battery_after=battery_after,
               battery_margin=1. - battery_after,
               direct_return_battery_margin=1. - battery_after - return_energy,
               travel_energy=travel_energy, load_on_arrival=load_on_arrival,
               load_after=load_after, capacity_margin=1. - load_after, demand=demand,
               travel_distance=distance, return_distance=return_distance,
               depot_detour=distance + return_distance - current_to_depot,
               is_depot=depot, is_customer=customer, is_station=station,
               terminal=terminal, next_route=next_route,
               direct_return_finite=return_finite, action_feasible=action,
               valid=valid, future_mask=future)
    return out


def _safe_attention(scores, valid):
    """Finite zero readout if no observation is valid, without inventing actions."""
    scores = scores.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    weights = torch.softmax(scores, dim=-1) * valid.float()
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


class ResourceDecisionAdapter(nn.Module):
    """Trajectory-independent, zero-output-initialized physical decision plugin."""
    def __init__(self, embedding_dim, *, hidden_dim=32, observation_mode='feasible',
                 use_resources=True, use_edge_relations=False, edge_relation_dim=16):
        super().__init__()
        if observation_mode not in {'feasible', 'dual'}:
            raise ValueError("decoder_observation_mode must be 'feasible' or 'dual'")
        self.use_resources = bool(use_resources)
        self.use_edge_relations = bool(use_edge_relations)
        self.observation_mode = observation_mode
        self.hidden_dim = int(hidden_dim)
        self.diagnostics_enabled = False
        self._diagnostics = {}
        if self.use_resources:
            self.typed_projections = nn.ModuleDict({name: nn.Sequential(
                nn.Linear(len(features), hidden_dim), nn.SiLU())
                for name, features in CANDIDATE_FEATURE_GROUPS.items()})
            self.resource_gates = nn.Linear(13, len(CANDIDATE_FEATURE_GROUPS))
            self.resource_film = nn.Linear(13, 2 * hidden_dim)
            self.action_key_out = nn.Linear(hidden_dim, embedding_dim, bias=False)
            self.action_bias_out = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.zeros_(self.action_key_out.weight)
            nn.init.zeros_(self.action_bias_out.weight)
            if observation_mode == 'dual':
                self.readout_node = nn.Linear(embedding_dim, 2 * hidden_dim, bias=False)
                self.readout_query = nn.Linear(embedding_dim, hidden_dim, bias=False)
                self.readout_resource = nn.Linear(13, hidden_dim, bias=False)
                self.readout_out = nn.Linear(2 * hidden_dim, embedding_dim, bias=False)
                nn.init.zeros_(self.readout_out.weight)
        if self.use_edge_relations:
            self.edge_action_key = nn.Linear(edge_relation_dim, embedding_dim, bias=False)
            self.edge_action_bias = nn.Linear(edge_relation_dim, 1, bias=False)
            nn.init.zeros_(self.edge_action_key.weight)
            nn.init.zeros_(self.edge_action_bias.weight)

    def precompute(self, node_embeddings):
        if self.use_resources and self.observation_mode == 'dual':
            return self.readout_node(node_embeddings).chunk(2, dim=-1)
        return None

    def forward(self, node_embeddings, query, state, *, cached_readout=None,
                edge_relations=None, edge_relation_valid=None, node_mask=None):
        B, T, D = query.shape
        N = node_embeddings.size(1)
        key_delta = query.new_zeros(B, T, N, D)
        bias_delta = query.new_zeros(B, T, N)
        query_delta = torch.zeros_like(query)
        diagnostics = {}
        if self.use_resources:
            features = candidate_transitions(state, node_embeddings, node_mask=node_mask)
            if 'graph_input_context' not in state.states:
                raise KeyError('Resource decoder requires graph_input_context with explicit resource flags')
            graph = state.states['graph_input_context'].float()
            if graph.ndim != 2 or graph.size(-1) != 10:
                raise ValueError('graph_input_context must have shape [B,10]')
            scalars = [torch.where(graph[:, flag, None, None] > .5,
                                   _step(state.states[key].float(), B, T), 0.)
                       for key, flag in (('current_time', 8), ('current_load', 7), ('current_battery', 6))]
            resource = torch.cat((graph[:, None].expand(B, T, 10), *scalars), -1)
            resource = torch.asinh(resource).to(self.resource_gates.weight.dtype)
            gates = self.resource_gates(resource).softmax(-1)
            projected = []
            active_flags = {'time': graph[:, 8], 'energy': graph[:, 6], 'capacity': graph[:, 7]}
            for index, (group, names) in enumerate(CANDIDATE_FEATURE_GROUPS.items()):
                raw = torch.stack([features[name].float() for name in names], -1)
                # Only finite current-edge candidates contribute. Finite return
                # flags explicitly distinguish missing return paths from zeros.
                raw = torch.nan_to_num(raw, nan=0., posinf=0., neginf=0.)
                projection = self.typed_projections[group]
                encoded = projection(torch.asinh(raw).to(projection[0].weight.dtype))
                if group in active_flags:
                    encoded = encoded * active_flags[group][:, None, None, None].to(encoded.dtype)
                projected.append(encoded * gates[..., index, None, None])
            hidden = sum(projected)
            scale, shift = self.resource_film(resource).chunk(2, -1)
            hidden = hidden * (1 + .5 * torch.tanh(scale)[..., None, :]) + shift[..., None, :]
            hidden = hidden * features['valid'][..., None]
            key_delta = self.action_key_out(hidden)
            bias_delta = self.action_bias_out(hidden).squeeze(-1)
            if self.observation_mode == 'dual':
                keys, values = cached_readout if cached_readout is not None else self.precompute(node_embeddings)
                read_query = self.readout_query(query) + self.readout_resource(resource)
                score = torch.einsum('bth,bnh->btn', read_query, keys) / math.sqrt(self.hidden_dim)
                feasible = features['action_feasible'] & features['valid']
                current_weight = _safe_attention(score, feasible)
                future_weight = _safe_attention(score, features['future_mask'])
                current_read = torch.einsum('btn,bnh->bth', current_weight.to(values.dtype), values)
                future_read = torch.einsum('btn,bnh->bth', future_weight.to(values.dtype), values)
                query_delta = self.readout_out(torch.cat((current_read, future_read), -1))
                if self.diagnostics_enabled:
                    diagnostics['future_attention_infeasible_mass'] = (future_weight * ~features['action_feasible']).sum(-1).mean()
                    diagnostics['future_observation_fraction'] = features['future_mask'].float().mean()
                    diagnostics['empty_future_fraction'] = (~features['future_mask'].any(-1)).float().mean()
            if self.diagnostics_enabled:
                for index, group in enumerate(CANDIDATE_FEATURE_GROUPS):
                    diagnostics['gate_' + group] = gates[..., index].float().mean()
                diagnostics['film_saturation_fraction'] = (torch.tanh(scale).abs() > .98).float().mean()
                diagnostics['invalid_candidate_fraction'] = (~features['valid']).float().mean()
        if self.use_edge_relations:
            if edge_relations is None:
                raise KeyError('Edge relation decoder requires cached edge_relations')
            current = state.get_current_node().long().reshape(B, T)
            bi = torch.arange(B, device=query.device)[:, None]
            row = edge_relations[bi, current]
            if edge_relation_valid is not None:
                row = torch.where(edge_relation_valid[bi, current, :, None], row, 0.)
            key_delta = key_delta + self.edge_action_key(row)
            bias_delta = bias_delta + self.edge_action_bias(row).squeeze(-1)
        if self.diagnostics_enabled:
            diagnostics.update(query_residual_rms=query_delta.float().square().mean().sqrt(),
                               action_key_residual_rms=key_delta.float().square().mean().sqrt(),
                               action_bias_residual_rms=bias_delta.float().square().mean().sqrt())
            self._diagnostics = {key: value.detach() for key, value in diagnostics.items()}
        return query_delta, key_delta, bias_delta

    def diagnostics(self):
        return dict(self._diagnostics)
