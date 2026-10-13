"""CVRP actor/critic input boundary: geometry/capacity/history, no road resources.

The environment retains its physical matrices, rewards and route validator. This
copies the observation mapping without mutating arrays. Matrix placeholders carry
only shape so the historical shared EVRPTW interfaces remain usable. Future raw
observation fields are excluded unless explicitly reviewed here.
"""
import numpy as np
import torch

KEEP = frozenset({
    'depot_loc', 'cus_loc', 'rs_loc', 'demand', 'loading_capacity', 'current_load',
    'last_node_idx', 'prev_node_idx', 'action_mask', 'instance_mask',
    'customer_visited', 'node_visit_count', 'cs_visited_current_route',
    'route_membership_current', 'route_order_rank',
    'visited_customers_ratio', 'visited_customers_raio',
    'remain_feasible_customers_ratio', 'remain_feasible_customers_raio',
    'route_served_customers_ratio', 'current_route_step_count', 'current_route_customer_count',
})

def cvrp_no_road_observation(obs):
    from .graph_attention_model_wrapper import prepare_observation_batch
    obs = prepare_observation_batch(obs)
    if obs['rs_loc'].shape[1] != 0:
        raise ValueError('E1 no-road input boundary is for CVRP without charging stations')
    out = {k: v for k, v in obs.items() if k in KEEP}
    like = obs['cus_loc']
    def constant(shape, value=0.):
        if torch.is_tensor(like):
            return torch.full(shape, value, device=like.device, dtype=torch.float32)
        return np.full(shape, value, dtype=np.float32)
    batch, customers = like.shape[:2]
    nodes = customers + 1
    for key in ('edge_distance', 'edge_time', 'edge_energy'):
        out[key] = constant((batch, nodes, nodes))
    out['service_time'] = constant((batch, nodes))
    out['time_window'] = constant((batch, nodes, 2))
    out['time_window'][..., 1] = 1.
    shape = obs['current_load'].shape
    out['current_time'] = constant(shape)
    out['current_battery'] = constant(shape)
    out['remaining_battery'] = constant(shape, 1.)
    out['battery_capacity'] = constant((batch, 1), 1.)
    out['full_charge_time'] = constant((batch, 1))
    out['fixed_full_charge'] = constant((batch, 1), 1.)
    return out


def vrptw_no_road_observation(obs):
    """Keep observable clock/windows and the physical mask, but no road matrices.

    Clock and feasibility are consequences of interaction with the real world;
    this is a no explicit road-input baseline, not a road-independent MDP.
    """
    from .graph_attention_model_wrapper import prepare_observation_batch
    source = prepare_observation_batch(obs)
    out = cvrp_no_road_observation(source)
    for key in ('time_window', 'service_time', 'current_time'):
        out[key] = source[key]
    return out
