"""P0 preserves policy/value against arbitrary inactive placeholders."""
from copy import deepcopy

import pytest
import torch

from offline2online.models import Agent
from caliroute.plugins.resource_isolation import (
    sanitize_resource_inputs, masked_raw_layer_norm, feature_resource_mask,
)
from test_physical_static_integration import physical_obs


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(task):
    obs = {k: torch.as_tensor(v).clone() for k, v in physical_obs().items()}
    if task != 'evrptw':
        n = 1 + obs['cus_loc'].shape[1]
        for key in ('edge_distance', 'edge_time', 'edge_energy'):
            obs[key] = obs[key][:, :n, :n]
        for key in ('time_window', 'service_time', 'demand', 'node_input_context'):
            obs[key] = obs[key][:, :n]
        for key in ('action_mask', 'node_visit_count', 'customer_visited',
                    'cs_visited_current_route', 'route_membership_current', 'route_order_rank'):
            if key in obs:
                obs[key] = obs[key][..., :n]
        obs['rs_loc'] = obs['rs_loc'][:, :0]
        obs['graph_input_context'][:, 6] = 0
    if task == 'cvrp':
        obs['graph_input_context'][:, 8] = 0
    return obs


def model(isolated=True, extended=False, bias_only=False):
    m = Agent(embedding_dim=32, n_encode_layers=1,
        use_dynamic_decision_encoder=True, use_resource_isolation=isolated,
        use_physical_input_context=True, use_typed_static_fusion=extended,
        use_joint_graph_encoder=extended, joint_graph_edge_dim=8,
        use_resource_decoder=extended, use_rdi_v2=extended, use_agda_v2=True,
        use_post_charge_adapter=True, optimize_dynamic_projections=True,
        dynamic_decision_delta_k=not bias_only, dynamic_decision_delta_v=not bias_only,
        dynamic_decision_delta_action_key=not bias_only,
        cache_static_observations=True, agda_physical_candidate_features=extended)
    # Exercise trained nonzero residuals, rather than relying on zero init.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'adapter' in name or 'static_fusion' in name or 'resource_decoder' in name or 'dynamic_graph' in name:
                p.add_(torch.randn_like(p) * .02)
    return m


def corrupt_inactive(obs, task, nonfinite=False):
    out = deepcopy(obs)
    inactive = ['edge_energy', 'current_battery', 'remaining_battery', 'battery_capacity',
                'full_charge_time', 'fixed_full_charge', 'rs_streak_ratio', 'cs_visited_current_route']
    graph_cols, node_cols = [2, 4, 5, 9], [6, 7]
    if task == 'cvrp':
        inactive += ['edge_time', 'time_window', 'service_time', 'current_time']
        graph_cols += [0, 3]
        node_cols += [4, 5]
    for i, key in enumerate(inactive):
        if key in out:
            value = out[key].float()
            out[key] = torch.full_like(value, float('nan') if nonfinite else 11. + i)
    out['graph_input_context'][:, graph_cols] = float('inf') if nonfinite else 39.
    out['node_input_context'][:, :, node_cols] = float('nan') if nonfinite else -73.
    return out


@pytest.mark.parametrize('task', ['cvrp', 'vrptw'])
@pytest.mark.parametrize('extended,bias_only', [(False, False), (False, True), (True, False)])
def test_full_policy_value_and_cache_invariant_to_inactive_finite_and_nonfinite(task, extended, bias_only):
    torch.manual_seed(812)
    m = model(extended=extended, bias_only=bias_only)
    obs = observation(task)
    before = deepcopy(obs)
    action = torch.ones(1, 3, dtype=torch.long)
    expected = m.get_action_and_value(obs, action=action)
    for nonfinite in (False, True):
        changed = corrupt_inactive(obs, task, nonfinite)
        actual = m.get_action_and_value(changed, action=action)
        cached = m.get_action_and_value_cached(changed, action=action,
                                              cached_embeddings=m.backbone.encode(changed))
        for left, right, replay in zip(expected, actual, cached[:4]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            torch.testing.assert_close(left, replay, atol=0, rtol=0)
        (-actual[1].mean() + actual[3].square().mean() - .01 * actual[2].mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.zero_grad(set_to_none=True)
    for key in obs:
        torch.testing.assert_close(obs[key], before[key], atol=0, rtol=0)


@pytest.mark.parametrize('task', ['cvrp', 'vrptw'])
def test_inactive_input_gradients_zero_and_active_distance_gradient_present(task):
    m = model(extended=True)
    obs = observation(task)
    inactive = ['edge_energy', 'current_battery', 'remaining_battery', 'battery_capacity', 'full_charge_time']
    if task == 'cvrp':
        inactive += ['edge_time', 'time_window', 'service_time', 'current_time']
    for key in inactive + ['edge_distance', 'node_input_context', 'graph_input_context']:
        obs[key] = obs[key].float().requires_grad_()
    _, lp, ent, value = m.get_action_and_value(obs, action=torch.ones(1, 3, dtype=torch.long))
    (-lp.mean() + value.square().mean() - .01 * ent.mean()).backward()
    for key in inactive:
        assert obs[key].grad is None or torch.count_nonzero(obs[key].grad) == 0, key
    assert obs['edge_distance'].grad.abs().sum() > 0
    assert torch.count_nonzero(obs['node_input_context'].grad[:, :, 6:8]) == 0
    assert torch.count_nonzero(obs['graph_input_context'].grad[:, [2, 4, 5, 9]]) == 0


def test_raw_normalization_excludes_inactive_values_and_affine_bias():
    norm = torch.nn.LayerNorm(4)
    with torch.no_grad():
        norm.bias.fill_(3.)
    x = torch.tensor([[1., float('nan'), 3., float('inf')]], requires_grad=True)
    active = torch.tensor([[True, False, True, False]])
    actual = masked_raw_layer_norm(norm, x, active)
    torch.testing.assert_close(actual, torch.tensor([[2., 0., 4., 0.]]), atol=2e-5, rtol=0)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad[~active] == 0).all()


def test_resource_decoder_gates_normalize_only_active_branches():
    m = model(extended=True)
    adapter = m.backbone.decoder.resource_decoder
    adapter.diagnostics_enabled = True
    m.get_action_and_value(observation('cvrp'), action=torch.ones(1, 3, dtype=torch.long))
    d = adapter.diagnostics()
    assert d['gate_time'] == 0 and d['gate_energy'] == 0
    torch.testing.assert_close(sum(d['gate_' + k] for k in ('capacity', 'relation', 'node_type')), torch.tensor(1.))


def test_enabled_flag_requires_explicit_physical_resource_schema():
    with pytest.raises(ValueError, match='Resource isolation requires'):
        Agent(use_resource_isolation=True)
    with pytest.raises(KeyError, match='explicit resource flags'):
        sanitize_resource_inputs({})


def test_disabled_flag_preserves_checkpoint_keys_and_outputs():
    torch.manual_seed(93)
    old = model(isolated=False)
    torch.manual_seed(93)
    duplicate = model(isolated=False)
    duplicate.load_state_dict(old.state_dict(), strict=True)
    assert 'use_resource_isolation' not in old.backbone.model_integration_settings
    obs = observation('evrptw')
    action = torch.ones(1, 3, dtype=torch.long)
    for left, right in zip(old.get_action_and_value(obs, action=action),
                           duplicate.get_action_and_value(obs, action=action)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_mixed_cvrp_vrptw_batch_isolates_each_instances_resources():
    m = model(extended=True)
    cv, vr = observation('cvrp'), observation('vrptw')
    batch = {key: torch.cat((cv[key], vr[key]), dim=0) for key in cv}
    corrupted = {key: torch.cat((corrupt_inactive(cv, 'cvrp', True)[key],
                                corrupt_inactive(vr, 'vrptw', True)[key]), dim=0) for key in cv}
    action = torch.ones(2, 3, dtype=torch.long)
    expected = m.get_action_and_value(batch, action=action)
    for left, right in zip(expected, m.get_action_and_value(corrupted, action=action)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    state = m.backbone._build_state(batch)
    mask = feature_resource_mask(state.states, torch.zeros(2, 3, 30), 'candidate')
    assert not mask[0, 0, 20:24].any() and mask[1, 0, 20:24].all()
    assert not mask[:, :, 19].any()


def test_active_evrptw_resource_information_still_changes_outputs_and_has_gradients():
    torch.manual_seed(65)
    m = model(extended=True)
    obs = observation('evrptw')
    obs['current_battery'] = obs['current_battery'].float().requires_grad_()
    obs['current_time'] = obs['current_time'].float().requires_grad_()
    action = torch.ones(1, 3, dtype=torch.long)
    expected = m.get_action_and_value(obs, action=action)
    (-expected[1].mean() + expected[3].square().mean()).backward()
    assert obs['current_battery'].grad.abs().sum() > 0
    assert obs['current_time'].grad.abs().sum() > 0
    changed = dict(obs)
    changed['current_battery'] = obs['current_battery'].detach() + .3
    changed['current_time'] = obs['current_time'].detach() + .2
    actual = m.get_action_and_value(changed, action=action)
    assert not torch.allclose(expected[1], actual[1])


def test_capacity_inactive_and_nonfinite_values_are_also_isolated():
    m = model(extended=True)
    obs = observation('cvrp')
    obs['graph_input_context'][:, 7] = 0
    action = torch.ones(1, 3, dtype=torch.long)
    expected = m.get_action_and_value(obs, action=action)
    changed = deepcopy(obs)
    for key in ('demand', 'current_load', 'loading_capacity'):
        changed[key] = torch.full_like(changed[key].float(), float('nan'))
    changed['graph_input_context'][:, 1] = float('inf')
    for left, right in zip(expected, m.get_action_and_value(changed, action=action)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_cpu_autocast_inactive_nonfinite_inputs_have_finite_gradients():
    m = model(extended=True)
    obs = corrupt_inactive(observation('cvrp'), 'cvrp', True)
    obs['edge_energy'].requires_grad_()
    obs['current_time'].requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        _, lp, ent, value = m.get_action_and_value(obs, action=torch.ones(1, 3, dtype=torch.long))
        loss = -lp.mean() + value.square().mean() - .01 * ent.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(obs['edge_energy'].grad).all()
    assert torch.count_nonzero(obs['edge_energy'].grad) == 0
    assert torch.isfinite(obs['current_time'].grad).all()
    assert torch.count_nonzero(obs['current_time'].grad) == 0
