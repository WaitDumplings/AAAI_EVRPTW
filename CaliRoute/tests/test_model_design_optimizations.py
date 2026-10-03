from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from offline2online.models import Agent
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.routing_adapters import DirectedEdgeBias, PostChargeAdapter
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations
from test_rollout_static_cache import _env


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def agent(**kwargs):
    return Agent(embedding_dim=32, n_encode_layers=1, use_dynamic_decision_encoder=True, **kwargs)


def observation():
    env = _env()
    obs, _ = env.reset(seed=7)
    return stack_observations([obs]), env


def copy_baseline(baseline, improved):
    missing, extra = improved.load_state_dict(baseline.state_dict(), strict=False)
    assert not extra
    assert all(key.startswith(('backbone.residual_edge_bias.', 'backbone.decoder.post_charge_adapter.')) for key in missing)
    return missing


def test_disabled_flags_preserve_legacy_state_keys_and_static_cache_default():
    base = agent()
    optimized = agent(optimize_dynamic_projections=True, cache_static_observations=True)
    optimized.load_state_dict(base.state_dict(), strict=True)
    assert base.backbone.supports_static_rollout_cache
    assert not agent(use_static_rollout_cache=False).backbone.supports_static_rollout_cache
    assert not any('residual_edge_bias' in key or 'post_charge_adapter' in key for key in base.state_dict())


@pytest.mark.parametrize('edge,charge', [(True, False), (False, True), (True, True)])
def test_zero_initialized_adapters_preserve_logits_and_receive_gradients(edge, charge):
    torch.manual_seed(23)
    base = agent()
    upgraded = agent(use_residual_edge_bias=edge, use_post_charge_adapter=charge)
    assert copy_baseline(base, upgraded)
    obs, _ = observation()
    original = base.backbone(obs)
    actual = upgraded.backbone(obs)
    torch.testing.assert_close(actual[0], original[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], original[1], atol=0, rtol=0)
    actions = torch.ones(1, 3, dtype=torch.long)
    _, logprob, _, value = upgraded.get_action_and_value(obs, action=actions)
    (-logprob.mean() + value.mean()).backward()
    for enabled, module in ((edge, upgraded.backbone.residual_edge_bias), (charge, upgraded.backbone.decoder.post_charge_adapter)):
        if enabled:
            assert torch.isfinite(module.net[-1].weight.grad).all()
            assert module.net[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('flags', [(False, False, True), (True, False, False), (False, True, True), (True, True, True), (False, False, False)])
def test_optimized_dde_matches_outputs_and_gradients(flags):
    torch.manual_seed(44)
    options = dict(dynamic_decision_delta_k=flags[0], dynamic_decision_delta_v=flags[1], dynamic_decision_delta_action_key=flags[2])
    base = agent(**options)
    # Nonzero learned deltas exercise the actual projection paths.
    with torch.no_grad():
        for name, parameter in base.backbone.decoder.dynamic_graph_kv_encoder.named_parameters():
            if 'proj' in name or 'candidate_' in name:
                parameter.normal_(0, 0.08)
    optimized = agent(**options, optimize_dynamic_projections=True, cache_static_observations=True)
    optimized.load_state_dict(base.state_dict(), strict=True)
    obs, _ = observation()
    action = torch.ones(1, 3, dtype=torch.long)
    expected = base.get_action_and_value(obs, action=action)
    cache = optimized.backbone.encode(obs)
    actual = optimized.get_action_and_value_cached(obs, action=action, state=cache)
    for left, right in zip(actual[:4], expected):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    (expected[1].mean() + expected[3].mean()).backward()
    (actual[1].mean() + actual[3].mean()).backward()
    for name, parameter in base.named_parameters():
        other = dict(optimized.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None or torch.count_nonzero(other.grad) == 0
        else:
            assert other.grad is not None or torch.count_nonzero(parameter.grad) == 0, name
            if other.grad is not None:
                torch.testing.assert_close(other.grad, parameter.grad, atol=3e-6, rtol=5e-4, msg=name)


def test_dde_node_projection_cached_once_and_bias_only_skips_token_attention():
    obs, _ = observation()
    model = agent(optimize_dynamic_projections=True)
    dynamic = model.backbone.decoder.dynamic_graph_kv_encoder
    with patch.object(dynamic.node_state_proj, 'forward', wraps=dynamic.node_state_proj.forward) as projection:
        cached = model.backbone.encode(obs)
        for _ in range(3):
            model.backbone.decode(obs, cached)
        assert projection.call_count == 1
    bias_only = agent(optimize_dynamic_projections=True, dynamic_decision_delta_k=False,
                      dynamic_decision_delta_v=False, dynamic_decision_delta_action_key=False)
    dynamic = bias_only.backbone.decoder.dynamic_graph_kv_encoder
    with patch.object(dynamic.token_attn, 'forward', wraps=dynamic.token_attn.forward) as token_attention:
        bias_only.backbone(obs)
        assert token_attention.call_count == 0


def test_static_cache_reuses_tensors_and_dynamic_state_changes():
    obs, env = observation()
    model = agent(cache_static_observations=True, optimize_dynamic_projections=True)
    cached = model.backbone.encode(obs)
    statics = cached[5]['static_state']
    next_obs, _, _, _, _ = env.step(np.full(3, 1))
    next_obs = stack_observations([next_obs])
    with patch.object(model.backbone, '_build_state', wraps=model.backbone._build_state) as build:
        cached_output = model.backbone.decode(next_obs, cached)
        passed = build.call_args.args[0]
        assert passed['edge_distance'] is statics['edge_distance']
        np.testing.assert_array_equal(passed['last_node_idx'], next_obs['last_node_idx'])
    fresh_output = model.backbone(next_obs)
    for a, b in zip(cached_output, fresh_output):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_directed_edge_bias_has_distinct_heads_and_preserves_energy_mask():
    obs, _ = observation()
    obs['edge_time'][0, 1, 2] = 0.37
    state = StateWrapper(obs, 'cpu')
    module = DirectedEdgeBias(n_heads=4, hidden_dim=8)
    with torch.no_grad():
        module.net[-1].weight.normal_()
    bias = module(state.states)
    assert bias.shape == (1, 4, 5, 5)
    assert not torch.allclose(bias[:, :, 1, 2], bias[:, :, 2, 1])
    assert not torch.allclose(bias[:, 0], bias[:, 1])
    upgraded = agent(use_residual_edge_bias=True)
    state.states['edge_energy'][0, 1, 2] = 2.0
    bias = upgraded.backbone._build_attn_bias(state)
    assert (bias[0, :, 1, 2] == -1e9).all()


@pytest.mark.parametrize('fixed', [False, True])
def test_post_charge_features_follow_environment_departure_and_battery(fixed):
    obs, env = observation()
    env.charging_mode = 'fixed_full' if fixed else 'proportional_full'
    env.battery_used_kwh[:] = 2.0
    env.current_time_s[:] = 300.0
    env._build_static_observation_cache()
    obs = stack_observations([env._make_observation()])
    state = StateWrapper(obs, 'cpu')
    adapter = PostChargeAdapter()
    features = adapter.features(state, torch.zeros(1, 5, 32))
    env.step(np.full(3, 4))
    np.testing.assert_allclose(features[0, :, 4, 0].detach().numpy(), env.current_time_s / env.horizon_s, atol=1e-7)
    np.testing.assert_allclose(features[0, :, 4, 1].detach().numpy(), 1 - env.battery_used_kwh / env.battery_capacity_kwh)
    assert (features[0, :, 4, 2] > 0).all()
    assert torch.isfinite(features).all()


def test_post_charge_features_exclude_visited_cs_from_escape_options():
    obs, _ = observation()
    state = StateWrapper(obs, 'cpu')
    adapter = PostChargeAdapter()
    shape = torch.zeros(1, 5, 32)
    before = adapter.features(state, shape)
    # Make depot unreachable from customer 1; station remains reachable.
    state.states['edge_energy'][0, 1, 0] = 2.0
    before = adapter.features(state, shape)
    state.states['cs_visited_current_route'][:, :, 4] = True
    after = adapter.features(state, shape)
    assert (before[:, :, 1, 5] > 0).all()
    assert (after[:, :, 1, 5] == 0).all()
    assert (after[:, :, 0, 5] == 1).all()  # depot resets the route's CS visit history


def test_station_free_observations_disable_charging_adapter():
    obs, _ = observation()
    obs['rs_loc'] = torch.empty(1, 0, 2)
    # Adapter exits on task type before accessing shapes or charging metadata.
    state = type('State', (), {'states': obs})()
    assert PostChargeAdapter()(state, torch.zeros(1, 4, 32)) == 0


def test_optional_adapters_do_not_shift_shared_parameter_initialization_or_rng():
    torch.manual_seed(19)
    baseline = agent()
    after_baseline = torch.rand(3)
    torch.manual_seed(19)
    upgraded = agent(use_residual_edge_bias=True, use_post_charge_adapter=True)
    after_upgraded = torch.rand(3)
    torch.testing.assert_close(after_upgraded, after_baseline, rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key], value, rtol=0, atol=0)
