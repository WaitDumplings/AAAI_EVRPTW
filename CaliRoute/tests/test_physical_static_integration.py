from unittest.mock import patch

import numpy as np
import pytest
import torch

from caliroute.plugins.physical_static import DirectedPhysicalRelationEncoder, TypedStaticFusion
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.encoder import GraphAttentionEncoder
from test_model_design_optimizations import agent, observation
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def physical_obs(obs=None):
    if obs is None:
        obs, _ = observation()
    batch, nodes = obs['edge_distance'].shape[:2]
    return {**obs, 'node_input_context': torch.rand(batch, nodes, 12),
            'graph_input_context': torch.tensor([[1., 1., 1., 1., 1., .2, 1., 1., 1., 0.]]).expand(batch, -1)}


def stage2(**kwargs):
    return agent(use_physical_input_context=True, **kwargs)


@pytest.mark.parametrize('flags', [
    {'use_typed_static_fusion': True},
    {'use_edge_relation_encoder': True},
    {'use_typed_static_fusion': True, 'use_edge_relation_encoder': True},
    {'use_edge_relation_encoder': True, 'use_edge_value_messages': True, 'use_edge_state_updates': True},
])
def test_zero_initialized_extensions_preserve_shared_initialization_rng_policy_value(flags):
    torch.manual_seed(195)
    base = stage2()
    random_base = torch.rand(4)
    torch.manual_seed(195)
    upgraded = stage2(**flags)
    torch.testing.assert_close(torch.rand(4), random_base, rtol=0, atol=0)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key], value, atol=0, rtol=0)
    obs = physical_obs()
    action = torch.ones(1, 3, dtype=torch.long)
    expected = base.get_action_and_value(obs, action=action)
    actual = upgraded.get_action_and_value(obs, action=action)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_disabled_extensions_preserve_strict_checkpoint_keys():
    model = agent()
    agent().load_state_dict(model.state_dict(), strict=True)
    assert not any('static_fusion' in key or 'edge_relation' in key for key in model.state_dict())


def test_every_static_branch_and_relation_path_receives_gradient_after_warmup():
    torch.manual_seed(10)
    model = stage2(use_typed_static_fusion=True, use_edge_relation_encoder=True,
                   use_edge_value_messages=True, use_edge_state_updates=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = physical_obs()
    action = torch.ones(1, 3, dtype=torch.long)
    for _ in range(3):
        optimizer.zero_grad()
        _, logprob, entropy, value = model.get_action_and_value(obs, action=action)
        loss = -logprob.mean() + value.square().mean() - .01 * entropy.mean()
        loss.backward()
        optimizer.step()
    for name, parameter in model.named_parameters():
        if any(token in name for token in ('static_fusion.', 'edge_relation_encoder.', 'edge_relation_adapter.')):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_directed_relations_distinguish_reverse_costs_and_invalid_edges():
    state = StateWrapper(physical_obs(), 'cpu')
    types = torch.tensor([[0, 2, 2, 2, 1]])
    encoder = DirectedPhysicalRelationEncoder()
    state.states['edge_distance'][0, 1, 2] = .2
    state.states['edge_distance'][0, 2, 1] = 1.3
    edge, valid = encoder(state.states, types)
    assert edge.shape == (1, 5, 5, 16) and valid.all()
    assert not torch.allclose(edge[0, 1, 2], edge[0, 2, 1])
    state.states['edge_distance'][0, 1, 2] = float('inf')
    state.states['edge_time'][0, 1, 2] = float('nan')
    edge, valid = encoder(state.states, types)
    assert not valid[0, 1, 2] and valid[0, 2, 1]
    assert torch.isfinite(edge).all()


def test_zero_energy_is_valid_when_battery_is_inactive_and_dummy_time_does_not_leak():
    state = StateWrapper(physical_obs(), 'cpu')
    state.states['graph_input_context'][:, 6] = 0
    state.states['graph_input_context'][:, 8] = 0
    state.states['edge_energy'].zero_()
    types = torch.tensor([[0, 2, 2, 2, 1]])
    encoder = DirectedPhysicalRelationEncoder()
    before, valid = encoder(state.states, types)
    assert valid.all()
    state.states['edge_time'].fill_(float('nan'))
    state.states['edge_energy'].fill_(float('inf'))
    after, _ = encoder(state.states, types)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_relation_encoder_and_transformer_are_permutation_equivariant_for_variable_sizes():
    torch.manual_seed(82)
    relation = DirectedPhysicalRelationEncoder(8)
    encoder = GraphAttentionEncoder(4, 16, 2, use_edge_relation_encoder=True,
                                    edge_relation_dim=8, use_edge_value_messages=True,
                                    use_edge_state_updates=True)
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            if 'edge_relation_adapter' in name:
                p.normal_(std=.04)
    for count in (5, 11):
        order = torch.randperm(count)
        states = {key: torch.rand(2, count, count) for key in ('edge_distance', 'edge_time', 'edge_energy')}
        states.update(node_input_context=torch.rand(2, count, 12), graph_input_context=torch.rand(2, 10),
                      battery_capacity=torch.ones(2, 1))
        # Valid physical activity flags, not arbitrary real numbers.
        states['graph_input_context'][:, 6:] = 1
        types = torch.randint(3, (2, count))
        edges, _ = relation(states, types)
        permuted = {key: value[:, order][:, :, order] if key.startswith('edge_') else value
                    for key, value in states.items()}
        permuted['node_input_context'] = states['node_input_context'][:, order]
        other, _ = relation(permuted, types[:, order])
        torch.testing.assert_close(other, edges[:, order][:, :, order])
        nodes = torch.randn(2, count, 16)
        before, before_edges = encoder(nodes, edge_relations=edges, return_edge_relations=True)
        after, after_edges = encoder(nodes[:, order], edge_relations=other, return_edge_relations=True)
        token_order = torch.cat((torch.zeros(1, dtype=torch.long), order + 1))
        torch.testing.assert_close(after, before[:, token_order], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(after_edges, before_edges[:, order][:, :, order], atol=1e-6, rtol=1e-5)


def test_cache_reuses_single_relation_matrix_for_all_trajectories_and_steps():
    raw, env = observation()
    obs = physical_obs(raw)
    model = stage2(use_typed_static_fusion=True, use_edge_relation_encoder=True,
                   cache_static_observations=True, use_edge_state_updates=True)
    relation = model.backbone.edge_relation_encoder
    with patch.object(relation, 'forward', wraps=relation.forward) as encode:
        cache = model.backbone.encode(obs)
        assert cache[5]['edge_relations'].shape == (1, 5, 5, 16)
        before = model.backbone.decode(obs, cache)
        next_obs, *_ = env.step(np.full(3, 1))
        next_obs = stack_observations([next_obs])
        next_obs.update(node_input_context=obs['node_input_context'], graph_input_context=obs['graph_input_context'])
        actual = model.backbone.decode(next_obs, cache)
        assert encode.call_count == 1
    expected = model.backbone(next_obs)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(before[0], actual[0])


@pytest.mark.parametrize('sdpa', [False, True])
def test_optional_edge_values_keep_graph_token_semantics_and_backward(sdpa):
    encoder = GraphAttentionEncoder(4, 16, 2, use_sdpa=sdpa, use_edge_relation_encoder=True,
                                    edge_relation_dim=7, use_edge_value_messages=True)
    nodes = torch.randn(2, 5, 16, requires_grad=True)
    edges = torch.randn(2, 5, 5, 7, requires_grad=True)
    result = encoder(nodes, edge_relations=edges)
    assert result.shape == (2, 6, 16)
    result.square().mean().backward()
    for layer in encoder.layers:
        assert torch.isfinite(layer.edge_relation_adapter.value_out.weight.grad).all()
        assert layer.edge_relation_adapter.value_out.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs,match', [
    ({'use_typed_static_fusion': True}, 'physical_input_context'),
    ({'use_edge_relation_encoder': True}, 'physical_input_context'),
    ({'use_edge_value_messages': True}, 'require use_edge_relation_encoder'),
    ({'use_edge_state_updates': True}, 'require use_edge_relation_encoder'),
    ({'edge_relation_dim': 0}, 'positive integer'),
    ({'decoder_observation_mode': 'other'}, 'feasible or dual'),
    ({'decoder_observation_mode': 'dual'}, 'use_resource_decoder'),
])
def test_invalid_flag_combinations_fail_early(kwargs, match):
    with pytest.raises(ValueError, match=match):
        agent(**kwargs)


def test_half_relation_encoder_compresses_large_physical_values_in_float32():
    state = StateWrapper(physical_obs(), 'cpu')
    state.states['edge_distance'].fill_(1e10)
    encoder = DirectedPhysicalRelationEncoder().half()
    edge, valid = encoder(state.states, torch.tensor([[0, 2, 2, 2, 1]]))
    assert valid.all() and torch.isfinite(edge).all()
    assert edge.dtype == torch.float16


def test_typed_static_fusion_ignores_nonfinite_inactive_time_and_capacity():
    state = StateWrapper(physical_obs(), 'cpu')
    state.states['graph_input_context'][:, 7:9] = 0
    fusion = TypedStaticFusion(16)
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.normal_(std=.1)
    embeddings = torch.randn(1, 5, 16)
    types = torch.tensor([[0, 2, 2, 2, 1]])
    before = fusion(embeddings, state.observations, state.states, types)
    state.observations['time_window'].fill_(float('nan'))
    state.observations['service_time'].fill_(float('inf'))
    state.observations['demand'].fill_(float('nan'))
    after = fusion(embeddings, state.observations, state.states, types)
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
