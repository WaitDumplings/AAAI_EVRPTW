from __future__ import annotations

import pytest
import torch

from caliroute.plugins.input_encoding import PhysicalInputContextAdapter
from offline2online.models.graph_attention_model_wrapper import prepare_observation_batch
from test_model_design_optimizations import agent, observation


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def contexts(batch=1, nodes=5):
    return torch.randn(batch, nodes, 12), torch.randn(batch, 10)


def with_context(obs):
    batch, nodes = obs['edge_distance'].shape[:2]
    node, graph = contexts(batch, nodes)
    return {**obs, 'node_input_context': node, 'graph_input_context': graph}


def test_disabled_adapter_loads_legacy_state_strictly_and_enabled_does_not_shift_rng():
    torch.manual_seed(119)
    base = agent()
    rng_after_base = torch.rand(3)
    torch.manual_seed(119)
    upgraded = agent(use_physical_input_context=True)
    rng_after_upgraded = torch.rand(3)
    torch.testing.assert_close(rng_after_upgraded, rng_after_base, rtol=0, atol=0)
    for key, expected in base.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key], expected, rtol=0, atol=0)
    agent(use_physical_input_context=False).load_state_dict(base.state_dict(), strict=True)
    assert not any('physical_input_adapter' in key for key in base.state_dict())
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing
    assert all(key.startswith('backbone.physical_input_adapter.') for key in missing)


def test_zero_adapter_preserves_policy_value_and_learns_from_physical_context():
    base = agent()
    upgraded = agent(use_physical_input_context=True)
    upgraded.load_state_dict(base.state_dict(), strict=False)
    obs, _ = observation()
    obs = with_context(obs)
    action = torch.ones(1, 3, dtype=torch.long)
    expected = base.get_action_and_value(obs, action=action)
    actual = upgraded.get_action_and_value(obs, action=action)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    (-actual[1].mean() + actual[3].mean()).backward()
    adapter = upgraded.backbone.physical_input_adapter
    for branch in (adapter.node_mlp, adapter.graph_mlp):
        assert torch.isfinite(branch[-1].weight.grad).all()
        assert branch[-1].weight.grad.abs().sum() > 0


def test_contexts_are_batched_and_cached_with_static_embeddings():
    obs, _ = observation()
    model = agent(use_physical_input_context=True, cache_static_observations=True)
    obs = with_context(obs)
    single = {key: value[0] for key, value in obs.items()}
    prepared = prepare_observation_batch(single)
    assert prepared['node_input_context'].shape == (1, 5, 12)
    assert prepared['graph_input_context'].shape == (1, 10)
    cache = model.backbone.encode(single)
    for key in ('node_input_context', 'graph_input_context'):
        torch.testing.assert_close(cache[5]['static_state'][key], obs[key])
    fresh = model.backbone(obs)
    dynamic_only = {key: value for key, value in obs.items() if key not in cache[5]['static_state']}
    cached = model.backbone.decode(dynamic_only, cache)
    for before, after in zip(fresh, cached):
        torch.testing.assert_close(after, before, atol=0, rtol=0)


def test_nonzero_adapter_is_node_permutation_equivariant_and_accepts_new_sizes():
    adapter = PhysicalInputContextAdapter(16)
    with torch.no_grad():
        adapter.node_mlp[-1].weight.normal_(std=.1)
        adapter.graph_mlp[-1].weight.normal_(std=.1)
    node, graph = contexts(2, 7)
    order = torch.tensor([0, 4, 1, 3, 6, 5, 2])
    torch.testing.assert_close(adapter(node[:, order], graph), adapter(node, graph)[:, order])
    assert adapter(*contexts(3, 101)).shape == (3, 101, 16)
    # The graph branch retains resource-scale differences even for equal nodes.
    same_nodes = node[:1].expand(2, -1, -1)
    assert not torch.allclose(adapter(same_nodes, graph)[0], adapter(same_nodes, graph)[1])


@pytest.mark.parametrize('key', ['node_input_context', 'graph_input_context'])
def test_enabled_model_requires_both_contexts(key):
    obs, _ = observation()
    obs = with_context(obs)
    del obs[key]
    with pytest.raises(KeyError, match=key):
        agent(use_physical_input_context=True).backbone(obs)


@pytest.mark.parametrize('bad_node,bad_graph', [
    ((2, 5, 11), (2, 10)), ((2, 5, 12), (2, 9)),
    ((2, 5, 12), (1, 10)), ((5, 12), (10,)),
    ((2, 0, 12), (2, 10)), ((0, 5, 12), (0, 10)),
])
def test_adapter_rejects_wrong_shape(bad_node, bad_graph):
    with pytest.raises(ValueError):
        PhysicalInputContextAdapter(16)(torch.zeros(bad_node), torch.zeros(bad_graph))


@pytest.mark.parametrize('key', ['node_input_context', 'graph_input_context'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_adapter_rejects_nonfinite_context_even_at_zero_initialization(key, value):
    node, graph = contexts()
    inputs = {'node_input_context': node, 'graph_input_context': graph}
    inputs[key].view(-1)[0] = value
    with pytest.raises(ValueError, match='finite'):
        PhysicalInputContextAdapter(16)(**inputs)


@pytest.mark.parametrize('shape', [(1, 4, 12), (2, 5, 12)])
def test_model_rejects_context_with_wrong_instance_or_node_count(shape):
    obs, _ = observation()
    obs = with_context(obs)
    obs['node_input_context'] = torch.zeros(shape)
    with pytest.raises(ValueError, match='batch/node dimensions'):
        agent(use_physical_input_context=True).backbone(obs)


@pytest.mark.parametrize('kwargs', [{'embedding_dim': 0}, {'embedding_dim': True},
                                  {'embedding_dim': 16, 'hidden_dim': 0}])
def test_invalid_adapter_dimensions_rejected(kwargs):
    with pytest.raises(ValueError, match='positive integer'):
        PhysicalInputContextAdapter(**kwargs)
