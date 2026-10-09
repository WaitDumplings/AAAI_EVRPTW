"""Graph backend respects cached PPO replay, real road costs and decoder masks."""
from copy import deepcopy
import numpy as np
from unittest.mock import patch

import pytest
import torch

from offline2online.models import Agent
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from test_physical_static_integration import physical_obs


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    return Agent(embedding_dim=32, n_encode_layers=2, use_dynamic_decision_encoder=True,
        use_physical_input_context=True, use_typed_static_fusion=True,
        use_joint_graph_encoder=True, joint_graph_edge_dim=8,
        use_resource_decoder=True, use_rdi_v2=True, use_agda_v2=True,
        optimize_dynamic_projections=True, cache_static_observations=True,
        agda_physical_candidate_features=True, agda_smooth_distance_features=True)


def test_cached_and_fresh_policy_and_gradients_match_with_per_instance_edges():
    torch.manual_seed(901)
    fresh = model()
    cached_model = deepcopy(fresh)
    obs = physical_obs()
    action = torch.ones(1, 3, dtype=torch.long)
    expected = fresh.get_action_and_value(obs, action=action)
    cache = cached_model.backbone.encode(obs)
    assert cache[5]['edge_relations'].shape == (1, 5, 5, 8)
    with patch.object(cached_model.backbone.joint_graph_encoder, 'forward', side_effect=AssertionError('reencoded')):
        actual = cached_model.get_action_and_value_cached(obs, action=action, cached_embeddings=cache)
    for x, y in zip(expected, actual[:4]):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for output in (expected, actual):
        (-output[1].mean() + output[3].square().mean() - .01 * output[2].mean()).backward()
    for (name, left), (_, right) in zip(fresh.named_parameters(), cached_model.named_parameters()):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)
    gradient = fresh.backbone.joint_graph_encoder.layers[0].edge_value.grad
    assert gradient is not None and gradient.abs().sum() > 0 and torch.isfinite(gradient).all()


def test_road_input_required_and_no_coordinate_fallback_or_physical_mutation():
    obs = physical_obs()
    before = deepcopy(obs)
    agent = model()
    with patch.object(agent.backbone, '_build_distance_matrix', side_effect=AssertionError('Euclidean fallback')):
        agent.backbone.encode(obs)
    for key in before:
        if isinstance(before[key], torch.Tensor):
            torch.testing.assert_close(obs[key], before[key], atol=0, rtol=0)
    broken = dict(obs)
    del broken['edge_distance']
    with pytest.raises(KeyError, match='edge_distance'):
        agent.backbone.encode(broken)


def test_encoder_unreachable_edges_finite_backward_and_inactive_dummy_invariance():
    obs = physical_obs()
    agent = model()
    state = agent.backbone._build_state(obs)
    state.states['edge_distance'][0, 1, 2] = float('inf')
    state.states['edge_time'][0, 1, 2] = float('nan')
    encoded, _ = agent.backbone._encode_from_state(state)
    assert torch.isfinite(encoded[0]).all()
    assert not encoded[5]['edge_relation_valid'][0, 1, 2]
    encoded[0].square().mean().backward()
    for name, p in agent.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    obs = physical_obs()
    obs['graph_input_context'] = obs['graph_input_context'].clone()
    obs['graph_input_context'][:, 6] = 0  # VRPTW: battery/energy inactive.
    obs['edge_energy'] = torch.zeros_like(torch.as_tensor(obs['edge_energy']))
    first = agent.backbone.encode(obs)[0]
    obs['edge_energy'].fill_(float('nan'))
    second = agent.backbone.encode(obs)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_decision_receives_learned_edges_and_keeps_environment_action_mask():
    obs = physical_obs()
    agent = model()
    adapter = agent.backbone.decoder.resource_decoder
    with patch.object(adapter, 'forward', wraps=adapter.forward) as call:
        logits, _ = agent.backbone(obs)
    relations = call.call_args.kwargs['edge_relations']
    assert relations.shape == (1, 5, 5, 8)
    mask = StateWrapper(obs, 'cpu').states['action_mask'].bool()
    assert torch.isneginf(logits[~mask]).all()
    probabilities = logits.softmax(-1)
    assert torch.isfinite(probabilities).all() and (probabilities[~mask] == 0).all()


def test_cpu_autocast_has_finite_new_graph_gradients():
    agent = model()
    obs = physical_obs()
    action = torch.ones(1, 3, dtype=torch.long)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        _, logprob, entropy, value = agent.get_action_and_value(obs, action=action)
        loss = -logprob.mean() + value.square().mean() - .01 * entropy.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None)


def test_graph_bias_batched_static_service_time_has_no_extra_batch_axis():
    obs = physical_obs()
    batch = {}
    for name, value in obs.items():
        if isinstance(value, torch.Tensor):
            batch[name] = torch.cat([value, value], dim=0)
        elif isinstance(value, np.ndarray):
            batch[name] = np.concatenate([value, value], axis=0)
        else:
            batch[name] = value
    agent = model()
    state = agent.backbone._build_state(batch)
    assert state.states['service_time'].shape == (2, 5)
    encoded = agent.backbone.encode(batch)
    assert encoded[0].shape[0] == 2
    assert torch.isfinite(encoded[0]).all()
    _, logprob, _, value = agent.get_action_and_value(batch, action=torch.ones(2, 3, dtype=torch.long))
    (-logprob.mean() + value.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None)
