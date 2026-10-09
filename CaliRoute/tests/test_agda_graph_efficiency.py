"""AGDA uses authoritative road information once per state and keeps replay finite."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from caliroute.plugins.agda import AdaptiveGraphAttention
from offline2online.models.graph_attention_model_wrapper import StateWrapper
from offline2online.models.nets.graph_model.decoder import DynamicGraphKVEncoder, candidate_transitions
from test_joint_graph_integration import model, physical_obs
from test_stage2_physical_decision import make_env


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def learned_adapter():
    adapter = DynamicGraphKVEncoder(embedding_dim=16, enabled=True,
        agda_physical_candidate_features=True, agda_smooth_distance_features=True,
        use_agda_v2=True)
    # Exercise learned branches, so identity initialization cannot hide leaks.
    for name in ('node_state_proj', 'decision_state_proj', 'step_state_proj',
                 'candidate_key_delta_proj', 'candidate_value_delta_proj',
                 'candidate_action_key_delta_proj'):
        torch.nn.init.normal_(getattr(adapter, name).weight, std=.03)
    torch.nn.init.normal_(adapter.action_bias_proj[-1].weight, std=.03)
    return adapter


@pytest.mark.parametrize('kind', ['vrptw', 'cvrp'])
def test_learned_agda_ignores_inactive_resource_dummies(kind):
    obs, _ = make_env(kind=kind).reset()
    adapter = learned_adapter()
    nodes = torch.randn(1, 5, 16)
    graph, query = torch.randn(1, 1, 16), torch.randn(1, 2, 16)
    expected = adapter(nodes, graph, query, StateWrapper(obs, 'cpu'))
    obs['edge_energy'][:] = np.nan
    obs['current_battery'][:] = np.inf
    if kind == 'cvrp':
        obs['edge_time'][:] = np.inf
        obs['current_time'][:] = np.nan
        obs['time_window'][:] = np.nan
        obs['service_time'][:] = np.inf
    actual = adapter(nodes, graph, query, StateWrapper(obs, 'cpu'))
    for before, after in zip(expected, actual):
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    sum(value.square().sum() for value in actual).backward()
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)


def test_masked_nonfinite_customer_features_cannot_poison_agda_gradients():
    obs, _ = make_env().reset()
    obs['action_mask'][:, 1:3] = False
    obs['demand'][1] = np.nan
    obs['time_window'][2] = [np.nan, np.nan]
    adapter = learned_adapter()
    output = adapter(torch.randn(1, 6, 16), torch.randn(1, 1, 16),
                     torch.randn(1, 2, 16), StateWrapper(obs, 'cpu'))
    assert all(torch.isfinite(value).all() for value in output)
    sum(value.square().sum() for value in output).backward()
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)


def test_decoder_builds_physical_transitions_once_and_skips_coordinate_proxy():
    agent, obs = model(), physical_obs()
    with patch('offline2online.models.nets.graph_model.decoder.candidate_transitions',
               wraps=candidate_transitions) as transitions, \
         patch('caliroute.plugins.physical_decision.candidate_transitions',
               side_effect=AssertionError('duplicate transition computation')), \
         patch('torch.linalg.norm', side_effect=AssertionError('unused coordinate proxy')):
        logits, glimpse = agent.backbone(obs)
    assert transitions.call_count == 1
    assert torch.isfinite(glimpse).all()
    mask = StateWrapper(obs, 'cpu').states['action_mask'].bool()
    assert torch.isfinite(logits[mask]).all()
    assert torch.isneginf(logits[~mask]).all()


def test_single_decision_query_preserves_full_attention_outputs_and_gradients():
    torch.manual_seed(185)
    optimized = AdaptiveGraphAttention(embedding_dim=16, enabled=True)
    for name in ('decision_state_proj', 'step_state_proj'):
        torch.nn.init.normal_(getattr(optimized, name).weight, std=.1)
    reference = deepcopy(optimized)
    original_attention = reference.token_attn.forward

    def full_attention(query, key, value, **kwargs):
        output, weights = original_attention(key, key, value, **kwargs)
        return output[:, :1], weights

    inputs = [torch.randn(2, 5, 16), torch.randn(2, 3, 9, 16),
              torch.randn(2, 3, 5, 30), torch.randn(2, 3, 15)]
    left_inputs = [value.clone().requires_grad_() for value in inputs]
    right_inputs = [value.clone().requires_grad_() for value in inputs]
    with patch.object(optimized.token_attn, 'forward', wraps=optimized.token_attn.forward) as attention:
        left = optimized(*left_inputs)
    assert attention.call_args.args[0].shape[1] == 1
    assert attention.call_args.args[1].shape[1] == 9
    with patch.object(reference.token_attn, 'forward', side_effect=full_attention):
        right = reference(*right_inputs)
    for actual, expected in zip(left, right):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    for output in (left, right):
        sum(value.square().sum() for value in output).backward()
    for actual, expected in zip(left_inputs, right_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7)
    for (name, actual), (_, expected) in zip(optimized.named_parameters(), reference.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=name)
