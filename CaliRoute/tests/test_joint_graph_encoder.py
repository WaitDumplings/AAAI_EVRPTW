"""Contracts for a directed graph encoder used by the shared routing policy.

These tests deliberately exercise graph semantics (direction, padding, edge
validity and node permutation), rather than compare the implementation with a
second copy of the same equations. No routing environment or GPU is required.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from caliroute.plugins.joint_graph import JointGraphEncoder


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph_inputs(batch=2, nodes=6, *, requires_grad=False):
    generator = torch.Generator().manual_seed(1729)
    return {
        'node_embeddings': torch.randn(batch, nodes, 32, generator=generator).requires_grad_(requires_grad),
        'edge_relations': torch.randn(batch, nodes, nodes, 8, generator=generator).requires_grad_(requires_grad),
        'edge_valid': torch.ones(batch, nodes, nodes, dtype=torch.bool),
        'graph_features': torch.randn(batch, 10, generator=generator).requires_grad_(requires_grad),
    }


def encoder():
    torch.manual_seed(58)
    return JointGraphEncoder(embedding_dim=32, edge_dim=8, n_heads=4, n_layers=2, dropout=0.0)


def assert_outputs_close(actual, expected, *, exact=False):
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0 if exact else 3e-6, rtol=0 if exact else 3e-6)


@pytest.mark.parametrize('bias_heads', [None, 1, 4])
def test_shared_graph_shape_and_finite_backward(bias_heads):
    model = encoder()
    inputs = graph_inputs(requires_grad=True)
    context = torch.randn(2, 1, 32, requires_grad=True)
    bias = None if bias_heads is None else torch.randn(
        (2, 6, 6) if bias_heads == 1 else (2, 4, 6, 6), requires_grad=True)
    nodes, edges = model(**inputs, graph_context=context, attn_bias=bias)
    # One graph token precedes physical nodes; there is no trajectory axis.
    assert nodes.shape == (2, 7, 32)
    assert edges.shape == (2, 6, 6, 8)
    assert torch.isfinite(nodes).all() and torch.isfinite(edges).all()
    # A nonuniform target avoids an accidentally constant LayerNorm sum loss.
    target = torch.linspace(-1., 1., nodes.numel()).reshape_as(nodes)
    (nodes * target).sum().backward()
    for key in ('node_embeddings', 'edge_relations', 'graph_features'):
        grad = inputs[key].grad
        assert grad is not None and torch.isfinite(grad).all(), key
        assert grad.abs().sum() > 0, key
    assert context.grad is not None and context.grad.abs().sum() > 0
    if bias is not None:
        assert bias.grad is not None and torch.isfinite(bias.grad).all()
        assert bias.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_permutation_equivariance_preserves_graph_token_and_depot():
    model = encoder()
    inputs = graph_inputs()
    inputs['edge_valid'][:, 1, 4] = False
    mask = torch.tensor([[False, False, False, False, True, False],
                         [False, False, True, False, False, False]])
    bias = torch.randn(2, 4, 6, 6)
    expected_nodes, expected_edges = model(**inputs, node_mask=mask, attn_bias=bias)
    # Physical depot remains index zero; customers/stations may be relabelled.
    order = torch.tensor([0, 4, 1, 5, 3, 2])
    changed = dict(inputs)
    changed['node_embeddings'] = inputs['node_embeddings'][:, order]
    changed['edge_relations'] = inputs['edge_relations'][:, order][:, :, order]
    changed['edge_valid'] = inputs['edge_valid'][:, order][:, :, order]
    actual_nodes, actual_edges = model(**changed, node_mask=mask[:, order],
                                      attn_bias=bias[:, :, order][:, :, :, order])
    encoded_order = torch.cat((torch.tensor([0]), order + 1))
    torch.testing.assert_close(actual_nodes, expected_nodes[:, encoded_order], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual_edges, expected_edges[:, order][:, :, order], atol=3e-6, rtol=3e-6)


def test_reversing_arcs_does_not_collapse_to_undirected_graph():
    model = encoder()
    inputs = graph_inputs()
    forward_nodes, forward_edges = model(**inputs)
    reversed_inputs = dict(inputs, edge_relations=inputs['edge_relations'].transpose(1, 2))
    reverse_nodes, reverse_edges = model(**reversed_inputs)
    assert not torch.allclose(forward_nodes, reverse_nodes, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(forward_edges, forward_edges.transpose(1, 2), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(forward_edges, reverse_edges, atol=1e-6, rtol=1e-6)


def test_node_and_edge_losses_train_value_paths_and_both_layer_updates():
    model = encoder()
    inputs = graph_inputs(requires_grad=True)
    inputs['edge_valid'][:, 2, 4] = False
    nodes, edges = model(**inputs)
    generator = torch.Generator().manual_seed(183)
    node_target = torch.randn(nodes.shape, generator=generator)
    edge_target = torch.randn(edges.shape, generator=generator)
    ((nodes * node_target).mean() + (edges * edge_target).mean()).backward()
    assert len(model.layers) == 2
    for layer_index, layer in enumerate(model.layers):
        for name in ('edge_value', 'edge_value_out.weight', 'edge_update.weight',
                     'edge_source.weight', 'edge_target.weight', 'edge_reverse.weight'):
            parameter = dict(layer.named_parameters())[name]
            assert parameter.grad is not None, (layer_index, name)
            assert torch.isfinite(parameter.grad).all(), (layer_index, name)
            assert parameter.grad.abs().sum() > 0, (layer_index, name)
    # An invalid road arc cannot train either its latent or a neighborhood sum.
    assert torch.count_nonzero(inputs['edge_relations'].grad[:, 2, 4]) == 0


def test_invalid_edge_values_cannot_influence_valid_graph():
    model = encoder()
    inputs = graph_inputs()
    inputs['edge_valid'][:, 2, :] = False
    inputs['edge_valid'][:, 1, 4] = False
    expected = model(**inputs)
    changed = dict(inputs)
    changed['edge_relations'] = inputs['edge_relations'].clone()
    changed['edge_relations'][~inputs['edge_valid']] = 1.e5
    actual = model(**changed)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1][inputs['edge_valid']], expected[1][inputs['edge_valid']], atol=0, rtol=0)


def test_padding_nodes_do_not_change_real_nodes_or_valid_edges():
    model = encoder()
    inputs = graph_inputs()
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -1] = True
    expected = model(**inputs, node_mask=mask)
    changed = dict(inputs)
    changed['node_embeddings'] = inputs['node_embeddings'].clone()
    changed['node_embeddings'][:, -1] = 1.e4
    changed['edge_relations'] = inputs['edge_relations'].clone()
    changed['edge_relations'][:, -1, :] = 1.e4
    changed['edge_relations'][:, :, -1] = -1.e4
    actual = model(**changed, node_mask=mask)
    # Graph token and every nonpadding node must ignore padding contents.
    torch.testing.assert_close(actual[0][:, :-1], expected[0][:, :-1], atol=0, rtol=0)
    torch.testing.assert_close(actual[1][:, :-1, :-1], expected[1][:, :-1, :-1], atol=0, rtol=0)


def test_nonfinite_contents_of_invalid_edges_and_padding_are_sanitized():
    model = encoder()
    inputs = graph_inputs()
    inputs['edge_valid'][:, 1, 4] = False
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -1] = True
    expected = model(**inputs, node_mask=mask)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed['node_embeddings'][:, -1] = torch.nan
    changed['edge_relations'][:, 1, 4] = torch.inf
    changed['edge_relations'][:, -1] = torch.nan
    changed['edge_relations'][:, :, -1] = -torch.inf
    actual = model(**changed, node_mask=mask)
    assert_outputs_close(actual, expected, exact=True)
    assert all(torch.isfinite(value).all() for value in actual)


@pytest.mark.parametrize('all_padding', [False, True])
def test_no_reachable_edges_or_no_real_nodes_stays_finite(all_padding):
    model = encoder()
    inputs = graph_inputs(requires_grad=True)
    inputs['edge_valid'].zero_()
    mask = torch.full((2, 6), all_padding, dtype=torch.bool)
    nodes, edges = model(**inputs, node_mask=mask)
    assert torch.isfinite(nodes).all() and torch.isfinite(edges).all()
    (nodes.square().mean() + edges.square().mean()).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_road_edge_validity_is_not_a_global_information_attention_mask():
    model = encoder()
    inputs = graph_inputs()
    inputs['edge_valid'].zero_()
    before, _ = model(**inputs)
    changed = dict(inputs)
    changed['node_embeddings'] = inputs['node_embeddings'].clone()
    changed['node_embeddings'][:, 4, 0] += 5.
    after, _ = model(**changed)
    # Nodes may exchange global information even without a traversable arc.
    # Action feasibility remains owned by the routing decoder/environment.
    assert not torch.allclose(before[:, 1], after[:, 1], atol=1e-7, rtol=1e-7)


def test_encoding_is_deterministic_in_train_eval_and_safe_to_cache():
    model = encoder()
    inputs = graph_inputs()
    model.train()
    train_outputs = model(**inputs)
    repeated = model(**inputs)
    model.eval()
    with torch.no_grad():
        eval_outputs = model(**inputs)
    assert_outputs_close(repeated, train_outputs, exact=True)
    assert_outputs_close(eval_outputs, train_outputs, exact=True)
    # Consumers can index the one static graph encoding for many trajectories.
    trajectory_graph = torch.tensor([0, 0, 0, 1, 1])
    expanded_inputs = {key: value[trajectory_graph] for key, value in inputs.items()}
    repeated_graph_outputs = model(**expanded_inputs)
    assert_outputs_close(repeated_graph_outputs, tuple(value[trajectory_graph] for value in eval_outputs))
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert not any(isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules())


def test_graph_batch_composition_does_not_change_a_policy():
    model = encoder()
    inputs = graph_inputs()
    together = model(**inputs)
    apart = [model(**{key: value[index:index + 1] for key, value in inputs.items()}) for index in range(2)]
    assert_outputs_close(together, tuple(torch.cat([part[slot] for part in apart]) for slot in range(2)))


def test_graph_feature_context_has_an_active_path_to_encoded_nodes():
    model = encoder()
    inputs = graph_inputs()
    expected, _ = model(**inputs)
    changed = dict(inputs, graph_features=inputs['graph_features'] + .75)
    actual, _ = model(**changed)
    assert not torch.allclose(actual[:, 1:], expected[:, 1:], atol=1e-6, rtol=1e-6)


def test_checkpoint_roundtrip_reproduces_graph_encoding():
    original = encoder()
    replacement = encoder()
    replacement.load_state_dict(original.state_dict(), strict=True)
    inputs = graph_inputs()
    assert_outputs_close(replacement(**inputs), original(**inputs), exact=True)


def test_dropout_cannot_break_ppo_replay_contract():
    with pytest.raises(ValueError, match='dropout|determin'):
        JointGraphEncoder(embedding_dim=32, edge_dim=8, n_heads=4, n_layers=2, dropout=.1)
