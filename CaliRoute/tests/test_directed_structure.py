"""Determinism, resource isolation and compatibility of light directed adapters."""
import pytest
import torch

from caliroute.plugins.directed_structure import (
    DirectedContentScoreMixer, DirectedRoadProfileFusion,
    build_directed_pair_features, directed_road_profiles,
)
from offline2online.models.nets.graph_model.encoder import GraphAttentionEncoder


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def distance(nodes=6, batch=2):
    return torch.rand(batch, nodes, nodes, generator=torch.Generator().manual_seed(212)) * 3


def encoder(enabled=False, sdpa=False):
    torch.manual_seed(23)
    return GraphAttentionEncoder(4, 32, 2, feed_forward_hidden=48,
                                 use_sdpa=sdpa, use_directed_score_mixer=enabled)


def activate(module):
    # Tests of representation must not accidentally only exercise zero heads.
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, DirectedContentScoreMixer):
                child.net[-1].weight.normal_(0., .15)
                child.net[-1].bias.normal_(0., .10)
            if isinstance(child, DirectedRoadProfileFusion):
                child.road[-1].weight.normal_(0., .15)
                child.road[-1].bias.normal_(0., .10)


@pytest.mark.parametrize('nodes', [1, 16, 51, 101, 1001])
def test_profiles_work_across_customer_counts_without_random_sampling(nodes):
    data = distance(nodes, 1)
    before = torch.get_rng_state().clone()
    profiles = directed_road_profiles(data)
    assert profiles.shape == (1, nodes, 16)
    assert torch.isfinite(profiles).all()
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(profiles, directed_road_profiles(data))
    if nodes == 1:
        assert torch.equal(profiles, torch.zeros_like(profiles))


def test_profiles_reverse_direction_and_follow_permutation():
    data = distance()
    forward = directed_road_profiles(data)
    reverse = directed_road_profiles(data.transpose(1, 2))
    torch.testing.assert_close(forward[..., :8], reverse[..., 8:])
    assert not torch.allclose(forward, reverse)
    permutation = torch.tensor([0, 4, 3, 1, 5, 2])
    actual = directed_road_profiles(data[:, permutation][:, :, permutation])
    torch.testing.assert_close(actual, forward[:, permutation])


def test_profiles_preserve_absolute_scale_and_ignore_diagonal_padding():
    data = distance(4, 1)
    expected = directed_road_profiles(data)
    assert not torch.allclose(expected, directed_road_profiles(data * 10))
    data[:, torch.arange(4), torch.arange(4)] = float('nan')
    torch.testing.assert_close(expected, directed_road_profiles(data))
    padded = torch.full((1, 7, 7), float('inf'))
    padded[:, :4, :4] = data
    mask = torch.tensor([[False] * 4 + [True] * 3])
    profiles = directed_road_profiles(padded, node_mask=mask)
    torch.testing.assert_close(expected, profiles[:, :4])
    assert not profiles[:, 4:].any()


def test_profile_empty_neighbourhood_is_finite_and_backward_finite():
    data = distance(4, 1).requires_grad_()
    valid = torch.zeros_like(data, dtype=torch.bool)
    profiles = directed_road_profiles(data, edge_valid=valid)
    assert not profiles.any()
    profiles.sum().backward()
    assert torch.isfinite(data.grad).all()
    assert not data.grad.any()


def test_fusion_identity_initialization_then_learns_directed_information():
    data = distance()
    nodes = torch.randn(2, 6, 32, requires_grad=True)
    model = DirectedRoadProfileFusion(32)
    assert torch.equal(model(nodes, data), nodes)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    target = torch.randn_like(nodes)
    for _ in range(2):
        optimizer.zero_grad()
        (model(nodes, data) * target).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.gate.weight.grad.abs().sum() > 0
    assert model.road[0].weight.grad.abs().sum() > 0
    assert not torch.allclose(model(nodes, data), model(nodes, data.transpose(1, 2)))


@pytest.mark.parametrize('time_on,energy_on', [(False, False), (True, False), (False, True), (True, True)])
def test_pair_features_resource_flags_and_inactive_nan_invariance(time_on, energy_on):
    data = distance()
    t, e = data * .2, data * .3
    actual = build_directed_pair_features(data, travel_time=t, energy=e,
                                          time_active=time_on, energy_active=energy_on)
    if not time_on:
        t = torch.full_like(data, float('nan'))
        assert not actual[..., 2:4].any()
    if not energy_on:
        e = torch.full_like(data, float('inf'))
        assert not actual[..., 4:6].any()
    changed = build_directed_pair_features(data, travel_time=t, energy=e,
                                           time_active=time_on, energy_active=energy_on)
    assert torch.equal(actual, changed)
    assert torch.isfinite(actual).all()


def test_pair_features_require_active_matrices_and_support_per_batch_flags():
    data = distance()
    with pytest.raises(ValueError, match='travel_time is required'):
        build_directed_pair_features(data, time_active=True)
    travel = data.clone()
    travel[0] = float('nan')
    result = build_directed_pair_features(data, travel_time=travel,
                                           time_active=torch.tensor([False, True]))
    assert torch.isfinite(result).all()
    assert not result[0, ..., 2:4].any()
    assert result[1, ..., 2:4].any()


def test_mixer_zero_residual_then_content_direction_interaction():
    pair = build_directed_pair_features(distance())
    content = torch.randn(4, 2, 7, 7, requires_grad=True)
    model = DirectedContentScoreMixer(4)
    assert torch.equal(model(content, pair), content)
    activate(model)
    output = model(content, pair)
    reverse = build_directed_pair_features(distance().transpose(1, 2))
    assert not torch.allclose(output, model(content, reverse))
    assert torch.equal(output[:, :, 0], content[:, :, 0])
    assert torch.equal(output[:, :, :, 0], content[:, :, :, 0])
    (output * torch.randn_like(output)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    assert torch.isfinite(content.grad).all()


@pytest.mark.parametrize('sdpa', [False, True])
def test_enabled_initialization_preserves_host_weights_and_rng(sdpa):
    legacy = encoder(False, sdpa)
    after_legacy = torch.get_rng_state().clone()
    adapted = encoder(True, sdpa)
    assert torch.equal(torch.get_rng_state(), after_legacy)
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, adapted.state_dict()[key]), key
    nodes = torch.randn(2, 6, 32)
    data = distance()
    bias = -data
    features = build_directed_pair_features(data)
    before = legacy(nodes, attn_bias=bias)
    after = adapted(nodes, attn_bias=bias, directed_pair_features=features)
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)
    assert not any('directed' in key for key in legacy.state_dict())


def test_encoder_permutation_and_inactive_invariance_with_learned_mixer():
    model = encoder(True)
    activate(model)
    nodes = torch.randn(2, 6, 32)
    data = distance()
    features = build_directed_pair_features(data)
    output = model(nodes, attn_bias=-data, directed_pair_features=features)
    altered = build_directed_pair_features(data, travel_time=data * 999,
                                           energy=torch.full_like(data, float('nan')))
    assert torch.equal(output, model(nodes, attn_bias=-data, directed_pair_features=altered))
    perm = torch.tensor([0, 4, 3, 1, 5, 2])
    result = model(nodes[:, perm], attn_bias=-data[:, perm][:, :, perm],
                   directed_pair_features=features[:, perm][:, :, perm])
    torch.testing.assert_close(result[:, 0], output[:, 0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(result[:, 1:], output[:, 1:][:, perm], atol=2e-6, rtol=2e-6)


def test_encoder_padding_and_finite_backward_with_cpu_amp():
    model = encoder(True)
    activate(model)
    nodes = torch.randn(2, 6, 32, requires_grad=True)
    data = distance()
    features = build_directed_pair_features(data)
    padding = torch.full((2, 9, 32), float('nan'))
    padding[:, :6] = nodes.detach()
    matrix = torch.full((2, 9, 9), float('nan'))
    matrix[:, :6, :6] = data
    mask = torch.tensor([[False] * 6 + [True] * 3] * 2)
    padded_features = build_directed_pair_features(matrix, node_mask=mask)
    normal = model(nodes, directed_pair_features=features)
    padded = model(padding, mask=mask, directed_pair_features=padded_features)
    torch.testing.assert_close(normal, padded[:, :7], atol=2e-6, rtol=2e-6)
    assert not padded[:, 7:].any()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = model(nodes, directed_pair_features=features)
        loss = (result * torch.randn_like(result)).sum()
    loss.backward()
    assert torch.isfinite(result).all() and torch.isfinite(nodes.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_directed_mixer_cannot_silently_omit_features_or_share_wrong_value_weights():
    with pytest.raises(ValueError, match='requires directed_pair_features'):
        encoder(True)(torch.randn(1, 3, 32))
    with pytest.raises(ValueError, match='cannot be combined'):
        GraphAttentionEncoder(4, 32, 1, use_directed_score_mixer=True,
                              use_edge_relation_encoder=True, use_edge_value_messages=True)
