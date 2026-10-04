from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from caliroute.plugins import AdaptiveGraphAttention, AdaptiveGraphDecisionAdapter, RoadDistanceInjection
from offline2online.models.nets.graph_model.multi_head_attention import MultiHeadAttentionEncoder
from test_model_design_optimizations import agent, observation


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def road_inputs():
    torch.manual_seed(271)
    distance = torch.rand(2, 6, 6) + .1
    distance.diagonal(dim1=-2, dim2=-1).zero_()
    windows = torch.stack((torch.rand(2, 6), torch.full((2, 6), 2.)), -1)
    return dict(distance=distance, travel_time=distance * .3, energy=distance * .8,
                time_windows=windows, service_time=torch.rand(2, 6) * .2,
                battery_capacity=torch.full((2, 1), 3.))


def test_rdi_physical_unit_scaling_and_node_permutation():
    module = RoadDistanceInjection(4)
    inputs = road_inputs()
    scaled = {key: value * (7. if key == 'distance' else 13. if key in ('energy', 'battery_capacity') else 5.)
              for key, value in inputs.items()}
    torch.testing.assert_close(module.features(**inputs), module.features(**scaled), atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        module.net[-1].weight.normal_(std=.2)
    order = torch.tensor([0, 4, 1, 3, 5, 2])
    permuted = {key: value[:, order][:, :, order] if value.ndim == 3 and value.shape[-1] == 6 else
                value[:, order] if value.shape[1] == 6 else value for key, value in inputs.items()}
    expected = module(**inputs)[:, :, order][:, :, :, order]
    torch.testing.assert_close(module(**permuted), expected, atol=2e-6, rtol=2e-6)
    actual = module(**inputs)
    assert not torch.allclose(actual, actual.transpose(-1, -2))
    assert actual.abs().max() <= module.max_bias


def test_rdi_zero_init_unreachable_edges_and_nonperturbing_diagnostics():
    inputs = road_inputs()
    inputs['distance'][0, 2, 3] = torch.inf
    inputs['energy'][0, 2, 3] = torch.inf
    module = RoadDistanceInjection(3)
    module.diagnostics_enabled = True
    out = module(**inputs, base_bias=torch.ones(2, 6, 6))
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(module.net[-1].weight.grad).all()
    assert module.net[-1].weight.grad.abs().sum() > 0
    assert all(not value.requires_grad and torch.isfinite(value) for value in module.diagnostics().values())
    assert module.diagnostics()['gate_mean'] == 1
    assert module.diagnostics()['residual_to_base'] == 0
    assert module(torch.ones(5, 5)).shape == (1, 3, 5, 5)


def test_agda_candidate_gates_identity_bounds_gradients_and_disabled_branches():
    torch.manual_seed(3)
    gate = AdaptiveGraphDecisionAdapter(7, hidden_dim=5)
    features = torch.randn(2, 3, 4, 7)
    residuals = (torch.randn(2, 3, 4, 8), None, 0, torch.randn(2, 3, 4))
    gate.diagnostics_enabled = True
    out = gate(features, residuals)
    assert torch.equal(out[0], residuals[0]) and torch.equal(out[3], residuals[3])
    assert out[1] is None and out[2] == 0
    (out[0].square().mean() + out[3].square().mean()).backward()
    assert gate.net[-1].weight.grad.abs().sum() > 0
    assert gate.diagnostics()['gate_mean'] == 1
    with torch.no_grad():
        gate.net[-1].weight.normal_(std=20.)
    out = gate(features, residuals)
    ratio = out[0] / residuals[0]
    assert ratio.min() >= .5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
    assert all(not value.requires_grad for value in gate.diagnostics().values())


@pytest.mark.parametrize('rdi,agda', [(True, False), (False, True), (True, True)])
def test_new_plugins_preserve_trained_initial_policy_and_existing_parameters(rdi, agda):
    torch.manual_seed(29)
    base = agent()
    # Exercise learned dynamic residuals, rather than only zero DDE initialization.
    with torch.no_grad():
        base.backbone.decoder.dynamic_graph_kv_encoder.candidate_action_key_delta_proj.weight.normal_(std=.04)
        base.backbone.decoder.dynamic_graph_kv_encoder.action_bias_proj[-1].weight.normal_(std=.04)
    improved = agent(use_rdi_v2=rdi, use_agda_v2=agda, optimize_dynamic_projections=True)
    missing, extra = improved.load_state_dict(base.state_dict(), strict=False)
    assert not extra
    allowed = ('backbone.rdi_adapter.', 'backbone.decoder.dynamic_graph_kv_encoder.agda_adapter.')
    assert missing and all(key.startswith(allowed) for key in missing)
    obs, _ = observation()
    for left, right in zip(base.backbone(obs), improved.backbone(obs)):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='only one'):
        agent(use_residual_edge_bias=True, use_rdi_v2=True)


@pytest.mark.parametrize('bias_heads', [0, 1, 4])
@pytest.mark.parametrize('masked', [False, True])
def test_encoder_sdpa_matches_reference_outputs_and_gradients(bias_heads, masked):
    torch.manual_seed(14)
    base = MultiHeadAttentionEncoder(16, 4).double()
    fast = MultiHeadAttentionEncoder(16, 4, use_sdpa=True).double()
    fast.load_state_dict(base.state_dict(), strict=True)
    originals = [torch.randn(2, n, 16, dtype=torch.float64, requires_grad=True) for n in (3, 5, 5)]
    copies = [x.detach().clone().requires_grad_() for x in originals]
    bias = None if bias_heads == 0 else torch.randn((2, 3, 5) if bias_heads == 1 else (2, 4, 3, 5), dtype=torch.float64, requires_grad=True)
    bias_copy = None if bias is None else bias.detach().clone().requires_grad_()
    mask = None if not masked else torch.tensor([[False, True, False, False, True], [False, False, True, False, False]])
    expected = base(*originals, mask=mask, attn_bias=bias)
    actual = fast(*copies, mask=mask, attn_bias=bias_copy)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for x, y in zip(originals, copies):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-12, rtol=2e-12)
    if bias is not None:
        torch.testing.assert_close(bias.grad, bias_copy.grad, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(base.project_out.weight.grad, fast.project_out.weight.grad, atol=2e-12, rtol=2e-12)


def test_sdpa_full_backbone_logits_values_and_checkpoint_compatibility():
    base = agent(use_residual_edge_bias=True)
    fast = agent(use_residual_edge_bias=True, use_encoder_sdpa=True)
    fast.load_state_dict(base.state_dict(), strict=True)
    obs, _ = observation()
    for original, actual in zip(base.backbone(obs), fast.backbone(obs)):
        torch.testing.assert_close(original, actual, atol=2e-6, rtol=2e-5)


def test_complete_agda_core_accepts_routing_wrapper_tensors_without_state():
    model = agent(use_agda_v2=True)
    dynamic = model.backbone.decoder.dynamic_graph_kv_encoder
    obs, _ = observation()
    captured = {}
    original_forward = AdaptiveGraphAttention.forward
    def record(self, *args, **kwargs):
        captured['args'], captured['kwargs'] = args, kwargs
        captured['out'] = original_forward(self, *args, **kwargs)
        return captured['out']
    with patch.object(AdaptiveGraphAttention, 'forward', record):
        model.backbone(obs)
    standalone = AdaptiveGraphAttention(embedding_dim=32, enabled=True, use_agda_v2=True)
    standalone.load_state_dict(dynamic.state_dict(), strict=True)
    actual = standalone(*captured['args'], **captured['kwargs'])
    for out, expected in zip(actual, captured['out']):
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_plugins_train_with_an_unrelated_tiny_attention_backbone():
    """A host contract smoke test, not an experiment on a second routing model."""
    torch.manual_seed(11)
    host = nn.Linear(2, 16)
    rdi = RoadDistanceInjection(n_heads=1, hidden_dim=8)
    agda = AdaptiveGraphAttention(embedding_dim=16, n_heads=2, enabled=True,
                                 candidate_feature_dim=7, system_feature_dim=3,
                                 num_tokens=2, use_agda_v2=True)
    points = torch.rand(2, 5, 2)
    nodes = host(points)
    distance = torch.cdist(points, points)
    bias = rdi(distance).squeeze(1)
    attended = (nodes @ nodes.transpose(1, 2) / 4. + bias).softmax(-1) @ nodes
    tokens = torch.stack((attended.mean(1), attended[:, 0]), 1).unsqueeze(1)
    residuals = agda(attended, tokens, torch.randn(2, 1, 5, 7), system_features=torch.randn(2, 1, 3))
    logits = (attended.unsqueeze(1) + residuals[2]).sum(-1) + residuals[3]
    (-logits.log_softmax(-1)[..., 2].mean()).backward()
    assert host.weight.grad.abs().sum() > 0
    assert rdi.net[-1].weight.grad.abs().sum() > 0
    assert agda.candidate_action_key_delta_proj.weight.grad.abs().sum() > 0
