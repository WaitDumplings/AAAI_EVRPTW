"""Gradient monitoring is diagnostic-only and compares complete contributions."""
import copy

import pytest
import torch
from torch import nn
from offline2online.slppo_diagnostics import DetachedGradientAccumulator


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, values):
        return self.head(self.trunk(values).tanh())


def run(model, monitor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    sampler = DetachedGradientAccumulator(model.head.parameters()) if monitor else None
    inputs = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    # Two PPO chunks and two separate auxiliary forwards, each backwarded once.
    for index, weight in enumerate((.4, .6)):
        output = model(inputs[index * 3:index * 3 + 3])
        loss = weight * (output.square().mean() - .02 * output.mean())
        if sampler is not None:
            saved = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
            sampler.sample('ppo_minibatch', loss)
            for name, p in model.named_parameters():
                if saved[name] is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, saved[name], rtol=0, atol=0)
        loss.backward()
    reference = tuple(p.grad.clone() for p in model.head.parameters())
    for name, coefficient in [('expert', .35 * .6), ('replay', .35 * .1)]:
        loss = -coefficient * model(inputs).log_softmax(-1)[:, 0].mean()
        if sampler is not None:
            sampler.sample(name, loss)
        loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    if sampler is not None:
        for actual, expected in zip(sampler.gradients['ppo_minibatch'], reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(not g.requires_grad and g.grad_fn is None for group in sampler.gradients.values() for g in group)
    optimizer.step()
    return gradients, optimizer.state_dict(), sampler


def test_auxiliary_monitor_preserves_gradients_parameters_and_adam_update():
    torch.manual_seed(33)
    original = Model()
    without, with_monitor = copy.deepcopy(original), copy.deepcopy(original)
    left, state_left, _ = run(without, False)
    right, state_right, sampler = run(with_monitor, True)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        torch.testing.assert_close(without.state_dict()[name], with_monitor.state_dict()[name], rtol=0, atol=0)
    for parameter in state_left['state']:
        for key in state_left['state'][parameter]:
            torch.testing.assert_close(state_left['state'][parameter][key], state_right['state'][parameter][key], rtol=0, atol=0)
    diagnostics = sampler.diagnostics()
    for name in ('expert', 'replay'):
        expected = diagnostics[f'grad_{name}_norm'] / diagnostics['grad_ppo_minibatch_norm']
        torch.testing.assert_close(diagnostics[f'grad_{name}_to_ppo_minibatch_ratio'], expected)
        assert -1.000001 <= diagnostics[f'grad_{name}_ppo_minibatch_cosine'] <= 1.000001


def test_reference_is_norm_of_gradient_sum_not_sum_of_chunk_norms():
    parameter = nn.Parameter(torch.tensor([1., 2.]))
    sampler = DetachedGradientAccumulator([parameter])
    for coefficient in (10., -9.):
        loss = coefficient * parameter.sum()
        sampler.sample('ppo_minibatch', loss)
        loss.backward()
    torch.testing.assert_close(sampler.gradients['ppo_minibatch'][0], torch.ones(2))
    assert sampler.diagnostics()['grad_ppo_minibatch_norm'].item() == pytest.approx(2 ** .5)
    assert sampler.diagnostics()['grad_expert_norm'].item() == 0.
    assert sampler.diagnostics()['grad_expert_to_ppo_minibatch_ratio'].item() == 0.


def test_zero_gradient_cosine_is_undefined_instead_of_a_false_direction():
    parameter = nn.Parameter(torch.tensor([1., 2.]))
    sampler = DetachedGradientAccumulator([parameter])
    sampler.sample('ppo_minibatch', parameter.sum())
    values = sampler.diagnostics()
    assert torch.isnan(values['grad_replay_ppo_minibatch_cosine'])
    assert values['grad_replay_to_ppo_minibatch_ratio'] == 0
    assert 'grad_ppo_minibatch_to_replay_ratio' not in values
