"""Gradient monitoring must reuse training graphs and release each after backward."""
import gc
import weakref

import pytest
import torch

from offline2online.slppo_diagnostics import (
    detached_component_gradients,
    gradient_diagnostics_from_components,
    tensors_to_floats,
)


class SavedActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, references):
        saved = value.detach().clone()
        references.append(weakref.ref(saved))
        ctx.save_for_backward(saved)
        return value.sin()

    @staticmethod
    def backward(ctx, derivative):
        (saved,) = ctx.saved_tensors
        return derivative * saved.cos(), None


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(3, 7)
        self.head = torch.nn.Linear(7, 2)
        self.saved_activations = []
        self.forwards = 0

    def forward(self, value):
        self.forwards += 1
        activation = SavedActivation.apply(self.body(value), self.saved_activations)
        return self.head(activation)


def run_training(monitored, shared):
    torch.manual_seed(901)
    policy = TinyPolicy().double()
    optimizer = torch.optim.SGD(policy.parameters(), lr=.02)
    inputs = torch.randn(8, 3, dtype=torch.double)
    # Deliberately retain loss Python references: normal backward must release
    # their saved activations regardless, unlike an orphan monitoring-only graph.
    retained_losses = []
    monitored_metrics = []
    steps = []
    for epoch in range(1, 22):
        optimizer.zero_grad(set_to_none=True)
        for minibatch in range(2):
            sample = monitored and epoch in (1, 20) and minibatch == 0
            output = policy(inputs)
            ppo_loss = output.square().mean() * .3 / 2
            retained_losses.append(ppo_loss)
            if sample:
                before = [None if p.grad is None else p.grad.clone() for p in policy.parameters()]
                components = {'ppo': detached_component_gradients(ppo_loss, policy.head.parameters())}
                for previous, p in zip(before, policy.parameters()):
                    if previous is None:
                        assert p.grad is None
                    else:
                        torch.testing.assert_close(previous, p.grad, rtol=0, atol=0)
            if shared:
                sl_loss = (output - .7).square().mean() * .5 / 2
                retained_losses.append(sl_loss)
                if sample:
                    components['sl'] = detached_component_gradients(sl_loss, policy.head.parameters())
                (ppo_loss + sl_loss).backward()
            else:
                ppo_loss.backward()
                assert all(ref() is None for ref in policy.saved_activations)
                sl_loss = (policy(inputs) - .7).square().mean() * .5 / 2
                retained_losses.append(sl_loss)
                if sample:
                    components['sl'] = detached_component_gradients(sl_loss, policy.head.parameters())
                sl_loss.backward()
            assert all(ref() is None for ref in policy.saved_activations)
            if sample:
                assert all(not value.requires_grad and value.grad_fn is None
                           for gradient in components.values() for value in gradient)
                monitored_metrics.append(tensors_to_floats(gradient_diagnostics_from_components(components)))
                del components
        steps.append([p.grad.clone() for p in policy.parameters()])
        optimizer.step()
    return policy, steps, monitored_metrics


@pytest.mark.parametrize('shared', [False, True])
def test_monitoring_preserves_training_and_frees_activations_across_epoch_20(shared):
    original, original_steps, _ = run_training(False, shared)
    monitored, monitored_steps, metrics = run_training(True, shared)
    assert monitored.forwards == original.forwards == 42 * (1 if shared else 2)
    for original_step, monitored_step in zip(original_steps, monitored_steps):
        for left, right in zip(original_step, monitored_step):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(original.parameters(), monitored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(metrics) == 2
    assert all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in metrics)
    assert all('grad_ppo_sl_cosine' in row and 'grad_ppo_to_sl_ratio' in row for row in metrics)


def test_activation_probe_detects_old_orphan_diagnostic_graph():
    policy = TinyPolicy()
    loss = policy(torch.ones(2, 3)).sum()
    gradients = detached_component_gradients(loss, policy.head.parameters())
    assert any(ref() is not None for ref in policy.saved_activations)
    # The diagnostic may retain detached head gradients, but the training loss
    # must be backwarded before the next independent forward is constructed.
    loss.backward()
    gc.collect()
    assert all(ref() is None for ref in policy.saved_activations)
    assert all(gradient.grad_fn is None for gradient in gradients)
