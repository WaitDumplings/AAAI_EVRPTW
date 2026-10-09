"""Trainer route replay must preserve the portable SL-PPO loss contract."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from caliroute.plugins.slppo import solution_level_ppo_loss
from offline2online.trainer import (
    _compute_solution_level_ppo_loss,
    _compute_solution_level_weighted_logprob_loss,
    _prepare_solution_level_ppo_weights,
)


class LogprobAgent(torch.nn.Module):
    def __init__(self, logprobs):
        super().__init__()
        self.logprobs = torch.nn.Parameter(logprobs)
        self.backbone = SimpleNamespace(encode=lambda observation: None)

    def get_action_and_value_cached(self, observation, **kwargs):
        return None, self.logprobs[int(observation["step"][0, 0])], None, None, None


def batch_for(old, valid):
    return SimpleNamespace(
        observations=[{"step": np.full((old.shape[1], 1), step)} for step in range(len(old))],
        old_logprobs=old,
        valid=valid,
        actions=torch.zeros_like(old, dtype=torch.int64),
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_host_full_and_chunked_loss_match_portable_with_nonfinite_padding(dtype):
    old = torch.full((4, 1, 3), -1., dtype=dtype)
    valid = torch.tensor([[[1, 1, 1]], [[1, 1, 1]], [[0, 1, 1]], [[0, 0, 1]]], dtype=torch.bool)
    old[~valid] = float("nan")
    new = old + .03
    new[1, 0, 1] = float("inf")  # Reject this whole route; preserve other routes.
    agent = LogprobAgent(new)
    batch = batch_for(old, valid)
    advantages = torch.tensor([[.6, -1., -.4]], dtype=dtype)
    feasible = torch.ones_like(advantages, dtype=torch.bool)
    indices = np.array([0])
    expected = solution_level_ppo_loss(agent.logprobs, old, valid, advantages)
    loss, info = _compute_solution_level_ppo_loss(agent, batch, advantages, feasible, {}, indices, "cpu")
    torch.testing.assert_close(loss, expected.loss)
    assert info["sl_route_invalid_logprob_routes"] == 1
    expected_grad, = torch.autograd.grad(expected.loss, agent.logprobs)
    gradient, = torch.autograd.grad(loss, agent.logprobs)
    torch.testing.assert_close(gradient, expected_grad)
    assert torch.isfinite(gradient).all()
    weights, counts, info = _prepare_solution_level_ppo_weights(agent, batch, advantages, feasible, {}, indices, "cpu")
    torch.testing.assert_close(weights, expected.weights)
    torch.testing.assert_close(counts, expected.valid_counts)
    for start in range(0, len(old), 2):
        chunk = _compute_solution_level_weighted_logprob_loss(agent, batch, weights, counts, indices, "cpu", start, start + 2)
        assert torch.isfinite(chunk)
        chunk.backward()
    torch.testing.assert_close(agent.logprobs.grad, expected_grad)


def test_detached_route_weights_use_training_autocast_precision():
    class AutocastAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 3)
            self.backbone = SimpleNamespace(encode=lambda observation: None)

        def get_action_and_value_cached(self, observation, **kwargs):
            return None, self.projection(torch.tensor([[.31, -.72, .43]])), None, None, None

    torch.manual_seed(3)
    agent = AutocastAgent()
    batch = batch_for(torch.zeros(4, 1, 3), torch.ones(4, 1, 3, dtype=torch.bool))
    advantages = torch.tensor([[.6, -1., -.4]])
    feasible = torch.ones_like(advantages, dtype=torch.bool)
    indices = np.array([0])
    cfg = {"training": {"mixed_precision": True}}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = _compute_solution_level_ppo_loss(agent, batch, advantages, feasible, cfg, indices, "cpu")
    expected.backward()
    expected_gradient = agent.projection.weight.grad.clone()
    agent.zero_grad()
    # Exercise the CUDA training context choice on CPU without allocating a GPU.
    with patch("offline2online.trainer._amp_enabled", return_value=True), patch(
        "offline2online.trainer._autocast_context",
        side_effect=lambda device, enabled: torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled),
    ) as context:
        weights, counts, info = _prepare_solution_level_ppo_weights(agent, batch, advantages, feasible, cfg, indices, "cpu")
    context.assert_called_once_with("cpu", True)
    assert info["sl_route_loss"] == pytest.approx(float(expected.detach()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        chunk = _compute_solution_level_weighted_logprob_loss(agent, batch, weights, counts, indices, "cpu", 0, 4)
    chunk.backward()
    torch.testing.assert_close(agent.projection.weight.grad, expected_gradient)
