"""Collector cutoff bootstrapping and valid-action reward diagnostics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import collect_rollout, compute_returns
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
from test_strict_reward_contract import env


class Backbone(nn.Module):
    supports_static_rollout_cache = False
    cache_static_observations = True

    def __init__(self, dropout=False):
        super().__init__()
        self.dropout = nn.Dropout(.3) if dropout else nn.Identity()

    def forward(self, observations):
        mask = torch.as_tensor(observations["action_mask"])
        logits = self.dropout(torch.ones(mask.shape)).masked_fill(~mask, -1e9)
        time = torch.as_tensor(observations["current_time"], dtype=torch.float32)
        return logits, time


class Critic(nn.Module):
    def forward(self, output):
        return (output[1] + 7.).unsqueeze(-1)


class Agent(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.backbone, self.critic = Backbone(dropout), Critic()


class MixedTerminalEnv:
    """One true termination, one true failure and one collector cutoff."""
    n_traj, num_customers = 3, 1
    instance = SimpleNamespace(instance_id="mixed_terminal")

    @property
    def unwrapped(self):
        return self

    def observation(self):
        return {"action_mask": np.ones((3, 2), dtype=bool), "current_time": np.full(3, self.steps, dtype=np.float32)}

    def reset(self, **kwargs):
        self.steps = 0
        return self.observation(), {"reward_shaping_initial_potential": np.array([.25, .5, .75])}

    def step(self, action):
        self.steps += 1
        reward = np.array([-1., -2., -3.]) if self.steps == 1 else np.array([99., 99., -3.])
        return self.observation(), reward, np.array([True, False, False]), np.array([False, True, False]), {}


def test_bootstrap_uses_next_state_only_for_unfinished_collector_cutoffs():
    batch = collect_rollout(Agent(), [MixedTerminalEnv()], 2, "greedy", "cpu", bootstrap_truncation=True)
    torch.testing.assert_close(batch.bootstrap_values, torch.tensor([[0., 0., 9.]]))
    torch.testing.assert_close(batch.initial_shaping_potential, torch.tensor([[.25, .5, .75]]))
    returns = compute_returns(batch.rewards, batch.dones, 1., last_values=batch.bootstrap_values)
    torch.testing.assert_close(returns[0], torch.tensor([[-1., -2., 3.]]))
    stats = batch.reward_component_stats["base"]
    assert stats == {"sum": -9., "sum_abs": 9., "sum_sq": 23., "count": 4., "max_abs": 3., "nonfinite_count": 0.}


def test_optional_bootstrap_preserves_actions_and_rng_including_dropout():
    agent = Agent(dropout=True)
    torch.manual_seed(931)
    baseline = collect_rollout(agent, [MixedTerminalEnv()], 2, "sample", "cpu")
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(931)
    bootstrapped = collect_rollout(agent, [MixedTerminalEnv()], 2, "sample", "cpu", bootstrap_truncation=True)
    torch.testing.assert_close(bootstrapped.actions, baseline.actions, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
    assert baseline.bootstrap_values is None


@pytest.mark.parametrize("fast", [False, True])
def test_completed_strict_shaped_rollout_gae_and_component_identity(fast):
    environment = env(fast, pbrs_config=PotentialRewardConfig(
        gamma=1., strict_contract=True, use_customer_pbrs=True, use_repair_distance_pbrs=True,
        use_feasible_ratio_pbrs=True, feasible_ratio_coef=.2))
    torch.manual_seed(73)
    batch = collect_rollout(Agent(), [environment], 32, "sample", "cpu", bootstrap_truncation=True)
    assert batch.dones[-1].all() and all(info["success"].all() for info in batch.final_infos)
    assert batch.bootstrap_values is None
    objective = torch.tensor(np.stack([i["objective_distance_km"] for i in batch.final_infos]), dtype=torch.float32)
    expected = -objective / 7. - batch.initial_shaping_potential
    actual = torch.where(batch.valid, batch.rewards, 0.).sum(0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    returns = compute_returns(batch.rewards, batch.dones, 1.)
    torch.testing.assert_close(returns[0], expected, atol=2e-6, rtol=1e-6)
    from offline2online.trainer import _compute_gae_returns
    gae_returns, _ = _compute_gae_returns(batch, gamma=1., gae_lambda=1.)
    torch.testing.assert_close(gae_returns[0], expected, atol=2e-6, rtol=1e-6)
    components = batch.reward_component_stats
    assert components["base"]["count"] == float(batch.valid.sum())
    assert components["failure_cost"]["sum"] == 0.
    assert components["shaping"]["sum"] == pytest.approx(-.4, abs=1e-6)
    assert components["shaped"]["sum"] == pytest.approx(float(expected.sum()), abs=1e-6)


def test_return_bootstrap_shape_is_checked():
    with pytest.raises(ValueError, match="last_values shape"):
        compute_returns(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool), 1., torch.zeros(4))
