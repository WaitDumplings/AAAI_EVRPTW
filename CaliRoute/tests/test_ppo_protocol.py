"""Training-signal invariants for the opt-in physical PPO protocol."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
import torch

from test_rollout_static_cache import _agent, _run
from offline2online.ppo_protocol import chunk_weight, fresh_policy_kl, validate_protocol, protocol_signature
from offline2online.trainer import _evaluate_policy_loss_with_stats, _compute_gae_from_rewards, _slice_obs_by_env


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _evaluate(agent, batch, returns, advantages, width):
    cfg = {'training': {'ppo_loss_reduction': 'valid_actions', 'vf_coef': .5, 'ent_coef': .01}}
    total = 0.
    for start in range(0, len(batch.observations), width):
        end = min(start + width, len(batch.observations))
        w = chunk_weight(batch.valid, start, end, 'valid_actions')
        if not w:
            continue
        loss = _evaluate_policy_loss_with_stats(agent, batch, returns, advantages, cfg, step_start=start, step_end=end)[0] * w
        total += float(loss.detach())
        loss.backward()
    return total, {n: p.grad.clone() for n, p in agent.named_parameters() if p.grad is not None}


def test_valid_action_loss_and_gradient_ignore_chunk_boundaries_and_padding():
    model = _agent()
    batch = _run(model)
    returns = batch.rewards.cumsum(0)
    advantages = torch.randn_like(batch.rewards)
    full = _evaluate(copy.deepcopy(model), batch, returns, advantages, len(batch.observations))
    chunked = _evaluate(copy.deepcopy(model), batch, returns, advantages, 3)
    assert full[0] == pytest.approx(chunked[0], abs=1e-6)
    for name in full[1]:
        torch.testing.assert_close(full[1][name], chunked[1][name], rtol=1e-4, atol=3e-6)
    padded = copy.deepcopy(batch)
    for key in ['actions', 'old_logprobs', 'rewards', 'dones', 'values', 'valid', 'entropies', 'route_boundaries']:
        value = getattr(padded, key)
        extra = value[-1:].clone()
        if key == 'valid':
            extra.zero_()
        setattr(padded, key, torch.cat([value, extra], dim=0))
    padded.observations.append(padded.observations[-1])
    padded_returns = torch.cat([returns, torch.full_like(returns[:1], 1e4)])
    padded_adv = torch.cat([advantages, torch.full_like(advantages[:1], 1e4)])
    result = _evaluate(copy.deepcopy(model), padded, padded_returns, padded_adv, 3)
    assert result[0] == pytest.approx(full[0], abs=1e-6)
    for name in full[1]:
        torch.testing.assert_close(full[1][name], result[1][name], rtol=1e-4, atol=3e-6)


def test_fresh_kl_reencodes_changed_weights_and_ignores_invalid_padding():
    model = _agent()
    batch = _run(model)
    before = fresh_policy_kl(model, batch, _slice_obs_by_env, minibatch_size=1)
    assert before['post_update_kl'] < 1e-9
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(torch.randn_like(p) * .02)
    batch.old_logprobs[~batch.valid] = float('nan')
    with patch.object(model.backbone, 'encode', wraps=model.backbone.encode) as encoder:
        after = fresh_policy_kl(model, batch, _slice_obs_by_env, minibatch_size=1)
    assert encoder.call_count == batch.actions.shape[1]
    assert after['post_update_kl'] > 1e-6
    assert after['post_update_kl_valid_actions'] == int(batch.valid.sum())
    # Independently calculate the all-valid-action estimator.
    values = []
    with torch.no_grad():
        for t, obs in enumerate(batch.observations):
            _, logp, _, _, _ = model.get_action_and_value_cached(obs, action=batch.actions[t].long(), state=model.backbone.encode(obs))
            delta = (logp - batch.old_logprobs[t])[batch.valid[t]].double()
            values.append(torch.expm1(delta) - delta)
    assert after['post_update_kl'] == pytest.approx(float(torch.cat(values).mean()), rel=1e-5, abs=1e-7)


def test_fresh_kl_nonfinite_valid_sample_fails_closed():
    model = _agent()
    batch = _run(model)
    batch.old_logprobs[batch.valid] = -10000
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        fresh_policy_kl(model, batch, _slice_obs_by_env)


def test_collector_bootstrap_changes_nonterminal_gae_only():
    rewards = torch.tensor([[[-1., -2.]], [[-3., -4.]]])
    dones = torch.tensor([[[False, False]], [[False, True]]])
    values = torch.zeros_like(rewards)
    returns, advantage = _compute_gae_from_rewards(rewards, values, dones, 1., 1., last_values=torch.tensor([[5., 9.]]))
    torch.testing.assert_close(returns, torch.tensor([[[1., -6.]], [[2., -4.]]]))
    torch.testing.assert_close(advantage, returns)


def test_protocol_validates_incompatible_settings_and_signatures():
    cfg = {'training': {'gamma': 1., 'ppo_loss_reduction': 'valid_actions'}, 'env': {'reward_contract': 'strict_distance'}}
    validate_protocol(cfg)
    with pytest.raises(ValueError, match='scalar critic'):
        validate_protocol(cfg, decomposed_critic=True)
    changed = copy.deepcopy(cfg)
    changed['training']['gamma'] = .99
    with pytest.raises(ValueError, match='gamma=1'):
        validate_protocol(changed)
    assert protocol_signature(cfg) != protocol_signature(changed)
    changed = copy.deepcopy(cfg)
    changed['training']['target_kl'] = 0.
    with pytest.raises(ValueError, match='target_kl'):
        validate_protocol(changed)
    changed = copy.deepcopy(cfg)
    changed['offline'] = {'branch_exploration_enabled': True}
    with pytest.raises(ValueError, match='structural'):
        validate_protocol(changed)


@pytest.mark.parametrize("key,value", [
    ("exploration_prefix_fractions", [0., .7]), ("exploration_max_prefix_steps", 10),
    ("exploration_exclude_anchor_action", False), ("exploration_force_anchor", True),
    ("exploration_seed", 79),
])
def test_resume_signature_protects_search_semantics(key, value):
    cfg = {'offline': {'branch_exploration_enabled': True}}
    before = protocol_signature(cfg)
    cfg['offline'][key] = value
    assert protocol_signature(cfg) != before


def test_single_trajectory_bootstrap_keeps_instance_axis():
    rewards = torch.tensor([[[-1.], [-2.]], [[-3.], [-4.]]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    values = torch.zeros_like(rewards)
    result, _ = _compute_gae_from_rewards(rewards, values, dones, 1., 1., last_values=torch.tensor([[5.], [9.]]))
    torch.testing.assert_close(result, torch.tensor([[[1.], [3.]], [[2.], [5.]]]))


def test_fresh_kl_preserves_random_stream_and_rejects_active_dropout():
    model = _agent()
    batch = _run(model)
    state = torch.random.get_rng_state().clone()
    fresh_policy_kl(model, batch, _slice_obs_by_env)
    assert torch.equal(state, torch.random.get_rng_state())
    model.add_module('active_test_dropout', torch.nn.Dropout(.1))
    with pytest.raises(ValueError, match='dropout'):
        fresh_policy_kl(model, batch, _slice_obs_by_env)
