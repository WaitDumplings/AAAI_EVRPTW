"""CPU host-controller checks, independent of the full routing trainer."""
from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from offline2online import reward_normalization as rn


class DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)
        self.use_decomposed_critic = False

    def forward(self, hidden):
        normalized = self.head(hidden)
        normalizer = getattr(self, '_value_normalizer', None)
        return normalized if normalizer is None else normalizer.denormalize(normalized)


class DummyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 4)
        self.critic = DummyCritic()
        self.policy = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.critic(torch.tanh(self.trunk(inputs))).squeeze(-1)


def _cfg(**training):
    return {
        'env': {'normalize_reward': True, 'reward_distance_scale_km': 2.},
        'training': {
            'reward_norm_mode': rn.MODE, 'gamma': 1., 'gae_lambda': .95,
            'normalization_beta': .25, 'actor_min_scale': .01,
            'critic_min_std': .01, **training,
        },
        'offline': {'method': 'sl_ppo'},
    }


def _make_normalized():
    torch.manual_seed(75)
    agent = DummyAgent().double()
    controller = rn.configure(agent, _cfg())
    optimizer = torch.optim.Adam(agent.parameters(), lr=.003, amsgrad=True)
    return agent, controller, optimizer


def _one_update(agent, controller, optimizer, multiplier=1.):
    x = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 10
    target = torch.tensor([-10., -25., -40., -80., -100., -140.], dtype=torch.float64) * multiplier
    advantages = torch.tensor([-3., -2., -1., 2., 4., 9.], dtype=torch.float64) * multiplier
    normalized_adv = controller.begin_rollout_update(target, advantages, torch.ones(6, dtype=torch.bool), agent.critic, optimizer)
    optimizer.zero_grad(set_to_none=True)
    prediction = agent(x)
    loss = F.mse_loss(controller.critic.normalize(prediction), controller.critic.normalize(target))
    loss.backward()
    optimizer.step()
    return normalized_adv


def _checkpoint_roundtrip(agent, controller, optimizer):
    stream = io.BytesIO()
    torch.save({
        'model_state_dict': rn.inference_model_state(agent),
        'normalization': controller.checkpoint_state(agent.critic),
        'optimizer': optimizer.state_dict(),
    }, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=True)


def test_disabled_legacy_controller_does_not_add_model_state_or_change_predictions():
    agent = DummyAgent()
    inputs = torch.randn(3, 3)
    original = copy.deepcopy(agent.state_dict())
    prediction = agent(inputs).detach()
    assert rn.configure(agent, {'training': {'reward_norm_mode': 'legacy'}}) is None
    assert not hasattr(agent.critic, '_value_normalizer')
    assert set(agent.state_dict()) == set(original)
    torch.testing.assert_close(agent(inputs), prediction)
    for key, value in rn.inference_model_state(agent).items():
        torch.testing.assert_close(value, original[key])


def test_normalizer_is_unregistered_and_legacy_model_keys_remain_strictly_loadable():
    agent = DummyAgent()
    before_keys = set(agent.state_dict())
    before_parameters = {name: id(value) for name, value in agent.named_parameters()}
    controller = rn.configure(agent, _cfg())
    assert getattr(agent.critic, '_value_normalizer') is controller.critic
    assert set(agent.state_dict()) == before_keys
    assert {name: id(value) for name, value in agent.named_parameters()} == before_parameters
    assert '_value_normalizer' not in agent.critic._modules
    assert not any('normaliz' in key for key in agent.state_dict())
    DummyAgent().load_state_dict(rn.inference_model_state(agent), strict=True)


def test_first_rollout_calibration_then_previous_history_scale_used_for_whole_next_rollout():
    agent, controller, optimizer = _make_normalized()
    valid = torch.tensor([True, True, False])
    returns = torch.tensor([-10., -30., float('nan')])
    adv = torch.tensor([3., 4., float('nan')])
    first = controller.begin_rollout_update(returns, adv, valid, agent.critic, optimizer)
    rms = (25 / 2)**.5
    torch.testing.assert_close(first, torch.tensor([3 / rms, 4 / rms, 0.]))
    assert controller.diagnostics['actor_calibration_rollout'] is True
    assert controller.critic.mean.item() == -20
    assert controller.critic.std.item() == 10
    assert controller.actor.sample_count.item() == 2
    frozen = controller.actor.snapshot()
    second = controller.begin_rollout_update(returns * 10, adv * 10, valid, agent.critic, optimizer)
    torch.testing.assert_close(second, torch.tensor([30 / rms, 40 / rms, 0.]))
    assert controller.diagnostics['actor_calibration_rollout'] is False
    assert controller.scale_used.item() == frozen.item()
    assert controller.actor.snapshot().item() > frozen.item()
    # Route advantages use exactly the same past-history scale as PPO.
    objectives = torch.tensor([[10., 20., 40.]])
    expected_raw = torch.tensor([[10., 2.5, -12.5]])
    torch.testing.assert_close(controller.route_advantages(objectives, torch.ones_like(objectives, dtype=torch.bool)), expected_raw / frozen.float())
    assert controller.actor.update_count.item() == 2


def test_leave_one_out_advantages_are_physical_cost_differences_in_fixed_reward_units():
    objectives = torch.tensor([[10., 20., 40., float('nan')], [2., 100., 200., 300.], [float('nan')] * 4], requires_grad=True)
    feasible = torch.tensor([[True, True, True, False], [True, False, False, False], [False] * 4])
    actual = rn.leave_one_out_cost_advantages(objectives, feasible, 2.)
    torch.testing.assert_close(actual, torch.tensor([[10., 2.5, -12.5, 0.], [0.] * 4, [0.] * 4]))
    assert not actual.requires_grad
    assert actual[0].sum().item() == 0
    # A change in physical units is canceled only by the common reward-unit change.
    torch.testing.assert_close(rn.leave_one_out_cost_advantages(objectives * 1000, feasible, 2000.), actual)
    torch.testing.assert_close(rn.leave_one_out_cost_advantages(objectives * 10, feasible, 2.), actual * 10)


def test_popart_rescales_only_adam_head_moments_and_preserves_current_physical_predictions():
    agent, controller, optimizer = _make_normalized()
    x = torch.randn(6, 3, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(agent(x), torch.arange(6, dtype=torch.float64)).backward()
    optimizer.step()
    previous_states = {name: copy.deepcopy(optimizer.state[param]) for name, param in agent.named_parameters() if param in optimizer.state}
    before = agent(x).detach().clone()
    returns = torch.tensor([-10., -20., -30., -60., -100., -150.])
    controller.begin_rollout_update(returns, torch.arange(6, dtype=torch.float32), torch.ones(6, dtype=torch.bool), agent.critic, optimizer)
    factor = controller.diagnostics['critic_head_moment_scale']
    assert factor != 1
    torch.testing.assert_close(agent(x), before, rtol=1e-10, atol=1e-10)
    for name, parameter in agent.named_parameters():
        if name not in previous_states:
            continue
        state, old = optimizer.state[parameter], previous_states[name]
        torch.testing.assert_close(state['step'], old['step'])
        for key, power in [('exp_avg', 1), ('exp_avg_sq', 2), ('max_exp_avg_sq', 2)]:
            expected = old[key] * (factor**power if name.startswith('critic.head.') else 1.)
            torch.testing.assert_close(state[key], expected)


def test_inference_state_folds_popart_without_mutation_or_legacy_evaluator_dependencies():
    agent, controller, optimizer = _make_normalized()
    _one_update(agent, controller, optimizer)
    head_before = copy.deepcopy(agent.critic.head.state_dict())
    x = torch.randn(10, 3, dtype=torch.float64)
    physical_values = agent(x).detach()
    state = rn.inference_model_state(agent)
    legacy = DummyAgent().double()
    legacy.load_state_dict(state, strict=True)
    assert not hasattr(legacy.critic, '_value_normalizer')
    torch.testing.assert_close(legacy(x), physical_values, atol=1e-12, rtol=1e-12)
    for key, value in agent.critic.head.state_dict().items():
        torch.testing.assert_close(value, head_before[key])
    repeated = rn.inference_model_state(agent)
    for key, value in state.items():
        torch.testing.assert_close(value, repeated[key])


def test_weights_only_initialization_uses_folded_head_and_fresh_statistics():
    agent, controller, optimizer = _make_normalized()
    _one_update(agent, controller, optimizer)
    checkpoint = _checkpoint_roundtrip(agent, controller, optimizer)
    fresh = DummyAgent().double()
    fresh.load_state_dict(checkpoint['model_state_dict'], strict=True)
    fresh_controller = rn.configure(fresh, _cfg())
    x = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(fresh(x), agent(x))
    assert fresh_controller.actor.update_count.item() == fresh_controller.critic.update_count.item() == 0
    assert fresh_controller.critic.mean.item() == 0
    assert fresh_controller.critic.std.item() == 1


def test_exact_epoch_boundary_resume_matches_next_optimizer_update():
    agent, controller, optimizer = _make_normalized()
    _one_update(agent, controller, optimizer)
    _one_update(agent, controller, optimizer, multiplier=2.)
    checkpoint = _checkpoint_roundtrip(agent, controller, optimizer)
    resumed = DummyAgent().double()
    resumed.load_state_dict(checkpoint['model_state_dict'], strict=True)
    resumed._pending_reward_normalization_state = checkpoint['normalization']
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.003, amsgrad=True)
    resumed_optimizer.load_state_dict(checkpoint['optimizer'])
    resumed_controller = rn.configure(resumed, _cfg(), resume=True)
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, atol=0, rtol=0)
    expected_adv = _one_update(agent, controller, optimizer, multiplier=3.)
    actual_adv = _one_update(resumed, resumed_controller, resumed_optimizer, multiplier=3.)
    torch.testing.assert_close(actual_adv, expected_adv, atol=0, rtol=0)
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, atol=0, rtol=0)
    for field in ('actor', 'critic'):
        for key, value in getattr(controller, field).state_dict().items():
            torch.testing.assert_close(getattr(resumed_controller, field).state_dict()[key], value, atol=0, rtol=0)
    for expected_group, actual_group in zip(optimizer.param_groups, resumed_optimizer.param_groups):
        for expected_parameter, actual_parameter in zip(expected_group['params'], actual_group['params']):
            for key, value in optimizer.state[expected_parameter].items():
                torch.testing.assert_close(resumed_optimizer.state[actual_parameter][key], value, atol=0, rtol=0)


@pytest.mark.parametrize('change', [
    {'training': {'gamma': .99}},
    {'training': {'normalization_beta': .1}},
    {'env': {'reward_distance_scale_km': 20.}},
])
def test_resume_rejects_changed_reward_or_normalization_signature(change):
    agent, controller, optimizer = _make_normalized()
    _one_update(agent, controller, optimizer)
    resumed = DummyAgent().double()
    resumed._pending_reward_normalization_state = controller.checkpoint_state(agent.critic)
    cfg = _cfg()
    for section, values in change.items():
        cfg[section].update(values)
    with pytest.raises(ValueError, match='changed on resume'):
        rn.configure(resumed, cfg, resume=True)


def test_resume_mode_switches_are_rejected_instead_of_reusing_incompatible_optimizer():
    agent = DummyAgent()
    with pytest.raises(ValueError, match='Legacy optimizer'):
        rn.configure(agent, _cfg(), resume=True)
    agent = DummyAgent()
    agent._pending_reward_normalization_state = {'present': True}
    with pytest.raises(ValueError, match='normalized optimizer as legacy'):
        rn.configure(agent, {'training': {'reward_norm_mode': 'legacy'}}, resume=True)


@pytest.mark.parametrize('case', ['ddp', 'no_gae', 'decomposed', 'pbrs', 'method'])
def test_unsupported_host_modes_fail_before_training(case):
    agent = DummyAgent()
    cfg = _cfg()
    distributed = None
    if case == 'ddp':
        distributed = SimpleNamespace(enabled=True)
    elif case == 'no_gae':
        cfg['training']['use_gae'] = False
    elif case == 'decomposed':
        agent.critic.use_decomposed_critic = True
    elif case == 'pbrs':
        cfg['pbrs'] = {'use_customer_pbrs': True}
    elif case == 'method':
        cfg['offline']['method'] = 'unknown'
    with pytest.raises(ValueError):
        rn.configure(agent, cfg, distributed=distributed)


def _strict_cfg():
    cfg = _cfg(ppo_loss_reduction='valid_actions')
    cfg['env'].update(reward_contract='strict_distance', failure_penalty_km=1000.)
    cfg['pbrs'] = {'use_customer_pbrs': True, 'use_repair_distance_pbrs': True}
    return cfg


def test_strict_terminal_correct_potential_is_accepted_and_recorded_in_resume_signature():
    agent = DummyAgent()
    controller = rn.configure(agent, _strict_cfg())
    assert controller.signature['reward_contract'] == 'strict_distance'
    assert controller.signature['potential_config']['strict_contract']
    assert controller.signature['potential_config']['gamma'] == 1.
    assert controller.loss_reduction == 'valid_actions_with_length_normalized_route_SL'
    resumed = DummyAgent()
    resumed._pending_reward_normalization_state = controller.checkpoint_state(agent.critic)
    changed = _strict_cfg()
    changed['pbrs']['repair_progress_coef'] = .8
    with pytest.raises(ValueError, match='changed on resume'):
        rn.configure(resumed, changed, resume=True)


@pytest.mark.parametrize('changes', [
    {'training': {'gamma': .99}},
    {'env': {'failure_penalty_km': None}},
    {'env': {'success_bonus': 1.}},
    {'pbrs': {'pbrs_clip': .2}},
    {'pbrs': {'use_terminal_heuristic': True}},
])
def test_strict_normalization_cannot_hide_an_invalid_reward_contract(changes):
    cfg = _strict_cfg()
    for section, values in changes.items():
        cfg[section].update(values)
    with pytest.raises(ValueError):
        rn.configure(DummyAgent(), cfg)
