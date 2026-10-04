from __future__ import annotations
import numpy as np
import pytest
import torch
from caliroute.plugins.slppo import clipped_route_surrogate, normalized_route_advantages, replay_weight_schedule, solution_level_ppo_loss
from offline2online.slppo_diagnostics import gradient_component_diagnostics, tensors_to_floats, value_and_advantage_diagnostics
from offline2online.trainer import _sl_candidate_improvement_stats


def test_loss_and_chunked_gradient_match_legacy_with_variable_lengths():
    torch.manual_seed(109)
    old = torch.randn(7, 2, 4, dtype=torch.float64)
    new = (old + torch.randn_like(old) * .4).requires_grad_()
    valid = torch.arange(7)[:, None, None] < torch.tensor([[7, 6, 2, 0], [5, 7, 3, 6]])
    adv = torch.tensor([[1., -1., .5, 1.], [0., -.5, 2., -2.]], dtype=torch.float64)
    feasible = torch.tensor([[True, True, False, True], [True, True, True, True]])
    count = valid.sum(0)
    ratio = (((new - old) * valid).sum(0) / count.clamp_min(1)).exp()
    mask = (count > 0) & feasible & (adv != 0)
    expected = -torch.minimum(ratio[mask] * adv[mask], ratio[mask].clamp(.8, 1.2) * adv[mask]).mean()
    result = solution_level_ppo_loss(new, old, valid, adv, feasible=feasible)
    torch.testing.assert_close(result.loss, expected, atol=1e-12, rtol=1e-12)
    actual_grad, = torch.autograd.grad(result.loss, new, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, new)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    chunk_new = new.detach().clone().requires_grad_()
    chunk_loss = -(result.weights / result.valid_counts.clamp_min(1) * chunk_new * valid).sum()
    chunk_grad, = torch.autograd.grad(chunk_loss, chunk_new)
    torch.testing.assert_close(chunk_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_loss_double_gradcheck_away_from_clip_boundaries():
    old = torch.zeros(3, 2, dtype=torch.float64)
    new = torch.full_like(old, .08, requires_grad=True)
    adv = torch.tensor([.6, -.8], dtype=torch.float64)
    valid = torch.tensor([[True, True], [True, True], [False, True]])
    assert torch.autograd.gradcheck(lambda p: solution_level_ppo_loss(p, old, valid, adv).loss, (new,))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_zero_route_and_empty_step_losses_backward(dtype):
    for shape in [(4, 3), (0, 3)]:
        new = torch.zeros(shape, dtype=dtype, requires_grad=True)
        result = solution_level_ppo_loss(new, torch.zeros_like(new), torch.zeros(shape, dtype=torch.bool), torch.ones(3))
        assert result.loss.item() == 0
        assert result.diagnostics['num_routes_used'].item() == 0
        result.loss.backward()
        assert torch.isfinite(new.grad).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_half_precision_large_ratio_and_nonfinite_mask_have_finite_gradient(dtype):
    delta = torch.tensor([30., -30., 0., float('nan'), 90.], dtype=dtype, requires_grad=True)
    result = clipped_route_surrogate(delta, torch.tensor([1., -1., 2., 1., -1.]), torch.ones(5, dtype=torch.bool))
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(delta.grad).all()
    assert result.diagnostics['rejected_nonfinite'] == 2


def test_invalid_masked_steps_do_not_poison_valid_routes():
    new = torch.tensor([[-1., -1.], [float('nan'), -2.]], requires_grad=True)
    old = torch.tensor([[-1., -1.], [float('nan'), -2.]])
    mask = torch.tensor([[True, True], [False, True]])
    result = solution_level_ppo_loss(new, old, mask, torch.ones(2))
    assert result.loss.item() == -1
    result.loss.backward()
    assert torch.isfinite(new.grad).all()
    assert result.route_mask.all()


def test_relative_advantages_are_invariant_to_objective_units_and_ignore_failures():
    objectives = torch.tensor([[100., 110., 120., float('nan')], [210., 211., 212., 1.]], dtype=torch.float64)
    feasible = torch.tensor([[True, True, True, False], [True, True, True, False]])
    reference = torch.tensor([90., 210.], dtype=torch.float64)
    expected = normalized_route_advantages(objectives, feasible, reference)
    for scale in [.001, 1000., 1e6]:
        actual = normalized_route_advantages(objectives * scale, feasible, reference * scale)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(expected).all()
    assert (expected[:, -1] == 0).all()
    equal = normalized_route_advantages(torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool))
    assert (equal == 0).all()


def test_training_candidate_scale_is_unit_invariant_and_legacy_unchanged():
    objective = np.array([100., 102., 104.])
    feasible = np.ones(3, dtype=bool)
    cfg = {'sl_advantage_scale_mode': 'relative', 'sl_relative_std_floor': .01}
    baseline = _sl_candidate_improvement_stats(objective, feasible, 101., cfg)
    for multiplier in [.01, 1000.]:
        result = _sl_candidate_improvement_stats(objective * multiplier, feasible, 101 * multiplier, cfg)
        np.testing.assert_allclose(np.asarray(result) / multiplier, baseline)
    assert _sl_candidate_improvement_stats(objective, feasible, 101., {}) == (102., 1., 5.)


def test_portable_loss_works_with_two_unrelated_backbones():
    torch.manual_seed(107)
    observations = torch.randn(5, 4, 3)
    valid = torch.ones(5, 4, dtype=torch.bool)
    actions = torch.randint(0, 2, (5, 4))
    models = [torch.nn.Linear(3, 2), torch.nn.Sequential(torch.nn.Linear(3, 9), torch.nn.Tanh(), torch.nn.Linear(9, 2))]
    for model in models:
        logits = model(observations)
        selected = logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        result = solution_level_ppo_loss(selected, selected.detach() - .02, valid, torch.tensor([1., -.5, .7, -.8]))
        result.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


def test_replay_schedule_delays_then_ramps_loss():
    assert replay_weight_schedule(25, .1, 25, 75) == 0
    assert replay_weight_schedule(26, .1, 25, 75) == pytest.approx(.1 / 75)
    assert replay_weight_schedule(100, .1, 25, 75) == .1
    assert replay_weight_schedule(1000, .1, 25, 75) == .1
    assert replay_weight_schedule(1, .2) == .2
    with pytest.raises(ValueError):
        replay_weight_schedule(1, -.1)


def test_monitoring_critic_fit_and_gradient_balance_are_detached():
    returns = torch.tensor([1., 2., 4.])
    metrics = tensors_to_floats(value_and_advantage_diagnostics(returns, returns, returns - 2, torch.ones(3, dtype=torch.bool)))
    assert metrics['value_explained_variance'] == 1
    assert metrics['value_rmse'] == 0
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    losses = {'ppo': p.square().sum(), 'sl': 2 * p.square().sum()}
    result = gradient_component_diagnostics(losses, [p])
    assert not any(v.requires_grad for v in result.values())
    assert p.grad is None
    metrics = tensors_to_floats(result)
    assert metrics['grad_ppo_sl_cosine'] == pytest.approx(1)
    assert metrics['grad_ppo_to_sl_ratio'] == pytest.approx(.5)
    sum(losses.values()).backward()
    torch.testing.assert_close(p.grad, 6 * p.detach())


def test_actual_trainer_relative_advantage_excludes_failed_routes_and_is_unit_invariant():
    from types import SimpleNamespace
    from offline2online.trainer import _solution_level_advantage_tensors, _prepare_sl_expert_candidates

    objective = np.array([[100., 120., 1., np.nan], [5., 7., 8., np.nan], [210., 211., 212., 0.1]])
    success = np.array([[True, True, False, True], [False, False, False, False], [True, True, True, False]])
    references = np.array([90., 3., np.nan])
    envs = [SimpleNamespace(instance=SimpleNamespace(instance_id=str(i))) for i in range(3)]
    cfg = {"data": {"num_customers": 50}, "offline": {"method": "slppo"}, "advantage": {
        "use_group_advantage": True, "use_reference_advantage": True, "use_expert_solution_level": True,
        "sl_advantage_scale_mode": "relative", "sl_relative_std_floor": .01,
        "group_adv_coef": .3, "sl_candidate_coef": .1, "reference_adv_coef": .1,
        "reference_advantage_mode": "remaining_gap", "sl_candidate_advantage_mode": "remaining_gap",
        # Relative mode remains feasibility-safe even if legacy opt-outs are supplied.
        "sl_candidate_success_only": False, "reference_success_only": False,
    }}

    def calculate(scale, config=cfg):
        batch = SimpleNamespace(actions=torch.zeros(1, 3, 4), old_logprobs=torch.zeros(1, 3, 4, dtype=torch.float64),
            final_infos=[{"objective_distance_km": row * scale, "success": succeeded,
                          "served_customers": np.where(succeeded, 50, 1)} for row, succeeded in zip(objective, success)])
        buffer = SimpleNamespace(reference_objective=lambda instance_id: references[int(instance_id)] * scale)
        adv, route_success, info = _solution_level_advantage_tensors(batch, config, envs, buffer, "cpu")
        return adv, route_success, info, batch

    expected, valid, info, _ = calculate(1.)
    for scale in [.001, 1000.]:
        actual, actual_valid, _, _ = calculate(scale)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
        assert torch.equal(actual_valid, valid)
    assert torch.equal(valid, torch.from_numpy(success & np.isfinite(objective)))
    assert (expected[~valid] == 0).all()
    assert (expected[1] == 0).all()
    assert info['sl_group_reference_count'] == 1  # No expert-only group for all-failed row.
    # Finite reference missing for row 2: only its feasible group signal survives.
    portable = normalized_route_advantages(torch.tensor(objective[2:3]), torch.tensor(success[2:3])) * .3
    torch.testing.assert_close(expected[2], portable[0], atol=1e-12, rtol=1e-12)

    # Preserve legacy absolute mode's partial-route group scores.
    legacy = {**cfg, 'advantage': {**cfg['advantage'], 'sl_advantage_scale_mode': 'absolute',
                                 'use_reference_advantage': False, 'use_expert_solution_level': False}}
    legacy_adv, _, _, _ = calculate(1., legacy)
    assert (legacy_adv[1] != 0).any()
    # An all-failed rollout cannot activate an expert candidate independently.
    _, _, _, failed_batch = calculate(1.)
    failed_batch.actions = failed_batch.actions[:, 1:2]
    failed_batch.old_logprobs = failed_batch.old_logprobs[:, 1:2]
    failed_batch.final_infos = failed_batch.final_infos[1:2]
    expert_buffer = SimpleNamespace(trajectory_for_instance=lambda _: SimpleNamespace(length=1, objective_distance_km=3.))
    candidates, _ = _prepare_sl_expert_candidates(None, failed_batch, cfg, [envs[1]], expert_buffer, None, 'cpu')
    assert candidates == []
