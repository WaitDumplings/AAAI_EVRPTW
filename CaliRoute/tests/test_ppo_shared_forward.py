from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_rollout_static_cache import _agent, _env, _run
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from offline2online.models import Agent
from offline2online.trainer import (
    SolutionCandidate, _compute_solution_level_weighted_logprob_loss,
    _evaluate_policy_loss_with_stats, _evaluate_policy_loss_policy_only_with_stats, _evaluate_policy_loss_decomposed,
    _expert_route_mean_logprobs, _policy_chunk_evaluations, _prepare_solution_level_ppo_weights,
)


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _grads(agent):
    return {name: parameter.grad.clone() for name, parameter in agent.named_parameters() if parameter.grad is not None}


@pytest.mark.parametrize("mode", ["standard", "policy_only", "decomposed", "all_model_flags"])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_shared_chunk_loss_and_gradients_match_separate_passes(mode, zero_weights):
    agent = _agent()
    if mode in {"decomposed", "all_model_flags"}:
        agent = Agent(embedding_dim=32, n_encode_layers=1, use_dynamic_decision_encoder=True,
                      use_decomposed_critic=mode == "decomposed",
                      cache_static_observations=mode == "all_model_flags",
                      optimize_dynamic_projections=mode == "all_model_flags",
                      use_residual_edge_bias=mode == "all_model_flags")
    batch = _run(agent)
    # Include both clipped and unclipped, positive and negative route advantages.
    offsets = torch.tensor([[0.5, -0.5, 0.0], [-0.3, 0.3, 0.0]])
    batch.old_logprobs = batch.old_logprobs + offsets
    route_adv = torch.zeros(2, 3) if zero_weights else torch.tensor([[1.0, -1.0, 0.7], [1.2, -0.8, 0.0]])
    cfg = {"training": {"clip_coef": 0.2, "vf_coef": 0.5, "ent_coef": 0.01}, "offline": {"sl_clip_coef": 0.2}}
    indices = np.array([1, 0])
    weights, counts, _ = _prepare_solution_level_ppo_weights(agent, batch, route_adv, torch.ones(2, 3, dtype=torch.bool), cfg, indices, "cpu")
    returns = batch.rewards.cumsum(0)
    advantages = torch.randn_like(batch.rewards)
    total_steps = len(batch.observations)
    baseline = copy.deepcopy(agent)
    shared = copy.deepcopy(agent)
    sums = []
    for model, reuse in ((baseline, False), (shared, True)):
        total_loss = 0.0
        for begin in range(0, total_steps, 3):
            end = min(begin + 3, total_steps)
            chunk_weight = (end - begin) / total_steps
            outputs = _policy_chunk_evaluations(model, batch, indices, begin, end) if reuse else None
            kwargs = dict(env_indices=indices, step_start=begin, step_end=end, evaluations=outputs)
            if mode == "decomposed":
                decomposed = {"total": returns, "boundary": returns * 0.3, "internal": returns * 0.7}
                ppo_loss = _evaluate_policy_loss_decomposed(model, batch, decomposed, advantages, cfg, "cpu", **kwargs)[0]
            elif mode == "policy_only":
                ppo_loss = _evaluate_policy_loss_policy_only_with_stats(model, batch, advantages, cfg, **kwargs)[0]
            else:
                ppo_loss = _evaluate_policy_loss_with_stats(model, batch, returns, advantages, cfg, **kwargs)[0]
            sl_loss = _compute_solution_level_weighted_logprob_loss(model, batch, weights, counts, indices, "cpu", begin, end, evaluations=outputs)
            joint = ppo_loss * chunk_weight + 0.5 * sl_loss
            total_loss += float(joint.detach())
            joint.backward()
        sums.append(total_loss)
    assert sums[0] == pytest.approx(sums[1], abs=1e-7)
    left, right = _grads(baseline), _grads(shared)
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=2e-5, atol=2e-6, msg=key)


def test_zero_route_weights_skip_encoder_and_allow_backward():
    agent = _agent()
    batch = _run(agent)
    with patch.object(agent.backbone, "encode", side_effect=AssertionError("unnecessary encoding")):
        loss = _compute_solution_level_weighted_logprob_loss(agent, batch, torch.zeros(2, 3), torch.ones(2, 3), np.arange(2), "cpu", 0, 2)
    assert float(loss) == 0
    loss.backward()


def _candidate(actions):
    env = make_terran_env(instance=_env().unwrapped.instance, n_traj=1, use_jit_mask=False, info_level="light")
    obs, _ = env.reset()
    observations = []
    for action in actions:
        observations.append(obs)
        obs, _, _, _, _ = env.step(np.array([action]))
    return SolutionCandidate(0, observations, actions, advantage=0.5, gate=1.0)


def test_expert_route_cache_keeps_variable_length_probabilities_and_gradients():
    candidates = [_candidate([1, 4, 2, 0, 3, 0]), _candidate([1, 0, 2, 3, 0])]
    baseline = Agent(embedding_dim=32, n_encode_layers=1, use_dynamic_decision_encoder=True,
                     cache_static_observations=True, optimize_dynamic_projections=True,
                     use_residual_edge_bias=True)
    cached = copy.deepcopy(baseline)
    old = _expert_route_mean_logprobs(baseline, candidates, "cpu", 20)
    with patch.object(cached.backbone.encoder, "forward", wraps=cached.backbone.encoder.forward) as encoder:
        new = _expert_route_mean_logprobs(cached, candidates, "cpu", 20, cache_static=True)
        assert encoder.call_count == 1
    torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-6)
    old.sum().backward()
    new.sum().backward()
    left, right = _grads(baseline), _grads(cached)
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=5e-5, atol=3e-6, msg=key)


@pytest.mark.parametrize("dtype,delta,advantage", [(torch.float16, 12.0, 1.0), (torch.float32, 100.0, 1.0),
                                                   (torch.float32, 100.0, -1.0), (torch.float32, float("nan"), 1.0)])
def test_expert_clipping_avoids_nonfinite_gradients(dtype, delta, advantage):
    from offline2online.trainer import _compute_sl_expert_candidate_loss
    logprob = torch.tensor([delta], dtype=dtype, requires_grad=True)
    candidate = SolutionCandidate(0, [], [1], advantage=advantage, gate=1.0, old_mean_logprob=0.0)
    with patch("offline2online.trainer._expert_route_mean_logprobs", return_value=logprob):
        loss, stats = _compute_sl_expert_candidate_loss(None, [candidate], {"offline": {"sl_clip_coef": 0.2}}, np.array([0]), "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logprob.grad).all()
    if advantage > 0 and np.isfinite(delta):
        assert float(loss) == pytest.approx(-1.2)
        assert float(logprob.grad) == 0.0


def test_stable_expert_surrogate_matches_original_loss_and_gradient():
    from offline2online.trainer import _compute_sl_expert_candidate_loss
    delta = torch.tensor([-2.0, -0.1, 0.1, 2.0, -2.0, -0.1, 0.1, 2.0], requires_grad=True)
    adv = torch.tensor([1.0] * 4 + [-1.0] * 4)
    candidates = [SolutionCandidate(0, [], [1], advantage=float(a), gate=1.0) for a in adv]
    with patch("offline2online.trainer._expert_route_mean_logprobs", return_value=delta):
        loss, _ = _compute_sl_expert_candidate_loss(None, candidates, {"offline": {"sl_clip_coef": 0.2}}, np.array([0]), "cpu")
    loss.backward()
    new_grad = delta.grad.clone()
    old_delta = delta.detach().clone().requires_grad_()
    ratio = old_delta.exp()
    reference = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
    reference.backward()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(new_grad, old_delta.grad)
