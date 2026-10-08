"""A shared forward must preserve joint PPO and solution-level gradients."""
import copy
import numpy as np
import pytest
import torch

from test_rollout_static_cache import _agent, _run
from offline2online.ppo_protocol import chunk_weight
from offline2online.trainer import (
    _evaluate_policy_loss_with_stats, _compute_solution_level_ppo_loss,
    _prepare_solution_level_ppo_weights, _compute_solution_level_weighted_logprob_loss,
    _policy_chunk_evaluations,
)


@pytest.mark.parametrize('width', [1, 3, 100])
def test_valid_action_shared_forward_preserves_ppo_plus_sl_gradients(width):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = _agent()
        batch = _run(agent)
        with torch.no_grad():
            for parameter in agent.backbone.parameters():
                parameter.add_(torch.randn_like(parameter) * .002)
        reference, actual = copy.deepcopy(agent), copy.deepcopy(agent)
        cfg = {'training': {'ppo_loss_reduction': 'valid_actions', 'vf_coef': .5, 'ent_coef': .01},
               'offline': {'sl_coef': .35, 'sl_clip_coef': .2, 'only_success_route_loss': True}}
        indices = np.arange(batch.actions.shape[1])
        returns, advantages = batch.rewards.cumsum(0), torch.randn_like(batch.rewards)
        route_advantages = torch.tensor([[1., -.5, .2], [-.3, .1, 1.3]])
        success = torch.tensor(np.stack([info['success'] for info in batch.final_infos]))
        policy = _evaluate_policy_loss_with_stats(reference, batch, returns, advantages, cfg)[0]
        solution = _compute_solution_level_ppo_loss(reference, batch, route_advantages, success, cfg, indices, 'cpu')[0]
        (policy + .35 * solution).backward()
        weights, counts, _ = _prepare_solution_level_ppo_weights(actual, batch, route_advantages, success, cfg, indices, 'cpu')
        for start in range(0, len(batch.observations), width):
            end = min(start + width, len(batch.observations))
            weight = chunk_weight(batch.valid, start, end, 'valid_actions')
            if not weight:
                continue
            evaluations = _policy_chunk_evaluations(actual, batch, indices, start, end)
            policy = _evaluate_policy_loss_with_stats(actual, batch, returns, advantages, cfg,
                       env_indices=indices, step_start=start, step_end=end, evaluations=evaluations)[0]
            solution = _compute_solution_level_weighted_logprob_loss(actual, batch, weights, counts,
                       indices, 'cpu', start, end, evaluations=evaluations)
            ((policy + .35 * solution / weight) * weight).backward()
        left, right = dict(reference.named_parameters()), dict(actual.named_parameters())
        for name in left:
            if left[name].grad is None:
                assert right[name].grad is None
            else:
                torch.testing.assert_close(left[name].grad, right[name].grad, rtol=1e-4, atol=3e-6, msg=name)
    finally:
        torch.set_num_threads(previous_threads)
