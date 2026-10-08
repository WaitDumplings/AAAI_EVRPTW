"""Explicit PPO sample weighting and fresh-policy trust-region diagnostics."""
from __future__ import annotations

import math
import numpy as np
import torch


def reduction_mode(cfg):
    mode = str(cfg.get('training', {}).get('ppo_loss_reduction', 'legacy_step_mean'))
    if mode not in {'legacy_step_mean', 'valid_actions'}:
        raise ValueError('ppo_loss_reduction must be legacy_step_mean or valid_actions')
    return mode


def chunk_weight(valid, start, end, mode):
    if mode == 'legacy_step_mean':
        return (end - start) / max(len(valid), 1)
    return float(valid[start:end].sum()) / max(float(valid.sum()), 1.)


def reduce_step_means(means, counts, mode):
    values = torch.stack(means)
    if mode == 'legacy_step_mean':
        return values.mean()
    weights = torch.stack(counts).to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)


@torch.no_grad()
def _fresh_policy_kl_impl(agent, batch, slice_observation, *, minibatch_size=16, distributed=None):
    """Re-encode with the current weights; sum/count every valid rollout action.

    This is a sampled old-to-current KL estimator, not an exact distribution KL
    or a hard upper bound. Called after the joint PPO/critic/SL optimizer passes.
    Every distributed worker participates in the same sum/count collective.
    """
    device = batch.old_logprobs.device
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for start in range(0, batch.actions.shape[1], minibatch_size):
        indices = np.arange(start, min(start + minibatch_size, batch.actions.shape[1]))
        state = agent.backbone.encode(slice_observation(batch.observations[0], indices))
        for t, obs in enumerate(batch.observations):
            valid = batch.valid[t, indices]
            if not bool(valid.any()):
                continue
            _, new, _, _, _ = agent.get_action_and_value_cached(
                slice_observation(obs, indices), action=batch.actions[t, indices].long(), state=state)
            delta = (new.float() - batch.old_logprobs[t, indices].float())[valid].double()
            terms = torch.expm1(delta) - delta
            finite = torch.isfinite(terms)
            totals[0] += torch.where(finite, terms, 0.).sum()
            totals[1] += terms.numel()
            totals[2] += (~finite).sum()
    if distributed is not None and distributed.enabled:
        torch.distributed.all_reduce(totals)
    if totals[2].item() or not math.isfinite(float(totals[0])):
        raise FloatingPointError('Nonfinite fresh-policy KL; stopping before another PPO pass')
    return {'post_update_kl': float(totals[0] / totals[1].clamp_min(1)),
            'post_update_kl_valid_actions': int(totals[1]),
            'post_update_kl_scope': 'fresh_policy_all_valid_rollout_actions'}


@torch.no_grad()
def fresh_policy_kl(agent, batch, slice_observation, *, minibatch_size=16, distributed=None):
    # PPO likelihood replay requires the same deterministic network semantics as
    # collection. Do not silently measure dropout noise as policy movement.
    for module in agent.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training and module.p > 0:
            raise ValueError('Fresh-policy KL requires deterministic likelihood replay; active dropout is unsupported')
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError('Fresh-policy KL requires a deterministic policy without training-mode batch normalization')
    device = batch.old_logprobs.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        return _fresh_policy_kl_impl(agent, batch, slice_observation,
                                    minibatch_size=minibatch_size, distributed=distributed)


def rollout_quality_diagnostics(batch):
    """Physical costs only, with explicit feasible coverage and group diversity."""
    costs = np.stack([np.asarray(x['objective_distance_km'], dtype=float) for x in batch.final_infos])
    valid = np.stack([np.asarray(x['success'], dtype=bool) for x in batch.final_infos]) & np.isfinite(costs)
    stds, gains, first_coverage = [], [], []
    actions = batch.actions.detach().cpu().numpy()
    for i in range(len(costs)):
        c = costs[i, valid[i]]
        if c.size:
            stds.append(float(c.std()))
            prefix = costs[i, :min(10, costs.shape[1])]
            prefix_ok = valid[i, :len(prefix)]
            if prefix_ok.any():
                gains.append(float(prefix[prefix_ok].min() - c.min()))
        first_coverage.append(float(len(np.unique(actions[0, i])) / max(actions.shape[2], 1)))
    return {'group_cost_std_km_mean': float(np.mean(stds)) if stds else 0.,
            'near_zero_cost_group_fraction': float(np.mean(np.asarray(stds) < 1e-6)) if stds else 1.,
            'sample_best10_to_bestK_gain_km': float(np.mean(gains)) if gains else 0.,
            'first_action_unique_fraction': float(np.mean(first_coverage)),
            'rollout_trajectory_feasible_fraction': float(valid.mean())}


def protocol_signature(cfg):
    """Settings which cannot silently change across an exact optimizer resume."""
    t, e, o = (cfg.get(k, {}) or {} for k in ('training', 'env', 'offline'))
    return {
        'version': 1,
        'gamma': float(t.get('gamma', .99)),
        'gae_lambda': float(t.get('gae_lambda', .95)),
        'ppo_loss_reduction': reduction_mode(cfg),
        'bootstrap_truncation': bool(t.get('bootstrap_truncation', False)),
        'target_kl': t.get('target_kl'),
        'reward_contract': e.get('reward_contract', 'legacy'),
        'failure_penalty_km': e.get('failure_penalty_km'),
        'prefer_explicit_edge_matrices': bool(e.get('prefer_explicit_edge_matrices', False)),
        'replay_selection': o.get('policy_replay_selection', 'legacy'),
        'branch_exploration_enabled': bool(o.get('branch_exploration_enabled', o.get('exploration_enabled', False))),
        'exploration_interval': int(o.get('exploration_interval', 5)),
        'exploration_instances': int(o.get('exploration_instances', 8)),
        'exploration_trajectories': int(o.get('exploration_trajectories', 8)),
        'exploration_temperature': float(o.get('exploration_temperature', 1.2)),
        'exploration_prefix_fractions': list(o.get('exploration_prefix_fractions', [0., .1, .25, .5])),
        'exploration_max_prefix_steps': int(o.get('exploration_max_prefix_steps', 32)),
        'exploration_exclude_anchor_action': bool(o.get('exploration_exclude_anchor_action', True)),
        'exploration_force_anchor': bool(o.get('exploration_force_anchor', False)),
        'exploration_prefer_stagnant_archive': bool(o.get('exploration_prefer_stagnant_archive', True)),
        'exploration_seed': int(o.get('exploration_seed', t.get('post_init_seed', cfg.get('seed', 3009)))),
    }


def validate_protocol(cfg, *, decomposed_critic=False):
    t, e, o = (cfg.get(k, {}) or {} for k in ('training', 'env', 'offline'))
    mode = reduction_mode(cfg)
    if decomposed_critic and (mode == 'valid_actions' or t.get('bootstrap_truncation', False)):
        raise ValueError('The new PPO protocol currently requires a scalar critic (use_decomposed_critic=false)')
    if mode == 'valid_actions' and int(t.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('valid_actions currently requires gradient_accumulation_steps=1; accumulation would reweight unequal valid-action counts')
    target = t.get('target_kl')
    if target is not None and (not math.isfinite(float(target)) or float(target) <= 0):
        raise ValueError('target_kl must be a finite positive threshold, or null for fixed PPO passes')
    if int(t.get('post_update_kl_interval', 0)) < 0:
        raise ValueError('post_update_kl_interval must be nonnegative')
    if e.get('reward_contract') == 'strict_distance' and float(t.get('gamma', .99)) != 1.:
        raise ValueError('strict_distance finite-route optimization requires gamma=1')
    if o.get('policy_replay_selection', 'legacy') not in {'legacy', 'structural'}:
        raise ValueError('policy_replay_selection must be legacy or structural')
    if o.get('branch_exploration_enabled', o.get('exploration_enabled', False)):
        if not o.get('policy_replay_enabled', False) or o.get('policy_replay_selection') != 'structural':
            raise ValueError('Branch exploration requires structural verified policy replay')
        if int(o.get('policy_replay_exploration_capacity', 0)) < 1:
            raise ValueError('Branch exploration requires a separate exploration reservoir')
        for name in ('interval', 'instances', 'trajectories'):
            if int(o.get('exploration_' + name, {'interval': 5, 'instances': 8, 'trajectories': 8}[name])) < 1:
                raise ValueError('Exploration interval/instances/trajectories must be positive')
        temp = float(o.get('exploration_temperature', 1.2))
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError('Exploration temperature must be finite and positive')
