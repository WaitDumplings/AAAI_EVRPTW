"""Optional physical-unit actor/critic normalization for controlled fine-tuning.

The rollout reward unit is fixed by the host. Customer count never enters a
normalizer. The initial rollout calibrates the actor RMS before its first update;
thereafter each rollout uses the previous EMA snapshot for every PPO pass.
Expert/replay terms remain separate dimensionless auxiliary objectives.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import math
import torch
import torch.distributed as dist
from caliroute.plugins.normalization import ActorAdvantageScale, ScalarPopArt, masked_moments

MODE = 'physical_shared_popart'
SCHEMA = 'physical_shared_popart_v1'


def mode(cfg):
    name = str(cfg.get('training', {}).get('reward_norm_mode', 'legacy'))
    if name not in {'legacy', MODE}:
        raise ValueError(f'Unknown training.reward_norm_mode: {name}')
    return name


def _strict_reward_signature(cfg):
    env = cfg.get('env', {})
    if env.get('reward_contract', 'legacy') != 'strict_distance':
        return None
    train = cfg.get('training', {})
    if float(train.get('gamma', .99)) != 1.:
        raise ValueError('strict_distance reward contract requires gamma=1')
    if not env.get('normalize_reward', True) or env.get('reward_distance_scale_km') is None:
        raise ValueError('strict_distance requires an explicit normalized reward distance unit')
    failure = env.get('failure_penalty_km')
    if failure is None or not math.isfinite(float(failure)) or float(failure) <= 0:
        raise ValueError('strict_distance requires an explicit finite positive failure_penalty_km')
    if env.get('reward_mode', 'distance') != 'distance' or float(env.get('success_bonus', 0.)) != 0.:
        raise ValueError('strict_distance forbids success bonuses')
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
    pbrs = cfg.get('pbrs', {}) or {}
    names = {field.name for field in fields(PotentialRewardConfig)} - {'gamma', 'strict_contract'}
    options = {name: pbrs[name] for name in names if name in pbrs}
    potential = PotentialRewardConfig(gamma=1., strict_contract=True, **options)
    return {'reward_contract': 'strict_distance', 'failure_penalty_km': float(failure),
            'potential_config': asdict(potential),
            'potential_annealing': copy.deepcopy(pbrs.get('annealing', {}))}


def signature(cfg):
    env = cfg.get('env', {})
    unit = float(env.get('reward_distance_scale_km', 1.0)) if env.get('normalize_reward', True) else 1.0
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError('reward_distance_scale_km must be finite and positive')
    train = cfg.get('training', {})
    result = dict(schema=SCHEMA, mode=mode(cfg), reward_unit_km=unit,
                  gamma=float(train.get('gamma', .99)), gae_lambda=float(train.get('gae_lambda', .95)),
                  beta=float(train.get('normalization_beta', .01)),
                  actor_min_scale=float(train.get('actor_min_scale', 1e-4)),
                  critic_min_std=float(train.get('critic_min_std', 1e-4)))
    strict = _strict_reward_signature(cfg)
    if strict is not None:
        result.update(strict)
    if not train.get('require_complete_feasible_rollouts', True):
        if strict is None:
            raise ValueError('Learning from failed rollouts requires strict_distance reward units')
        result['require_complete_feasible_rollouts'] = False
    return result


def leave_one_out_cost_advantages(objectives, feasible, reward_unit_km):
    """A[b,k] = (mean of other feasible costs - own cost) / fixed reward unit.

    A single feasible sample has no independent group baseline and gets zero.
    Neither experts nor per-instance std/size enter the on-policy group.
    """
    work = objectives.detach().float()
    valid = feasible.bool() & torch.isfinite(work)
    safe = torch.where(valid, work, 0.)
    counts = valid.sum(-1, keepdim=True)
    baseline = (safe.sum(-1, keepdim=True) - safe) / (counts - 1).clamp_min(1)
    return torch.where(valid & (counts > 1), (baseline - safe) / reward_unit_km, 0.)


class RewardNormalization:
    def __init__(self, cfg, critic):
        self.signature = signature(cfg)
        self.distributed = False
        self.require_complete_feasible_rollouts = bool(cfg.get('training', {}).get('require_complete_feasible_rollouts', True))
        self.loss_reduction = ('valid_actions_with_length_normalized_route_SL'
                               if cfg.get('training', {}).get('ppo_loss_reduction', 'legacy_step_mean') == 'valid_actions'
                               else 'legacy_step_and_length_normalized_route_surrogates')
        device = critic.head.weight.device
        self.actor = ActorAdvantageScale(self.signature['beta'], self.signature['actor_min_scale']).to(device)
        self.critic = ScalarPopArt(self.signature['beta'], self.signature['critic_min_std']).to(device)
        # State is saved explicitly, so legacy model_state_dict keys remain valid.
        object.__setattr__(critic, '_value_normalizer', self.critic)
        self.scale_used = self.actor.snapshot()
        self.diagnostics = {}

    def synchronize_error(self, error):
        """Every rank fails together before entering the next collective."""
        errors = [error]
        if self.distributed:
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        failures = [f'rank {rank}: {message}' for rank, message in enumerate(errors) if message]
        if failures:
            raise ValueError('Reward normalization contract: ' + '; '.join(failures))

    def validate_rollout(self, batch, objectives, success):
        """Completed failures are valid PPO samples only under an explicit contract.

        A collector cutoff is not a failed environment termination. It remains
        unsupported by this whole-route identity check, even when the generic
        collector can bootstrap it. SL advantages use the separate success mask.
        """
        error, costs, identity_error = None, None, None
        try:
            device = batch.rewards.device
            costs = torch.as_tensor(objectives, dtype=torch.float32, device=device)
            feasible = torch.as_tensor(success, dtype=torch.bool, device=device)
            if costs.shape != batch.rewards.shape[1:] or feasible.shape != costs.shape:
                raise ValueError('Objective/success shape does not match rollout trajectories')
            if not bool(batch.dones[-1].all()):
                raise ValueError('physical_shared_popart requires complete environment episodes; collector cutoff is not terminal failure')
            if not bool(torch.isfinite(costs).all()):
                raise ValueError('physical_shared_popart requires finite traveled distances, including failed episodes')
            if self.require_complete_feasible_rollouts and not bool(feasible.all()):
                raise ValueError('physical_shared_popart requires complete feasible rollouts')
            failure = (~feasible).float() * (float(self.signature.get('failure_penalty_km', 0.)) / self.signature['reward_unit_km'])
            initial = getattr(batch, 'initial_shaping_potential', None)
            offset = initial if initial is not None else 0.
            identity_error = (torch.where(batch.valid, batch.rewards, 0.).sum(0)
                              + costs / self.signature['reward_unit_km'] + failure + offset).abs().max()
            if not bool(torch.isfinite(identity_error)) or float(identity_error) > 1e-4:
                raise ValueError(f'Rollout rewards disagree with physical distance/failure cost: {float(identity_error)}')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        self.synchronize_error(error)
        if self.distributed:
            dist.all_reduce(identity_error, op=dist.ReduceOp.MAX)
        return costs, float(identity_error)

    def _check_statistics_synchronized(self):
        if not self.distributed:
            return
        values = torch.stack([value.to(dtype=torch.float64).reshape(())
                              for module in (self.actor, self.critic) for value in module.state_dict().values()])
        lower, upper = values.clone(), values.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX)
        if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all()) or not torch.equal(lower, upper):
            raise ValueError('Reward normalization statistics diverged across ranks')

    def begin_rollout_update(self, returns, advantages, valid, critic, optimizer):
        error = None
        try:
            if returns.shape != advantages.shape or valid.shape != advantages.shape or valid.dtype != torch.bool:
                raise ValueError('Returns, advantages and boolean valid mask must have matching shapes')
            if returns.device != advantages.device or valid.device != advantages.device:
                raise ValueError('Normalization tensors must be on the same device')
            head = critic.head
            if not isinstance(head, torch.nn.Linear) or head.out_features != 1 or head.bias is None:
                raise ValueError('PopArt requires a biased scalar nn.Linear head')
            if head.weight.dtype not in (torch.float32, torch.float64) or head.bias.dtype != head.weight.dtype:
                raise ValueError('PopArt requires float32/float64 master parameters')
            if not bool(torch.isfinite(head.weight).all() & torch.isfinite(head.bias).all()):
                raise ValueError('PopArt head must be finite')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        self.synchronize_error(error)
        self._check_statistics_synchronized()
        raw = advantages.detach().float()
        # Compute and validate BOTH global moments before mutating either state.
        # Unequal valid-action counts and an empty local rank are count weighted.
        actor_moments = masked_moments(raw, valid, distributed=self.distributed)
        critic_moments = masked_moments(returns.detach().float(), valid, distributed=self.distributed)
        if actor_moments.count.item() == 0:
            raise ValueError('Reward normalization requires at least one globally valid action')
        calibration = self.actor.update_count.item() == 0
        if calibration:
            self.actor.update_from_moments(actor_moments)
        self.scale_used = self.actor.snapshot()
        normalized = self.actor.normalize(raw, self.scale_used, valid)
        if not calibration:
            self.actor.update_from_moments(actor_moments)
        old_std = self.critic.std.detach().clone()
        old_mean = self.critic.mean.detach().clone()
        self.critic.update_from_moments(critic_moments, critic.head)
        factor = old_std / self.critic.std
        # Keep Adam's historical normalized-loss gradient moments in the new
        # target units; this does not promise optimizer-trajectory invariance.
        for parameter in (critic.head.weight, critic.head.bias):
            state = optimizer.state.get(parameter, {})
            for key, power in [('exp_avg', 1), ('exp_avg_sq', 2), ('max_exp_avg_sq', 2)]:
                if key in state:
                    state[key].mul_(factor.to(state[key]) ** power)
        raw_second_moment = actor_moments.variance + actor_moments.mean.square()
        self.diagnostics = {
            'mode': MODE, 'schema': SCHEMA, 'reward_unit_km': self.signature['reward_unit_km'],
            'gamma': self.signature['gamma'], 'actor_scale_used': float(self.scale_used),
            'actor_scale_next': float(self.actor.snapshot()), 'actor_calibration_rollout': calibration,
            'actor_raw_advantage_mean': float(actor_moments.mean),
            'actor_raw_advantage_std': float(actor_moments.variance.sqrt()),
            'critic_mean': float(self.critic.mean), 'critic_std': float(self.critic.std),
            'critic_rollout_target_mse_raw': float(raw_second_moment),
            'critic_rollout_target_mse_normalized': float(raw_second_moment / self.critic.std.square()),
            'critic_previous_mean': float(old_mean), 'critic_previous_std': float(old_std),
            'critic_head_moment_scale': float(factor),
            'normalizer_updates': int(self.actor.update_count),
            'normalizer_valid_actions': int(actor_moments.count),
            'normalizer_statistics_scope': 'global_valid_actions' if self.distributed else 'local_valid_actions',
            'normalizer_world_size': dist.get_world_size() if self.distributed else 1,
            'auxiliary_units': 'legacy_dimensionless_expert_and_replay_weights; excluded_from_actor_RMS',
            'loss_reduction': self.loss_reduction,
        }
        return normalized

    def route_advantages(self, objectives, feasible):
        raw = leave_one_out_cost_advantages(objectives, feasible, self.signature['reward_unit_km'])
        normalized = self.actor.normalize(raw, self.scale_used, feasible)
        selected = raw[feasible]
        self.diagnostics.update(sl_raw_advantage_std=float(selected.std(unbiased=False)) if selected.numel() else 0.,
                                sl_normalized_advantage_std=float(normalized[feasible].std(unbiased=False)) if selected.numel() else 0.)
        return normalized

    def checkpoint_state(self, critic):
        return dict(signature=dict(self.signature), actor=copy.deepcopy(self.actor.state_dict()),
                    critic=copy.deepcopy(self.critic.state_dict()),
                    normalized_head=copy.deepcopy(critic.head.state_dict()))

    def restore(self, state, critic):
        if state['signature'] != self.signature:
            raise ValueError('Normalization/reward units/gamma changed on resume; use weights-only initialization')
        self.actor.load_state_dict(state['actor'], strict=True)
        self.critic.load_state_dict(state['critic'], strict=True)
        critic.head.load_state_dict(state['normalized_head'], strict=True)
        self.scale_used = self.actor.snapshot()


def _configure_local(agent, cfg, *, resume=False):
    name = mode(cfg)
    pending = getattr(agent, '_pending_reward_normalization_state', None)
    if getattr(agent, '_reward_normalization', None) is not None or getattr(agent.critic, '_value_normalizer', None) is not None:
        raise ValueError('Normalization is already configured; use a fresh agent for mode changes')
    if name == 'legacy':
        if resume and pending is not None:
            raise ValueError('Cannot resume a normalized optimizer as legacy; initialize weights instead')
        return None
    train = cfg.get('training', {})
    if not train.get('use_gae', True) or getattr(agent.critic, 'use_decomposed_critic', False):
        raise ValueError('physical_shared_popart requires scalar critic and GAE')
    if cfg.get('offline', {}).get('method', 'ppo') not in {'ppo', 'sl_ppo', 'sl-ppo'}:
        raise ValueError('physical_shared_popart supports PPO and SL-PPO only')
    strict = _strict_reward_signature(cfg)
    if any(cfg.get('pbrs', {}).get(key, False) for key in ('use_customer_pbrs', 'use_repair_distance_pbrs', 'use_feasible_ratio_pbrs', 'use_terminal_heuristic')) and strict is None:
        raise ValueError('physical_shared_popart requires unshaped physical distance rewards or strict_distance terminal-correct potential shaping')
    if resume:
        if pending is None:
            raise ValueError('Legacy optimizer cannot resume as normalized; use weights-only initialization')
        if pending.get('signature') != signature(cfg):
            raise ValueError('Normalization/reward units/gamma changed on resume; use weights-only initialization')
    controller = RewardNormalization(cfg, agent.critic)
    if resume:
        controller.restore(pending, agent.critic)
    agent._reward_normalization = controller
    return controller


def configure(agent, cfg, *, resume=False, distributed=None):
    enabled = distributed is not None and distributed.enabled
    if enabled and not (dist.is_available() and dist.is_initialized()):
        raise ValueError('Distributed reward normalization requires an initialized process group')
    if not enabled:
        return _configure_local(agent, cfg, resume=resume)
    error, controller = None, None
    try:
        controller = _configure_local(agent, cfg, resume=resume)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, dict(error=error, signature=controller.signature if controller else None))
    failures = [f'rank {rank}: {state["error"]}' for rank, state in enumerate(states) if state['error']]
    if failures:
        raise ValueError('Distributed reward normalization configuration: ' + '; '.join(failures))
    if any(state['signature'] != states[0]['signature'] for state in states):
        raise ValueError('Reward normalization configuration differs across ranks')
    if controller is not None:
        controller.distributed = True
        controller._check_statistics_synchronized()
    return controller


def inference_model_state(agent):
    """Checkpoint model weights always predict original reward-unit values.

    Exact normalized training weights/statistics are saved separately for resume.
    Legacy evaluators can load model_state_dict without any normalization module.
    """
    state = agent.state_dict()
    controller = getattr(agent, '_reward_normalization', None)
    if controller is not None:
        std, mean = controller.critic.std, controller.critic.mean
        for name in ['weight', 'bias']:
            key = 'critic.head.' + name
            value = state[key]
            state[key] = value * std.to(value) + (mean.to(value) if name == 'bias' else 0.)
    return state
