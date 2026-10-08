"""Optional, separately budgeted routing search; never an on-policy PPO batch.

Fresh environments replay verified historical prefixes and sample completions.
Only independently verified complete solutions are admitted to the route pools.
All random state and model train/eval flags are restored before returning.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import random
import time
from typing import Any

import numpy as np
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import encode_static_rollout, stack_observations
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    build_pbrs_config, pbrs_scale_for_epoch, set_pbrs_reward_scale,
)


@contextmanager
def _search_rng(seed, device):
    """Seed only the CPU and selected CUDA generator, preserving all train RNGs."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(int(seed))
            random.seed(int(seed))
            np.random.seed(int(seed) % 2**32)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _instance_seed(seed, instance_id):
    digest = int.from_bytes(hashlib.sha256(str(instance_id).encode()).digest()[:4], 'little')
    return (int(seed) + digest) % (2**32)


def run_branch_exploration(agent, instances, instance_ids, pool, cfg, epoch, device) -> dict[str, Any]:
    """Run bounded, independent prefix searches at the configured interval.

    All options are in ``offline``. ``branch_exploration_enabled`` defaults to
    false (``exploration_enabled`` is an older alias). The defaults are interval
    5, 8 instances, 8 trajectories each, temperature 1.2, prefix fractions
    [0, .1, .25, .5], prefix cap 32. The total action horizon is bounded by
    ``training.rollout_steps``. Instances must belong to ``pool.instances``.
    Enabled search defaults to prioritizing stagnant exploration archives from
    supplied, registered training instances; remaining slots use the current
    instance_ids. Set ``exploration_prefer_stagnant_archive=false`` for current
    instances only. Returned counts explicitly include extra trajectories and
    actions; no observations, likelihoods or rewards are returned for PPO.
    """
    offline = cfg.get('offline', {}) or {}
    if not bool(offline.get('branch_exploration_enabled', offline.get('exploration_enabled', False))):
        return {}
    interval = int(offline.get('exploration_interval', 5))
    if interval < 1:
        raise ValueError('exploration_interval must be positive')
    if int(epoch) <= 0 or int(epoch) % interval:
        return {}
    if pool is None:
        raise ValueError('Independent branch exploration requires a training route pool')
    training = cfg.get('training', {}) or {}
    instance_budget = int(offline.get('exploration_instances', 8))
    trajectories = int(offline.get('exploration_trajectories', 8))
    temperature = float(offline.get('exploration_temperature', 1.2))
    max_steps = int(training.get('rollout_steps', 201))
    prefix_cap = int(offline.get('exploration_max_prefix_steps', 32))
    fractions = tuple(float(value) for value in offline.get('exploration_prefix_fractions', (0, .1, .25, .5)))
    if instance_budget < 1 or trajectories < 1 or max_steps < 1 or prefix_cap < 1:
        raise ValueError('Branch search instance, trajectory, horizon and prefix budgets must be positive')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('exploration_temperature must be finite and positive')
    if not fractions or any(not math.isfinite(value) or not 0 <= value < 1 for value in fractions):
        raise ValueError('exploration_prefix_fractions must be a nonempty sequence in [0,1)')
    supplied = instances.values() if isinstance(instances, dict) else instances
    available = {str(instance.instance_id): instance for instance in supplied}
    requested = list(dict.fromkeys(str(instance_id) for instance_id in instance_ids))
    unknown = [instance_id for instance_id in requested if instance_id not in available or instance_id not in pool.instances]
    if unknown:
        raise ValueError('Branch search may only use supplied training-pool instances: ' + ', '.join(unknown[:3]))
    # Use pool-owned instances, so a same-ID foreign object cannot change the task.
    # Training pools can take many epochs to revisit an instance. Restricting
    # search to the current rollout would leave historical prefixes dormant.
    prefer_archive = bool(offline.get('exploration_prefer_stagnant_archive', True))
    eligible_archive = []
    if prefer_archive:
        eligible_archive = sorted(instance_id for instance_id, routes in pool.exploration_routes.items()
                                  if routes and instance_id in available and instance_id in pool.instances
                                  and int(epoch) - pool.best_improvement_epoch.get(instance_id, int(epoch))
                                  >= pool.exploration_stagnation_epochs)
    event = int(epoch) // interval - 1
    archive_count = min(instance_budget, len(eligible_archive))
    archive_offset = event * instance_budget % max(len(eligible_archive), 1)
    selected = [eligible_archive[(archive_offset + index) % len(eligible_archive)] for index in range(archive_count)]
    selected_set = set(selected)
    offset = event * instance_budget % max(len(requested), 1)
    for index in range(len(requested)):
        if len(selected) >= instance_budget:
            break
        instance_id = requested[(offset + index) % len(requested)]
        if instance_id not in selected_set:
            selected.append(instance_id)
            selected_set.add(instance_id)
    count = len(selected)
    seed = int(offline.get('exploration_seed', training.get('post_init_seed', cfg.get('seed', 3009)))) + 71_000_003 + int(epoch) * 1009
    seed %= 2**63 - 1
    env_cfg = dict(cfg.get('env', {}) or {})
    # Exactly the ordinary training factory and PBRS schedule; rewards are not
    # used for search ranking, which always uses physical objective_distance_km.
    pbrs = build_pbrs_config(cfg)
    pbrs_scale = pbrs_scale_for_epoch(cfg, int(epoch), int(training.get('epochs', max_steps)))
    exclude_next = bool(offline.get('exploration_exclude_anchor_action', True))
    force_anchor = bool(offline.get('exploration_force_anchor', False))
    diagnostic = {
        'branch_search_enabled': True, 'branch_search_epoch': int(epoch),
        'branch_search_scope': 'independent_search_only; excluded_from_onpolicy_PPO',
        'branch_search_seed': seed, 'branch_search_temperature': temperature,
        'branch_search_configured_instance_budget': instance_budget,
        'branch_search_configured_trajectories_per_instance': trajectories,
        'branch_search_max_steps_per_trajectory': max_steps,
        'branch_search_extra_trajectory_budget': count * trajectories,
        'branch_search_extra_action_budget': count * trajectories * max_steps,
        'branch_search_total_action_budget_including_verification': 2 * count * trajectories * max_steps,
        'branch_search_verification_action_upper_bound': 0,
        'branch_search_instances': count, 'branch_search_sampled_trajectories': 0,
        'branch_search_stagnant_archive_instances': archive_count,
        'branch_search_new_current_instances': count - archive_count,
        'branch_search_eligible_stagnant_archive_instances': len(eligible_archive),
        'branch_search_instance_selection': 'stagnant_archive_then_current' if prefer_archive else 'current_only',
        'branch_search_selected_instance_ids': selected.copy(),
        'branch_search_completed_trajectories': 0, 'branch_search_feasible_trajectories': 0,
        'branch_search_verified_routes': 0, 'branch_search_elite_additions': 0,
        'branch_search_exploration_additions': 0, 'branch_search_prefix_steps': 0,
        'branch_search_free_steps': 0, 'branch_search_action_steps': 0,
        'branch_search_vector_env_steps': 0, 'branch_search_anchor_instances': 0,
        'branch_search_fresh_start_trajectories': 0, 'branch_search_invalid_prefixes': 0,
        'branch_search_excluded_anchor_actions': 0, 'branch_search_mask_violations': 0,
        'branch_search_improved_instance_count': 0,
    }
    staleness, improvements, costs = [], [], []
    modes = [(module, module.training) for module in agent.modules()]
    start = time.perf_counter()
    try:
        with _search_rng(seed, device), torch.no_grad():
            agent.eval()
            for instance_id in selected:
                instance = pool.instances[instance_id]
                anchor = pool.choose_exploration_anchor(instance_id, int(epoch), max_prefix_steps=prefix_cap, force=force_anchor)
                full_anchor = tuple(anchor['full_actions']) if anchor else ()
                if anchor:
                    diagnostic['branch_search_anchor_instances'] += 1
                    staleness.append(float(anchor['stagnation_epochs']))
                prefix_lengths = np.asarray([
                    min(prefix_cap, max_steps - 1, len(full_anchor) - 1, int(len(full_anchor) * fractions[index % len(fractions)]))
                    if full_anchor else 0 for index in range(trajectories)
                ], dtype=np.int64)
                prefix_lengths = np.maximum(prefix_lengths, 0)
                diagnostic['branch_search_fresh_start_trajectories'] += int((prefix_lengths == 0).sum())
                previous_best = min((route.objective for route in pool.routes.get(instance_id, [])), default=np.inf)
                env = make_terran_env(instance=instance, n_traj=trajectories, pbrs_config=pbrs, **env_cfg)
                try:
                    set_pbrs_reward_scale([env], pbrs_scale)
                    observation, info = env.reset(seed=_instance_seed(seed, instance_id))
                    done = np.zeros(trajectories, dtype=bool)
                    free_started = np.zeros(trajectories, dtype=bool)
                    sequences = [[] for _ in range(trajectories)]
                    static_cache = {}
                    embeddings = None
                    diagnostic['branch_search_sampled_trajectories'] += trajectories
                    for step in range(max_steps):
                        alive = ~done
                        if not alive.any():
                            break
                        mask = np.asarray(observation['action_mask'], dtype=bool)
                        if np.any(alive & ~mask.any(axis=-1)):
                            raise RuntimeError('Independent search found an active trajectory without a feasible action')
                        obs_batch = stack_observations([observation], static_cache=static_cache)
                        if step == 0:
                            embeddings = encode_static_rollout(agent, obs_batch)
                        output = agent.backbone(obs_batch) if embeddings is None else agent.backbone.decode(obs_batch, embeddings)
                        logits = output[0].squeeze(0).float() / temperature
                        allowed = mask.copy()
                        forced = np.full(trajectories, -1, dtype=np.int64)
                        for index in np.flatnonzero(alive):
                            if step < prefix_lengths[index]:
                                action = full_anchor[step]
                                if 0 <= action < mask.shape[-1] and mask[index, action]:
                                    forced[index] = action
                                else:
                                    # Reject the invalid prefix, then continue a fresh free branch.
                                    prefix_lengths[index] = step
                                    diagnostic['branch_search_invalid_prefixes'] += 1
                            if forced[index] < 0 and not free_started[index]:
                                if exclude_next and step < len(full_anchor) and allowed[index].sum() > 1:
                                    next_action = full_anchor[step]
                                    if 0 <= next_action < mask.shape[-1] and allowed[index, next_action]:
                                        allowed[index, next_action] = False
                                        diagnostic['branch_search_excluded_anchor_actions'] += 1
                                free_started[index] = True
                        allowed[done] = False
                        allowed[done, 0] = True
                        mask_tensor = torch.as_tensor(allowed, device=logits.device)
                        logits = logits.masked_fill(~mask_tensor, -torch.inf)
                        logits[torch.as_tensor(done, device=logits.device), 0] = 0
                        if not torch.isfinite(logits[mask_tensor]).all():
                            raise RuntimeError('Nonfinite policy logits on feasible independent-search actions')
                        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
                        forced_mask = forced >= 0
                        actions[forced_mask] = forced[forced_mask]
                        violations = int((~mask[np.flatnonzero(alive), actions[alive]]).sum())
                        diagnostic['branch_search_mask_violations'] += violations
                        if violations:
                            raise RuntimeError('Independent branch search selected an infeasible action')
                        for index in np.flatnonzero(alive):
                            sequences[index].append(int(actions[index]))
                        prefix_steps = int((alive & forced_mask).sum())
                        action_steps = int(alive.sum())
                        diagnostic['branch_search_prefix_steps'] += prefix_steps
                        diagnostic['branch_search_free_steps'] += action_steps - prefix_steps
                        diagnostic['branch_search_action_steps'] += action_steps
                        diagnostic['branch_search_vector_env_steps'] += 1
                        observation, _, terminated, truncated, info = env.step(actions)
                        done |= np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                    feasible = (np.asarray(info.get('success', np.zeros(trajectories)), dtype=bool)
                                & np.asarray(env.unwrapped.terminated, dtype=bool)
                                & ~np.asarray(env.unwrapped.truncated, dtype=bool))
                    objectives = np.asarray(info['objective_distance_km'], dtype=np.float64)
                    feasible &= np.isfinite(objectives) & (objectives > 0)
                    diagnostic['branch_search_completed_trajectories'] += int(done.sum())
                    diagnostic['branch_search_feasible_trajectories'] += int(feasible.sum())
                    for index in np.flatnonzero(feasible):
                        # Identical cached routes may skip replay, hence this is an upper bound.
                        diagnostic['branch_search_verification_action_upper_bound'] += len(sequences[index])
                        outcome = pool.ingest(instance_id, sequences[index], float(objectives[index]), epoch=int(epoch))
                        diagnostic['branch_search_verified_routes'] += int(outcome['verified'])
                        diagnostic['branch_search_elite_additions'] += int(outcome['elite_added'])
                        diagnostic['branch_search_exploration_additions'] += int(outcome['exploration_added'])
                        if outcome['verified']:
                            costs.append(float(objectives[index]))
                    best = min((route.objective for route in pool.routes.get(instance_id, [])), default=np.inf)
                    if np.isfinite(previous_best) and best < previous_best:
                        diagnostic['branch_search_improved_instance_count'] += 1
                        improvements.append((previous_best - best) / previous_best)
                finally:
                    env.close()
    finally:
        for module, training_flag in modes:
            module.training = training_flag
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(torch.device(device))
        diagnostic['branch_search_wall_time_s'] = time.perf_counter() - start
    diagnostic['branch_search_feasible_fraction'] = diagnostic['branch_search_feasible_trajectories'] / max(diagnostic['branch_search_sampled_trajectories'], 1)
    diagnostic['branch_search_verified_fraction'] = diagnostic['branch_search_verified_routes'] / max(diagnostic['branch_search_sampled_trajectories'], 1)
    diagnostic['branch_search_verified_mean_cost_km'] = float(np.mean(costs)) if costs else None
    diagnostic['branch_search_relative_improvement_mean'] = float(np.mean(improvements)) if improvements else 0.0
    diagnostic['branch_search_anchor_staleness_mean_epochs'] = float(np.mean(staleness)) if staleness else 0.0
    diagnostic['branch_search_staleness_scope'] = 'verified_archive_best_no_improvement; not_all_onpolicy_samples'
    diagnostic.update(pool.exploration_diagnostics(int(epoch)))
    return diagnostic
