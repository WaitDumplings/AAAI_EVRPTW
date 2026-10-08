from __future__ import annotations

import copy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip('gymnasium')
from test_rollout_static_cache import _agent, _env
from offline2online.policy_route_replay import PolicyRoutePool
from offline2online.branch_exploration import run_branch_exploration
import offline2online.branch_exploration as branch


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup_search(max_steps=24):
    envs = [_env(), _env(1.3)]
    instances = [env.unwrapped.instance for env in envs]
    env_config = {'use_fast_env': True, 'use_jit_mask': False, 'info_level': 'light'}
    pool = PolicyRoutePool(instances, env_config, structure_enabled=True, exploration_capacity=2,
                           exploration_max_relative_gap=1.0)
    cfg = {'env': env_config, 'training': {'rollout_steps': max_steps, 'epochs': 300, 'post_init_seed': 42},
           'offline': {'branch_exploration_enabled': True, 'exploration_interval': 5,
                       'exploration_instances': 2, 'exploration_trajectories': 4,
                       'exploration_temperature': 1.2}}
    return _agent(), envs, instances, pool, cfg


def test_search_isolates_rng_model_and_onpolicy_environment_and_reports_extra_budget():
    agent, envs, instances, pool, cfg = setup_search()
    agent.train()
    agent.critic.eval()  # Mixed submodule modes must survive, too.
    modes = [module.training for module in agent.modules()]
    for env in envs:
        env.reset(seed=987)
    environments = [(env.unwrapped.last.copy(), env.unwrapped.visited.copy()) for env in envs]
    parameters = {key: value.clone() for key, value in agent.state_dict().items()}
    torch_state, numpy_state, python_state = torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate()
    result = run_branch_exploration(agent, instances, [instance.instance_id for instance in instances], pool, cfg, 5, 'cpu')
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert [module.training for module in agent.modules()] == modes
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
    for env, (last, visited) in zip(envs, environments):
        np.testing.assert_array_equal(env.unwrapped.last, last)
        np.testing.assert_array_equal(env.unwrapped.visited, visited)
    assert result['branch_search_extra_trajectory_budget'] == 8
    assert result['branch_search_extra_action_budget'] == 8 * 24
    assert result['branch_search_sampled_trajectories'] == 8
    assert result['branch_search_action_steps'] <= 8 * 24
    assert result['branch_search_feasible_trajectories'] > 0
    assert result['branch_search_verified_routes'] == result['branch_search_feasible_trajectories']
    assert result['branch_search_mask_violations'] == 0
    assert result['branch_search_wall_time_s'] > 0
    assert result['branch_search_scope'].endswith('excluded_from_onpolicy_PPO')
    assert not any(key in result for key in ('observations', 'old_logprobs', 'returns', 'advantages'))


def test_every_search_action_obeys_fresh_environment_masks_and_prefix_budget():
    agent, _, instances, pool, cfg = setup_search()
    reference = [1, 2, 0, 3, 0]
    for instance in instances:
        cost = sum(float(instance.distance_matrix_km[left, right]) for left, right in zip([0] + reference, reference))
        pool.ingest(instance.instance_id, reference, cost, epoch=1)
    # Deterministic valid history exposes all prefix lengths, irrespective of reservoir quality admission.
    def anchor(instance_id, epoch, **kwargs):
        return {'instance_id': instance_id, 'full_actions': tuple(reference),
                'actions': (1,), 'stagnation_epochs': epoch - 1}
    checked_actions = []
    original_factory = branch.make_terran_env
    def fresh_factory(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        original_step = env.step
        def checked_step(actions):
            alive = ~(env.unwrapped.terminated | env.unwrapped.truncated)
            mask = env.unwrapped._current_action_mask
            assert mask[np.flatnonzero(alive), np.asarray(actions)[alive]].all()
            checked_actions.append(np.asarray(actions).copy())
            return original_step(actions)
        env.step = checked_step
        return env
    with patch.object(pool, 'choose_exploration_anchor', side_effect=anchor), patch.object(branch, 'make_terran_env', side_effect=fresh_factory):
        result = run_branch_exploration(agent, instances, [instance.instance_id for instance in instances], pool, cfg, 15, 'cpu')
    assert checked_actions
    assert result['branch_search_prefix_steps'] > 0
    assert result['branch_search_free_steps'] > 0
    assert result['branch_search_action_steps'] == result['branch_search_prefix_steps'] + result['branch_search_free_steps']
    assert result['branch_search_anchor_instances'] == 2
    assert result['branch_search_excluded_anchor_actions'] > 0
    assert result['branch_search_invalid_prefixes'] == 0
    assert result['branch_search_prefix_steps'] <= 2 * 4 * 32


def test_incomplete_searches_never_enter_either_archive():
    agent, _, instances, pool, cfg = setup_search(max_steps=1)
    result = run_branch_exploration(agent, instances, [instance.instance_id for instance in instances], pool, cfg, 5, 'cpu')
    assert result['branch_search_action_steps'] == 8
    assert result['branch_search_feasible_trajectories'] == 0
    assert result['branch_search_verified_routes'] == 0
    assert not pool.routes and not pool.exploration_routes


def test_disabled_interval_and_foreign_instance_contract():
    agent, _, instances, pool, cfg = setup_search()
    ids = [instance.instance_id for instance in instances]
    with patch.object(branch, 'make_terran_env', side_effect=AssertionError('must not construct search env')):
        assert run_branch_exploration(agent, instances, ids, pool, cfg, 4, 'cpu') == {}
        cfg['offline']['branch_exploration_enabled'] = False
        assert run_branch_exploration(agent, instances, ids, pool, cfg, 5, 'cpu') == {}
    cfg['offline']['branch_exploration_enabled'] = True
    with pytest.raises(ValueError, match='training-pool'):
        run_branch_exploration(agent, instances, ['foreign_test_instance'], pool, cfg, 5, 'cpu')


def test_invalid_prefix_falls_back_to_valid_search_and_restores_rng_on_error():
    agent, _, instances, pool, cfg = setup_search()
    def invalid_anchor(instance_id, epoch, **kwargs):
        return {'full_actions': (999, 1, 2, 0, 3, 0), 'stagnation_epochs': 50}
    with patch.object(pool, 'choose_exploration_anchor', side_effect=invalid_anchor):
        result = run_branch_exploration(agent, instances, [instances[0].instance_id], pool, cfg, 5, 'cpu')
    assert result['branch_search_invalid_prefixes'] > 0
    assert result['branch_search_mask_violations'] == 0
    before = torch.get_rng_state().clone()
    modes = [module.training for module in agent.modules()]
    with patch.object(agent.backbone, 'encode', side_effect=RuntimeError('test inference exception')):
        with pytest.raises(RuntimeError, match='test inference exception'):
            run_branch_exploration(agent, instances, [instances[0].instance_id], pool, cfg, 10, 'cpu')
    assert torch.equal(before, torch.get_rng_state())
    assert [module.training for module in agent.modules()] == modes


def test_search_forwards_strict_physical_reward_and_shared_pbrs_schedule():
    agent, _, instances, _, cfg = setup_search()
    cfg['training']['gamma'] = 1.0
    cfg['env'].update(reward_contract='strict_distance', normalize_reward=True,
                      reward_distance_scale_km=10.0, failure_penalty_km=100.0)
    cfg['pbrs'] = {'use_customer_pbrs': True, 'customer_pbrs_mode': 'progress',
                   'customer_progress_budget': .5}
    pool = PolicyRoutePool(instances, cfg['env'], structure_enabled=True, exploration_capacity=2)
    factory = branch.make_terran_env
    observed = []
    def checked_factory(*args, **kwargs):
        assert kwargs['reward_contract'] == 'strict_distance'
        assert kwargs['reward_distance_scale_km'] == 10.0
        assert kwargs['pbrs_config'].strict_contract
        assert kwargs['pbrs_config'].gamma == 1.0
        observed.append(kwargs)
        return factory(*args, **kwargs)
    with patch.object(branch, 'make_terran_env', side_effect=checked_factory):
        result = run_branch_exploration(agent, instances, [instances[0].instance_id], pool, cfg, 5, 'cpu')
    assert len(observed) == 1
    assert result['branch_search_verified_routes'] > 0
    assert result['branch_search_total_action_budget_including_verification'] == 2 * 4 * 24
    assert result['branch_search_verification_action_upper_bound'] <= 4 * 24


def _seed_stagnant_search_archive(pool, instance, epoch=1):
    # Keep one elite so the alternate customer assignment enters only the
    # exploration archive, not the positive imitation set.
    pool.capacity = 1
    routes = [[1, 2, 0, 3, 0], [1, 3, 0, 2, 0], [1, 0, 2, 3, 0]]
    priced = []
    for route in routes:
        cost = sum(float(instance.distance_matrix_km[left, right]) for left, right in zip([0] + route, route))
        priced.append((cost, route))
    for cost, route in sorted(priced):
        assert pool.ingest(instance.instance_id, route, cost, epoch=epoch)['verified']
    assert pool.exploration_routes[instance.instance_id]


def test_stagnant_training_archive_can_be_searched_outside_current_rollout():
    agent, _, instances, pool, cfg = setup_search()
    archived, current = instances
    _seed_stagnant_search_archive(pool, archived)
    cfg['offline']['exploration_instances'] = 1
    result = run_branch_exploration(agent, instances, [current.instance_id], pool, cfg, 15, 'cpu')
    assert result['branch_search_selected_instance_ids'] == [archived.instance_id]
    assert result['branch_search_stagnant_archive_instances'] == 1
    assert result['branch_search_new_current_instances'] == 0
    assert result['branch_search_anchor_instances'] == 1
    assert result['branch_search_prefix_steps'] > 0
    # The opt-out retains the old current-only selection contract.
    cfg['offline']['exploration_prefer_stagnant_archive'] = False
    result = run_branch_exploration(agent, instances, [current.instance_id], pool, cfg, 20, 'cpu')
    assert result['branch_search_selected_instance_ids'] == [current.instance_id]
    assert result['branch_search_stagnant_archive_instances'] == 0
    assert result['branch_search_new_current_instances'] == 1


def test_archive_priority_fills_remaining_budget_with_current_and_never_admits_foreign_ids():
    from dataclasses import replace
    agent, _, instances, pool, cfg = setup_search()
    archived, current = instances
    _seed_stagnant_search_archive(pool, archived)
    # A stale foreign key must never broaden the registered training scope.
    pool.exploration_routes['test_foreign'] = list(pool.exploration_routes[archived.instance_id])
    pool.best_improvement_epoch['test_foreign'] = 0
    foreign = replace(current, instance_id='test_foreign')
    result = run_branch_exploration(agent, instances + [foreign], [current.instance_id], pool, cfg, 15, 'cpu')
    assert result['branch_search_selected_instance_ids'] == [archived.instance_id, current.instance_id]
    assert result['branch_search_stagnant_archive_instances'] == 1
    assert result['branch_search_new_current_instances'] == 1
    assert result['branch_search_eligible_stagnant_archive_instances'] == 1
    with pytest.raises(ValueError, match='training-pool'):
        run_branch_exploration(agent, instances + [foreign], ['test_foreign'], pool, cfg, 20, 'cpu')


def test_stagnant_archive_budget_rotates_across_registered_instances():
    agent, _, instances, pool, cfg = setup_search()
    for instance in instances:
        _seed_stagnant_search_archive(pool, instance)
    cfg['offline']['exploration_instances'] = 1
    first = run_branch_exploration(agent, instances, [], pool, cfg, 15, 'cpu')
    second = run_branch_exploration(agent, instances, [], pool, cfg, 20, 'cpu')
    assert set(first['branch_search_selected_instance_ids'] + second['branch_search_selected_instance_ids']) == {instance.instance_id for instance in instances}
    assert first['branch_search_stagnant_archive_instances'] == second['branch_search_stagnant_archive_instances'] == 1
