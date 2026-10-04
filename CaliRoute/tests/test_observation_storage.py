import numpy as np

from offline2online.observation_storage import snapshot_observation
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import stack_observations


def test_static_storage_is_owned_and_reused_but_dynamic_states_do_not_alias():
    obs = {'edge_distance': np.eye(4), 'current_time': np.array([0.]), 'action_mask': np.ones((1, 4), bool)}
    cache = {}
    first = snapshot_observation(obs, cache)
    obs['current_time'][0] = 1.
    obs['action_mask'][0, 2] = False
    second = snapshot_observation(obs, cache)
    assert first['edge_distance'] is second['edge_distance']
    assert first['edge_distance'] is not obs['edge_distance']
    assert first['current_time'][0] == 0
    assert second['current_time'][0] == 1
    assert first['action_mask'][0, 2]
    assert not second['action_mask'][0, 2]
    fresh_episode = snapshot_observation({**obs, 'edge_distance': np.eye(4) * 2}, {})
    assert fresh_episode['edge_distance'][0, 0] == 2


def test_rollout_stack_shares_only_episode_static_values():
    initial = [{'edge_distance': np.eye(3) * k, 'current_time': np.array([0.])} for k in (1, 2)]
    changed = [{**obs, 'current_time': np.array([3.])} for obs in initial]
    cache = {}
    first = stack_observations(initial, cache)
    second = stack_observations(changed, cache)
    expected = stack_observations(changed)
    assert first['edge_distance'] is second['edge_distance']
    for key in second:
        np.testing.assert_array_equal(second[key], expected[key])
    np.testing.assert_array_equal(first['current_time'], 0)
