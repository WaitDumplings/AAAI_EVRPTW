from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from caliroute.plugins.exploration import (
    route_structure, structure_distances, diverse_route_indices,
    select_rollout_candidates, rollout_structure_diagnostics,
)


def test_structure_ignores_vehicle_order_but_preserves_directed_edges_and_partition():
    left = route_structure([1, 2, 0, 3, 4, 0], 4)
    permuted = route_structure([3, 4, 0, 1, 2, 0], 4)
    reverse = route_structure([2, 1, 0, 4, 3, 0], 4)
    reassigned = route_structure([1, 3, 0, 2, 4, 0], 4)
    assert left == permuted
    assert structure_distances(left, permuted) == (0, 0)
    assert structure_distances(left, reverse)[0] > 0
    assert structure_distances(left, reverse)[1] == 0
    assert structure_distances(left, reassigned)[1] == 1
    # Charging stations influence travel edges, never customer co-membership.
    with_cs = route_structure([1, 5, 2, 0, 3, 4, 0], 4)
    assert structure_distances(left, with_cs)[1] == 0
    assert structure_distances(left, with_cs)[0] > 0


def test_selector_rejects_three_canonical_copies_and_keeps_distinct_assignment():
    routes = [[1, 2, 0, 3, 4, 0], [3, 4, 0, 1, 2, 0],
              [0, 1, 2, 0, 3, 4, 0], [1, 3, 0, 2, 4, 0]]
    selected = diverse_route_indices(routes, [100, 100.01, 100.02, 103], 4, capacity=3)
    assert selected == [0, 3]
    assert len({route_structure(routes[index], 4) for index in selected}) == 2


def _rollout(batch=64):
    routes = np.asarray([[1, 2, 0, 3, 4, 0], [1, 3, 0, 2, 4, 0], [1, 4, 0, 2, 3, 0]])
    actions = np.broadcast_to(routes.T[:, None, :], (6, batch, 3)).copy()
    return actions, np.ones_like(actions, bool), np.broadcast_to([100., 102., 104.], (batch, 3)).copy(), np.ones((batch, 3), bool)


def test_small_intake_budget_has_multiple_routes_and_fair_rotating_instances():
    actions, valid, cost, feasible = _rollout()
    cursor, covered = 0, set()
    first_indices = None
    for _ in range(6):
        selected, cursor, stats = select_rollout_candidates(actions, valid, cost, feasible,
                                                            num_customers=4, budget=32, cursor=cursor)
        assert len(selected) == 32
        counts = Counter(route.env_idx for route in selected)
        assert max(counts.values()) == 3
        assert stats['selection_multi_route_instances'] >= 10
        current = set(counts)
        if first_indices is None:
            first_indices = current
        elif len(covered) == len(first_indices):
            assert not (current & first_indices)
        covered.update(current)
    assert covered == set(range(64))


def test_structure_metrics_detect_collapsed_and_diverse_rollouts():
    actions, valid, _, feasible = _rollout(batch=2)
    diverse = rollout_structure_diagnostics(actions, valid, feasible, num_customers=4, max_pairs=2)
    assert diverse['rollout_unique_structure_fraction'] == 1
    assert diverse['rollout_partition_distance'] == 1
    assert diverse['rollout_structure_pairs'] == 4
    actions[:, :, :] = actions[:, :, :1]
    collapsed = rollout_structure_diagnostics(actions, valid, feasible, num_customers=4)
    assert collapsed['rollout_unique_structure_fraction'] == pytest.approx(1/3)
    assert collapsed['rollout_directed_edge_distance'] == 0
    assert collapsed['rollout_partition_distance'] == 0


def _bounded_pool(monkeypatch, **kwargs):
    pytest.importorskip('gymnasium')
    from offline2online.policy_route_replay import PolicyRoutePool
    instance = SimpleNamespace(instance_id='train_0', customers=np.zeros((4, 2)))
    pool = PolicyRoutePool([instance], {}, structure_enabled=True, exploration_capacity=2, **kwargs)
    monkeypatch.setattr(pool, 'fingerprint', lambda _: 'verified_test_instance')
    # Test admission mechanics separately; real masked-env checks remain in test_policy_route_replay.py.
    monkeypatch.setattr(pool, 'replay', lambda route: [{}] if len(route.actions) == 6 and route.actions[-1] == 0 else None)
    return pool


def test_exploration_pool_is_not_positive_imitation_and_has_quality_cost_bound(monkeypatch):
    pool = _bounded_pool(monkeypatch, capacity=1, max_relative_gap=.05, exploration_max_relative_gap=.25)
    elite = [1, 2, 0, 3, 4, 0]
    other = [1, 3, 0, 2, 4, 0]
    bad = [1, 4, 0, 2, 3, 0]
    assert pool.ingest('train_0', elite, 100, epoch=1)['elite_added']
    outcome = pool.ingest('train_0', other, 115, epoch=2)
    assert outcome == {'verified': True, 'elite_added': False, 'exploration_added': True}
    assert [route.actions for route in pool.routes['train_0']] == [tuple(elite)]
    assert pool.exploration_routes['train_0'][0].actions == tuple(other)
    outcome = pool.ingest('train_0', bad, 130, epoch=3)
    assert outcome['verified'] and not outcome['exploration_added']
    assert pool.choose_exploration_anchor('train_0', 5) is None
    anchor = pool.choose_exploration_anchor('train_0', 15, prefix_fraction=.5, max_prefix_steps=2)
    assert anchor['actions'] == tuple(other[:2])
    assert anchor['full_actions'] == tuple(other)
    assert anchor['partition_distance'] == 1
    assert anchor['stagnation_epochs'] == 14
    assert pool.exploration_diagnostics(15)['exploration_pool_routes'] == 1
    assert not pool.ingest('train_0', [1, 2], 80, epoch=16)['verified']
    assert pool.best_objectives['train_0'] == 100


def test_new_best_prunes_exploration_and_checkpoint_preserves_roles(monkeypatch):
    pool = _bounded_pool(monkeypatch, capacity=1)
    first, other, improved = [1, 2, 0, 3, 4, 0], [1, 3, 0, 2, 4, 0], [1, 4, 0, 2, 3, 0]
    pool.ingest('train_0', first, 100, epoch=1)
    pool.ingest('train_0', other, 115, epoch=2)
    state = copy.deepcopy(pool.state_dict())
    restored = _bounded_pool(monkeypatch, capacity=1)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    restored.ingest('train_0', improved, 80, epoch=20)
    assert restored.best_improvement_epoch['train_0'] == 20
    assert all(route.objective <= 100 for route in restored.exploration_routes['train_0'])
    assert all(route.objective == 80 for route in restored.routes['train_0'])
    changed = _bounded_pool(monkeypatch, capacity=2)
    with pytest.raises(ValueError, match='configuration changed'):
        changed.load_state_dict(state)


def test_opt_in_archive_still_requires_real_masked_environment_verification():
    pytest.importorskip('gymnasium')
    from test_rollout_static_cache import _env
    from offline2online.policy_route_replay import PolicyRoutePool
    instance = _env().unwrapped.instance
    pool = PolicyRoutePool([instance], {'use_jit_mask': False, 'info_level': 'light'},
                           structure_enabled=True, exploration_capacity=2)
    actions = [1, 2, 0, 3, 0]
    cost = sum(float(instance.distance_matrix_km[left, right]) for left, right in zip([0] + actions, actions))
    assert pool.ingest(instance.instance_id, actions, cost, epoch=1)['verified']
    assert not pool.ingest(instance.instance_id, [1, 1, 2, 0, 3, 0], cost, epoch=2)['verified']
    assert not pool.ingest(instance.instance_id, [1, 2], cost, epoch=2)['verified']
    assert not pool.ingest(instance.instance_id, actions, cost + 1, epoch=2)['verified']
    assert len(pool) == 1


def test_exploration_checkpoint_rejects_search_to_imitation_role_corruption(monkeypatch):
    pool = _bounded_pool(monkeypatch, capacity=1)
    pool.ingest('train_0', [1, 2, 0, 3, 4, 0], 100, epoch=1)
    pool.ingest('train_0', [1, 3, 0, 2, 4, 0], 115, epoch=2)
    state = copy.deepcopy(pool.state_dict())
    state['exploration_routes'] = copy.deepcopy(state['routes'])
    with pytest.raises(ValueError, match='role isolation'):
        _bounded_pool(monkeypatch, capacity=1).load_state_dict(state)


@pytest.mark.parametrize('field', ['travel_time_matrix_s', 'energy_matrix_kwh'])
def test_explicit_physical_edges_bind_archive_resume_even_when_routes_remain_feasible(field):
    from dataclasses import replace
    from test_rollout_static_cache import _env
    from offline2online.policy_route_replay import PolicyRoute, PolicyRoutePool
    from evrptw_core.physical import resolve_physical_edge_matrices
    instance = _env().unwrapped.instance
    physical = resolve_physical_edge_matrices(instance)
    explicit = replace(instance, travel_time_matrix_s=physical['travel_time_s'].copy(),
                        energy_matrix_kwh=physical['energy_kwh'].copy())
    changed = replace(explicit, **{field: np.asarray(getattr(explicit, field)) * 1.01})
    options = {'use_jit_mask': False, 'info_level': 'light', 'prefer_explicit_edge_matrices': True}
    source = PolicyRoutePool([explicit], options, structure_enabled=True, exploration_capacity=2)
    actions = [1, 2, 0, 3, 0]
    cost = sum(float(instance.distance_matrix_km[left, right]) for left, right in zip([0] + actions, actions))
    assert source.ingest(instance.instance_id, actions, cost, epoch=1)['verified']
    restored = PolicyRoutePool([changed], options, structure_enabled=True, exploration_capacity=2)
    # The unchanged routes still satisfy all masks and preserve the same distance;
    # the semantic fingerprint, rather than feasibility alone, must reject resume.
    current_route = PolicyRoute(instance.instance_id, restored.fingerprint(instance.instance_id), tuple(actions), cost)
    assert restored.replay(current_route) is not None
    assert source.fingerprint(instance.instance_id) != restored.fingerprint(instance.instance_id)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        restored.load_state_dict(source.state_dict())
    legacy_options = {**options, 'prefer_explicit_edge_matrices': False}
    old = PolicyRoutePool([explicit], legacy_options)
    old_changed = PolicyRoutePool([changed], legacy_options)
    assert old.fingerprint(instance.instance_id) == old_changed.fingerprint(instance.instance_id)
    assert old.fingerprint(instance.instance_id) != source.fingerprint(instance.instance_id)
