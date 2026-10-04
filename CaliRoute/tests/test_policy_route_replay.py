from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
from test_rollout_static_cache import _agent, _env, _run
from offline2online.policy_route_replay import PolicyRoute, PolicyRoutePool, require_train_dataset
from offline2online.trainer import _init_policy_route_pool, _prepare_policy_replay_candidates


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _pool(capacity=3, **kwargs):
    instances = [_env().unwrapped.instance, _env(1.3).unwrapped.instance]
    return PolicyRoutePool(instances, {"use_jit_mask": False, "info_level": "light"}, capacity=capacity, **kwargs)


def _cost(pool, actions, instance_id="synthetic_1.0"):
    matrix = pool.instances[instance_id].distance_matrix_km
    return sum(float(matrix[a, b]) for a, b in zip([0] + actions, actions))


def test_pool_rejects_invalid_incomplete_wrong_cost_and_foreign_routes():
    pool = _pool()
    valid = [1, 2, 0, 3, 0]
    assert not pool.add("test_only", valid, 7)
    assert not pool.add("synthetic_1.0", [1, 2], 3)
    assert not pool.add("synthetic_1.0", [1, 1, 2, 0, 3, 0], 7)
    assert not pool.add("synthetic_1.0", valid, 999)
    assert pool.add("synthetic_1.0", valid, _cost(pool, valid))
    assert not pool.add("synthetic_1.0", valid, _cost(pool, valid))
    assert len(pool) == 1


def test_diversity_quality_capacity_and_compact_checkpoint_roundtrip():
    pool = _pool(capacity=2, max_relative_gap=0.2)
    sequences = [[1, 2, 0, 3, 0], [2, 1, 0, 3, 0], [3, 0, 1, 2, 0], [1, 3, 0, 2, 0]]
    for actions in sequences:
        pool.add("synthetic_1.0", actions, _cost(pool, actions))
    assert 1 <= len(pool) <= 2
    state = pool.state_dict()
    assert all(set(route) == {"instance_id", "fingerprint", "actions", "objective"} for route in state["routes"])
    restored = _pool(capacity=2, max_relative_gap=0.2)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    corrupted = copy.deepcopy(state)
    corrupted["routes"][0]["fingerprint"] = "wrong_dataset"
    with pytest.raises(ValueError, match="fingerprint"):
        _pool().load_state_dict(corrupted)


def test_replay_requires_train_split_and_disabled_is_noop(tmp_path):
    directory = tmp_path / "train" / "Cus50"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"split": "test"}))
    with pytest.raises(ValueError, match="train split"):
        require_train_dataset(directory)
    assert _init_policy_route_pool({"offline": {"policy_replay_enabled": False}}, None) is None
    (directory / "metadata.json").write_text(json.dumps({"split": "train"}))
    require_train_dataset(directory)


def test_candidates_use_history_budget_and_refresh_old_policy_logprob():
    agent = _agent()
    batch = _run(agent)
    envs = [_env(), _env(1.3)]
    pool = _pool(max_relative_gap=0.2)
    actions = [1, 3, 0, 2, 0]
    for instance_id in pool.instances:
        assert pool.add(instance_id, actions, _cost(pool, actions, instance_id))
    # Set current rollout quality below the verified stored route to open its quality gate.
    for info in batch.final_infos:
        info["objective_distance_km"] = np.asarray(info["objective_distance_km"]) + 100
    cfg = {"training": {"cache_expert_route_encoding": True}, "offline": {
        "policy_replay_enabled": True, "policy_replay_fraction": 1.0,
        "policy_replay_max_candidates": 1, "policy_replay_max_new_routes": 0,
        "policy_replay_weight": 0.2,
    }, "advantage": {}}
    first, stats = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 1)
    assert len(first) == 1
    assert stats["policy_replay_verified"] == 1
    assert stats["policy_replay_added"] == 0
    first_id = first[0].env_idx
    old_logprob = first[0].old_mean_logprob
    with torch.no_grad():
        for parameter in agent.backbone.decoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.002)
    pool.cursor = first_id
    second, _ = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 1)
    assert len(second) == 1
    assert np.isfinite(second[0].old_mean_logprob)
    assert abs(old_logprob - second[0].old_mean_logprob) > 1e-7


def test_replay_warmup_collects_without_loss_and_rejects_weak_history():
    agent = _agent()
    batch = _run(agent)
    envs = [_env(), _env(1.3)]
    pool = _pool(max_relative_gap=0.2)
    actions = [1, 3, 0, 2, 0]
    for instance_id in pool.instances:
        assert pool.add(instance_id, actions, _cost(pool, actions, instance_id))
    cfg = {"training": {"cache_expert_route_encoding": True}, "offline": {
        "policy_replay_enabled": True, "policy_replay_fraction": 1.0,
        "policy_replay_max_candidates": 2, "policy_replay_max_new_routes": 0,
        "policy_replay_weight": 0.1, "policy_replay_warmup_epochs": 25,
        "policy_replay_ramp_epochs": 75, "policy_replay_require_current_improvement": True,
        "policy_replay_min_current_improvement": 0.002,
    }, "advantage": {}}
    for info in batch.final_infos:
        info["objective_distance_km"] = np.asarray(info["objective_distance_km"]) + 100
    first, stats = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 25)
    assert first == [] and stats['policy_replay_weight'] == 0
    second, stats = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 26)
    assert len(second) == 2
    assert stats['policy_replay_weight'] == pytest.approx(.1 / 75)
    assert stats['policy_replay_current_best_gap_mean'] > .002
    # History is better than the sample mean, but equal to the current best:
    # legacy gate allows it; v2 refuses to reinforce an already matched route.
    for index, info in enumerate(batch.final_infos):
        instance_id = envs[index].unwrapped.instance.instance_id
        cost = _cost(pool, actions, instance_id)
        info['objective_distance_km'] = np.array([cost, cost + 10, cost + 20])
        info['success'] = np.ones(3, dtype=bool)
    rejected, stats = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 100)
    assert not rejected
    assert stats['policy_replay_quality_rejected'] == 2
    cfg['offline']['policy_replay_require_current_improvement'] = False
    allowed, _ = _prepare_policy_replay_candidates(agent, batch, cfg, envs, pool, "cpu", 100)
    assert len(allowed) == 2
