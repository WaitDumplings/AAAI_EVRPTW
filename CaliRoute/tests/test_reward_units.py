"""Physical reward units are independent of model observation units and size."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import _configure_dataset_reward_scale
from evrptw_core.schema import EVRPTWInstance


def _instance(n_customers=4):
    coordinates = np.column_stack((np.arange(n_customers + 1), np.zeros(n_customers + 1)))
    coordinates = np.vstack((coordinates, [[0.0, 1.0]]))
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    return EVRPTWInstance.from_dict({
        "instance_id": f"reward_units_{n_customers}",
        "working_start_s": 0.0, "working_end_s": 1_000_000.0,
        "depot": coordinates[0], "customers": coordinates[1:-1],
        "charging_stations": coordinates[-1:], "distance_matrix_km": distances,
        "demands_cm3": np.ones(n_customers), "package_counts": np.ones(n_customers),
        "service_time_s": np.ones(n_customers), "tw_s": [[0.0, 1_000_000.0]] * n_customers,
        "cs_time_to_depot_s": [1.0],
        "vehicle": {"cargo_capacity_cm3": 2.0, "battery_capacity_kwh": 1_000_000.0,
                    "consumption_kwh_per_km": 0.4, "full_charge_time_s": 10.0},
        "speed_profile": {"effective_speed_kmh": 3600.0},
    })


def _env(fast, n_customers=4, **kwargs):
    return make_terran_env(instance=_instance(n_customers), n_traj=2,
                          use_fast_env=fast, use_jit_mask=False, **kwargs)


def _episode(env):
    observation, _ = env.reset(seed=19)
    observations, rewards = [observation], []
    # Two vehicles, a charging visit, and both final depot legs must contribute.
    for destination in (1, 5, 2, 0, 3, 4, 0):
        assert observation["action_mask"][:, destination].all()
        observation, reward, terminated, truncated, info = env.step([destination] * 2)
        observations.append(observation)
        rewards.append(reward)
        assert not truncated.any()
    assert terminated.all() and info["success"].all()
    return observations, np.stack(rewards), info


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("unit", [1.0, 7.0, 43.638668060302734])
def test_completed_rollout_rewards_equal_physical_distance(fast, unit):
    env = _env(fast, reward_distance_scale_km=unit, observation_distance_scale_km=3.0)
    _, rewards, info = _episode(env)
    np.testing.assert_allclose(rewards.sum(axis=0), -info["objective_distance_km"] / unit,
                               rtol=2e-7, atol=1e-7)
    assert (info["vehicle_count"] == 2).all()


@pytest.mark.parametrize("fast", [False, True])
def test_same_edge_has_same_reward_across_customer_counts(fast):
    rewards = []
    for n in (15, 50, 100, 1000):
        env = _env(fast, n_customers=n, reward_distance_scale_km=7.0,
                   observation_distance_scale_km=3.0)
        observation, _ = env.reset(seed=19)
        assert observation["action_mask"][:, 1].all()
        _, reward, _, truncated, _ = env.step([1, 1])
        assert not truncated.any()
        rewards.append(reward)
    np.testing.assert_array_equal(rewards[0], rewards[1])
    np.testing.assert_array_equal(rewards[0], rewards[2])
    np.testing.assert_allclose(rewards[0], -1.0 / 7.0)


@pytest.mark.parametrize("fast", [False, True])
def test_reward_unit_change_preserves_every_observation_and_route(fast):
    left = _episode(_env(fast, reward_distance_scale_km=2.0, observation_distance_scale_km=5.0))
    right = _episode(_env(fast, reward_distance_scale_km=8.0, observation_distance_scale_km=5.0))
    for obs_left, obs_right in zip(left[0], right[0]):
        assert obs_left.keys() == obs_right.keys()
        for key in obs_left:
            np.testing.assert_array_equal(obs_left[key], obs_right[key], err_msg=key)
    np.testing.assert_array_equal(left[1], right[1] * 4.0)
    np.testing.assert_array_equal(left[2]["objective_distance_km"], right[2]["objective_distance_km"])
    assert left[2]["routes"] == right[2]["routes"]


@pytest.mark.parametrize("fast", [False, True])
def test_observation_unit_change_does_not_change_reward_or_dynamics(fast):
    left = _episode(_env(fast, reward_distance_scale_km=7.0, observation_distance_scale_km=2.0))
    right = _episode(_env(fast, reward_distance_scale_km=7.0, observation_distance_scale_km=8.0))
    np.testing.assert_array_equal(left[1], right[1])
    for obs_left, obs_right in zip(left[0], right[0]):
        for key in obs_left:
            if key == "edge_distance":
                np.testing.assert_array_equal(obs_left[key], obs_right[key] * 4.0)
            else:
                np.testing.assert_array_equal(obs_left[key], obs_right[key], err_msg=key)


@pytest.mark.parametrize("fast", [False, True])
def test_legacy_defaults_keep_reward_scale_for_observations_and_reset(fast):
    env = _env(fast)
    for n in (4, 6):
        observation, _ = env.reset(seed=19, options={"instance": _instance(n)})
        expected_scale = float(n + 1)  # median depot round trips: 2, 4, ..., 2N.
        assert env.reward_distance_scale_km == expected_scale
        assert env.observation_distance_scale_km == expected_scale
        np.testing.assert_array_equal(observation["edge_distance"],
                                      (env.distance_km / expected_scale).astype(np.float32))
        _, reward, _, _, _ = env.step([1, 1])
        np.testing.assert_allclose(reward, -1.0 / expected_scale)


@pytest.mark.parametrize("fast", [False, True])
def test_explicit_observation_unit_stays_fixed_when_instance_changes(fast):
    env = _env(fast, observation_distance_scale_km=3.0)
    for n in (4, 6):
        obs, _ = env.reset(options={"instance": _instance(n)})
        assert env.reward_distance_scale_km == float(n + 1)
        assert env.observation_distance_scale_km == 3.0
        np.testing.assert_array_equal(obs["edge_distance"], (env.distance_km / 3.0).astype(np.float32))


def test_explicit_reward_unit_takes_precedence_over_dataset_fitting():
    cfg = {"env": {"reward_distance_scale_mode": "dataset_single_customer_repair_median",
                   "reward_distance_scale_km": 7.0, "observation_distance_scale_km": 3.0}}
    pool = SimpleNamespace(reward_distance_scale_km=Mock(side_effect=AssertionError("must not fit")))
    _configure_dataset_reward_scale(cfg, pool)
    pool.reward_distance_scale_km.assert_not_called()
    assert cfg["env"] == {"reward_distance_scale_mode": "single_customer_repair_median",
                          "reward_distance_scale_km": 7.0, "observation_distance_scale_km": 3.0}
    assert cfg["normalization"]["reward_distance_scale_source"] == "explicit"
    # Resolving the same configuration a second time must not refit or change it.
    before = {section: dict(values) for section, values in cfg.items()}
    _configure_dataset_reward_scale(cfg, object())
    assert cfg == before


def test_legacy_dataset_scale_is_still_fitted_once():
    cfg = {"env": {"reward_distance_scale_mode": "dataset_single_customer_repair_median"}}
    pool = SimpleNamespace(reward_distance_scale_km=Mock(return_value=11.0), region_pool_status="train")
    _configure_dataset_reward_scale(cfg, pool)
    _configure_dataset_reward_scale(cfg, pool)
    pool.reward_distance_scale_km.assert_called_once_with("single_customer_repair_median")
    assert cfg["env"]["reward_distance_scale_km"] == 11.0
    assert cfg["normalization"]["reward_distance_scale_source"] == "train"


@pytest.mark.parametrize("name", ["reward_distance_scale_km", "observation_distance_scale_km"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_explicit_units_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        _env(True, **{name: value})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_dataset_override_is_rejected_before_pool_access(value):
    cfg = {"env": {"reward_distance_scale_mode": "dataset_single_customer_repair_median",
                   "reward_distance_scale_km": value}}
    with pytest.raises(ValueError, match="reward_distance_scale_km"):
        _configure_dataset_reward_scale(cfg, object())
