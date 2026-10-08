"""Whole-episode objective and terminal semantics of the opt-in reward contract."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import build_pbrs_config
from evrptw_core.schema import EVRPTWInstance


def instance(n=4):
    coordinates = np.vstack((np.column_stack((np.arange(n + 1), np.zeros(n + 1))), [[0., 1.]]))
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    return EVRPTWInstance.from_dict({
        "instance_id": f"strict_{n}", "working_start_s": 0, "working_end_s": 1000000,
        "depot": coordinates[0], "customers": coordinates[1:-1], "charging_stations": coordinates[-1:],
        "distance_matrix_km": distances, "demands_cm3": np.ones(n), "package_counts": np.ones(n),
        "service_time_s": np.ones(n), "tw_s": [[0., 1000000.]] * n, "cs_time_to_depot_s": [1.],
        "vehicle": {"cargo_capacity_cm3": 2., "battery_capacity_kwh": 1000000.,
                    "consumption_kwh_per_km": .4, "full_charge_time_s": 10.},
        "speed_profile": {"effective_speed_kmh": 3600.},
    })


def env(fast, **overrides):
    kwargs = dict(instance=instance(), n_traj=2, use_fast_env=fast, use_jit_mask=False,
                  reward_contract="strict_distance", reward_distance_scale_km=7.,
                  observation_distance_scale_km=3., failure_penalty_km=70.)
    kwargs.update(overrides)
    return make_terran_env(**kwargs)


def run_two_routes(environment):
    observation, initial_info = environment.reset(seed=19)
    histories = []
    # One trajectory finishes a step before the other; finished rewards must be zero.
    left, right = (1, 2, 0, 3, 4, 0, 0), (1, 5, 2, 0, 3, 4, 0)
    for actions in zip(left, right):
        observation, reward, terminated, truncated, info = environment.step(actions)
        assert not truncated.any()
        histories.append((reward, info))
    assert terminated.all()
    return initial_info, histories


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("shaping", [False, True])
def test_complete_routes_preserve_total_distance_and_zero_terminal_potential(fast, shaping):
    pbrs = PotentialRewardConfig(gamma=1., strict_contract=True, use_customer_pbrs=True,
                                use_repair_distance_pbrs=True, use_feasible_ratio_pbrs=True,
                                feasible_ratio_coef=.2) if shaping else None
    initial, steps = run_two_routes(env(fast, pbrs_config=pbrs))
    total = np.stack([r for r, _ in steps]).sum(0)
    final = steps[-1][1]
    initial_phi = initial.get("reward_shaping_initial_potential", np.zeros(2))
    np.testing.assert_allclose(total, -final["objective_distance_km"] / 7. - initial_phi, atol=1e-6)
    assert steps[-1][0][0] == 0.
    for reward, info in steps:
        np.testing.assert_allclose(reward, info["reward_base"] + info["reward_shaping"], atol=1e-7)
        assert not info["reward_failure_cost"].any()
    if shaping:
        # The feasible potential is initially nonzero; its offset must be retained.
        np.testing.assert_allclose(initial_phi, .2)
        total_shaping = np.stack([i["reward_shaping"] for _, i in steps]).sum(0)
        np.testing.assert_allclose(total_shaping, -initial_phi, atol=1e-7)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("unit", [7., 14.])
@pytest.mark.parametrize("failure", ["invalid", "timeout"])
def test_each_true_failure_cost_is_physical_and_occurs_once(fast, unit, failure):
    environment = env(fast, instance=instance(1), reward_distance_scale_km=unit,
                      max_steps_factor=0 if failure == "timeout" else 4,
                      invalid_action_penalty=-999.)
    environment.reset()
    action = 1 if failure == "timeout" else 99
    _, reward, terminated, truncated, info = environment.step([action, action])
    assert truncated.all() and not terminated.any()
    assert not info["success"].any()
    if failure == "timeout":
        # Serving all customers without returning to depot is still failure.
        np.testing.assert_array_equal(info["served_customers"], 1)
    np.testing.assert_allclose(reward, -(info["objective_distance_km"] + 70.) / unit)
    np.testing.assert_allclose(info["reward_failure_cost"], 70. / unit)
    _, repeat, _, _, info = environment.step([99, 99])
    np.testing.assert_array_equal(repeat, 0.)
    np.testing.assert_array_equal(info["reward_failure_cost"], 0.)


@pytest.mark.parametrize("fast", [False, True])
def test_failed_terminal_closes_shaping_potential(fast):
    environment = env(fast, pbrs_config=PotentialRewardConfig(
        gamma=1., strict_contract=True, use_customer_pbrs=True,
        use_repair_distance_pbrs=True, use_feasible_ratio_pbrs=True, feasible_ratio_coef=.2))
    _, initial = environment.reset()
    _, first, _, _, _ = environment.step([1, 1])
    _, second, _, truncated, info = environment.step([99, 99])
    assert truncated.all()
    np.testing.assert_allclose(first + second, -(info["objective_distance_km"] + 70.) / 7.
                               - initial["reward_shaping_initial_potential"], atol=1e-6)
    np.testing.assert_array_equal(environment.step([0, 0])[1], 0.)


@pytest.mark.parametrize("overrides,match", [
    ({"failure_penalty_km": None}, "failure_penalty_km"),
    ({"failure_penalty_km": -1}, "failure_penalty_km"),
    ({"reward_distance_scale_km": None}, "explicit positive"),
    ({"normalize_reward": False}, "normalize_reward"),
    ({"reward_mode": "distance_success"}, "success bonuses"),
    ({"success_bonus": 1.}, "success bonuses"),
])
def test_strict_environment_rejects_ambiguous_units_and_bonuses(overrides, match):
    with pytest.raises(ValueError, match=match):
        env(True, **overrides)


@pytest.mark.parametrize("kwargs,match", [
    ({"gamma": .99}, "gamma=1"),
    ({"pbrs_clip": .1}, "clipping breaks"),
    ({"customer_pbrs_mode": "direct_progress"}, "customer_pbrs_mode"),
    ({"use_terminal_heuristic": True}, "terminal bonuses"),
])
def test_strict_potential_rejects_non_telescoping_options(kwargs, match):
    options = dict(gamma=1., strict_contract=True)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PotentialRewardConfig(**options)


def test_potential_scale_is_fixed_for_the_whole_episode():
    environment = env(True, pbrs_config=PotentialRewardConfig(gamma=1., strict_contract=True, use_customer_pbrs=True))
    environment.reset()
    environment.set_reward_scale(.5)
    with pytest.raises(ValueError, match="cannot change within an episode"):
        environment.step([1, 1])
    # An epoch-level schedule may change the next episode's frozen scale.
    environment.reset()
    environment.step([1, 1])


def test_training_builder_enforces_gamma_even_when_shaping_is_disabled():
    cfg = {"env": {"reward_contract": "strict_distance"}, "training": {"gamma": .99}}
    with pytest.raises(ValueError, match="training.gamma=1"):
        build_pbrs_config(cfg)
    cfg["training"]["gamma"] = 1.
    assert build_pbrs_config(cfg) is None
    cfg["pbrs"] = {"use_customer_pbrs": True}
    assert build_pbrs_config(cfg).strict_contract


@pytest.mark.parametrize("fast", [False, True])
def test_independent_time_and_energy_matrices_reach_observations_and_dynamics(fast):
    original = instance()
    distance = original.distance_matrix_km
    time, energy = distance * 3., distance * .7
    time[0, 1], energy[0, 1] = 11., 2.5
    instance_with_edges = replace(original, travel_time_matrix_s=time, energy_matrix_kwh=energy)
    environment = env(fast, instance=instance_with_edges, prefer_explicit_edge_matrices=True)
    observation, _ = environment.reset()
    np.testing.assert_allclose(observation["edge_time"], time / 1000000.)
    np.testing.assert_allclose(observation["edge_energy"], energy / 1000000.)
    _, reward, _, truncated, _ = environment.step([1, 1])
    assert not truncated.any()
    np.testing.assert_allclose(environment.unwrapped.current_time_s, 12.)
    np.testing.assert_allclose(environment.unwrapped.battery_used_kwh, 2.5)
    np.testing.assert_allclose(reward, -distance[0, 1] / 7.)
    legacy_edges = env(fast, instance=instance_with_edges)
    np.testing.assert_allclose(legacy_edges.unwrapped.travel_time_s, distance)
    np.testing.assert_allclose(legacy_edges.unwrapped.energy_kwh, distance * .4)


@pytest.mark.parametrize("bad,match", [("shape", "shape"), ("nan", "nonnegative"), ("negative", "nonnegative"), ("unreachable", "reachability")])
def test_independent_matrices_are_validated(bad, match):
    original = instance()
    time = original.distance_matrix_km.copy()
    if bad == "shape":
        time = time[:-1]
    else:
        time[0, 1] = {"nan": float("nan"), "negative": -1., "unreachable": float("inf")}[bad]
    with pytest.raises(ValueError, match=match):
        env(True, instance=replace(original, travel_time_matrix_s=time), prefer_explicit_edge_matrices=True)
