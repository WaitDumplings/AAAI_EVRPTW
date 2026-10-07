"""Physical-input transformations must preserve routing dynamics and units."""
from __future__ import annotations

import copy
import numpy as np
import pytest

pytest.importorskip("gymnasium")
from caliroute.input_normalization import (
    build_input_context, input_normalization_signature,
    NODE_INPUT_CONTEXT_FEATURES, GRAPH_INPUT_CONTEXT_FEATURES,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env import EVRPTWVectorEnv
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.env_fast import EVRPTWVectorEnvFast
from evrptw_core.schema import EVRPTWInstance


def _instance():
    coords = np.array([[0, 0], [-2, 1], [3, 5], [6, -4], [14, 9], [2, -1]], dtype=float)
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    distance *= np.where(np.arange(6)[:, None] < np.arange(6)[None, :], 1.2, 1.0)
    coords += [1024, 2048]
    return EVRPTWInstance.from_dict({
        "instance_id": "input_norm", "working_start_s": 1000, "working_end_s": 3000,
        "depot": coords[0], "customers": coords[1:5], "charging_stations": coords[5:],
        "distance_matrix_km": distance, "demands_cm3": np.ones(4), "package_counts": np.ones(4),
        "service_time_s": np.full(4, 2.), "tw_s": [[1000, 3000], [1030, 3000], [1000, 3000], [1000, 3000]],
        "cs_time_to_depot_s": [3.],
        "vehicle": {"cargo_capacity_cm3": 2., "battery_capacity_kwh": 200.,
                    "consumption_kwh_per_km": .4, "full_charge_time_s": 10.},
        "speed_profile": {"effective_speed_kmh": 3600.},
    })


def _env(fast=False, instance=None, **kwargs):
    cls = EVRPTWVectorEnvFast if fast else EVRPTWVectorEnv
    extra = {"use_jit_mask": False} if fast else {}
    return cls(instance=instance or _instance(), n_traj=2, reward_distance_scale_km=7., **extra, **kwargs)


def _episode(env):
    obs, _ = env.reset(seed=19)
    observations, rewards = [obs], []
    states = [(env.current_time_s.copy(), env.load_cm3.copy(), env.battery_used_kwh.copy())]
    for destination in (1, 5, 2, 0, 3, 4, 0):
        assert obs["action_mask"][:, destination].all()
        obs, reward, terminated, truncated, info = env.step([destination] * 2)
        assert not truncated.any()
        observations.append(obs)
        rewards.append(reward)
        states.append((env.current_time_s.copy(), env.load_cm3.copy(), env.battery_used_kwh.copy()))
    assert terminated.all() and info["success"].all()
    assert (info["vehicle_count"] == 2).all()
    return observations, np.asarray(rewards), info, states


@pytest.mark.parametrize("fast", [False, True])
def test_depot_fixed_is_isotropic_unclipped_and_matches_declared_space(fast):
    env = _env(fast, observation_coordinate_mode="depot_fixed", observation_distance_scale_km=5.)
    obs, _ = env.reset()
    xy = np.concatenate([obs["depot_loc"], obs["cus_loc"], obs["rs_loc"]])
    np.testing.assert_allclose(xy, (env.coords_raw - env.coords_raw[0]) / 5., atol=1e-7)
    np.testing.assert_array_equal(xy[0], [0., 0.])
    assert xy.min() < 0 and xy.max() > 1
    np.testing.assert_allclose(np.linalg.norm(xy[1] - xy[4]),
                               np.linalg.norm(env.coords_raw[1] - env.coords_raw[4]) / 5., rtol=1e-6)
    for name in ("cus_loc", "depot_loc", "rs_loc"):
        assert env.observation_space[name].contains(obs[name])


@pytest.mark.parametrize("fast", [False, True])
def test_default_legacy_coordinates_unchanged(fast):
    env = _env(fast)
    obs, _ = env.reset()
    xy = env.coords_raw
    expected = ((xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-6)).astype(np.float32)
    np.testing.assert_array_equal(obs["cus_loc"], expected[1:5])
    assert "node_input_context" not in obs and "graph_input_context" not in obs


@pytest.mark.parametrize("fast", [False, True])
def test_translation_preserves_representation_and_added_far_customer_does_not_recenter(fast):
    inst = _instance()
    env = _env(fast, instance=inst, observation_coordinate_mode="depot_fixed", observation_distance_scale_km=5.)
    left, _ = env.reset()
    shifted = copy.deepcopy(inst)
    shifted.depot[...] += [512, -1024]
    shifted.customers[...] += [512, -1024]
    shifted.charging_stations[...] += [512, -1024]
    right, _ = env.reset(options={"instance": shifted})
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    # Existing nodes must not change when a different customer extends bounds.
    changed = copy.deepcopy(inst)
    changed.customers[-1] += [1000, 2000]
    last, _ = env.reset(options={"instance": changed})
    np.testing.assert_array_equal(left["cus_loc"][:-1], last["cus_loc"][:-1])
    np.testing.assert_array_equal(left["depot_loc"], last["depot_loc"])


@pytest.mark.parametrize("fast", [False, True])
def test_fixed_coords_preserve_scale_that_legacy_minmax_erases(fast):
    inst = _instance()
    larger = copy.deepcopy(inst)
    larger.customers[...] = larger.depot + (larger.customers - larger.depot) * 2
    larger.charging_stations[...] = larger.depot + (larger.charging_stations - larger.depot) * 2
    larger.distance_matrix_km[...] *= 2
    for mode in ("legacy_minmax", "depot_fixed"):
        left, _ = _env(fast, instance=inst, observation_coordinate_mode=mode, observation_distance_scale_km=5.).reset()
        right, _ = _env(fast, instance=larger, observation_coordinate_mode=mode, observation_distance_scale_km=5.).reset()
        np.testing.assert_allclose(right["cus_loc"], left["cus_loc"] * (2 if mode == "depot_fixed" else 1))
        np.testing.assert_allclose(right["edge_distance"], left["edge_distance"] * 2)
        np.testing.assert_allclose(right["edge_time"], left["edge_time"] * 2)
        np.testing.assert_allclose(right["edge_energy"], left["edge_energy"] * 2)


@pytest.mark.parametrize("fast", [False, True])
def test_input_changes_preserve_complete_route_rewards_and_physical_constraints(fast):
    original = _episode(_env(fast, observation_distance_scale_km=5.))
    changed = _episode(_env(fast, observation_distance_scale_km=5.,
                            observation_coordinate_mode="depot_fixed", observation_input_context=True))
    np.testing.assert_array_equal(original[1], changed[1])
    np.testing.assert_array_equal(original[3], changed[3])
    assert original[2]["routes"] == changed[2]["routes"]
    np.testing.assert_array_equal(original[2]["objective_distance_km"], changed[2]["objective_distance_km"])
    for before, after in zip(original[0], changed[0]):
        for key in before:
            if key not in {"cus_loc", "depot_loc", "rs_loc"}:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    np.testing.assert_allclose(changed[1].sum(0), -changed[2]["objective_distance_km"] / 7., rtol=1e-6)


def test_fast_and_reference_context_and_rollout_identical():
    options = dict(observation_distance_scale_km=5., observation_coordinate_mode="depot_fixed", observation_input_context=True)
    slow, fast = _episode(_env(False, **options)), _episode(_env(True, **options))
    for before, after in zip(slow[0], fast[0]):
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    np.testing.assert_array_equal(slow[1], fast[1])
    np.testing.assert_array_equal(slow[3], fast[3])
    assert slow[2]["routes"] == fast[2]["routes"]


@pytest.mark.parametrize("fast", [False, True])
def test_units_keep_speed_energy_and_resource_relationships(fast):
    env = _env(fast, observation_coordinate_mode="depot_fixed", observation_input_context=True, observation_distance_scale_km=5.)
    obs, _ = env.reset()
    speed = env.speed_km_per_s * env.horizon_s / 5.
    energy_per_distance = env.energy_per_km * 5. / env.battery_capacity_kwh
    np.testing.assert_allclose(obs["edge_time"], obs["edge_distance"] / speed, rtol=1e-6)
    np.testing.assert_allclose(obs["edge_energy"], obs["edge_distance"] * energy_per_distance, rtol=1e-6)
    np.testing.assert_allclose(obs["time_window"] * env.horizon_s + env.working_start_s, env.tw_s)
    np.testing.assert_allclose(obs["demand"] * env.cargo_capacity_cm3, env.demand_cm3)
    graph = obs["graph_input_context"]
    np.testing.assert_allclose(np.expm1(graph[3]), speed, rtol=1e-6)
    np.testing.assert_allclose(np.expm1(graph[4]), energy_per_distance, rtol=1e-6)


def _context_args():
    return dict(distance_km=np.array([[0., 1., 4.], [2., 0., 3.], [5., 6., 0.]]),
                travel_time_s=np.array([[0., 10., 40.], [20., 0., 30.], [50., 60., 0.]]),
                energy_kwh=np.array([[0., .2, .8], [.4, 0., .6], [1., 1.2, 0.]]),
                distance_scale_km=2., horizon_s=100., cargo_capacity_cm3=1e6,
                battery_capacity_kwh=2., speed_kmh=360., energy_per_km=.2,
                full_charge_time_s=10., charging_mode="fixed_full")


def test_context_preserves_directed_fixed_unit_information_and_permutation():
    args = _context_args()
    output = build_input_context(**args)
    nodes = output["node_input_context"]
    assert nodes.shape == (3, len(NODE_INPUT_CONTEXT_FEATURES))
    assert output["graph_input_context"].shape == (len(GRAPH_INPUT_CONTEXT_FEATURES),)
    np.testing.assert_allclose(np.expm1(nodes[1, :8]), [1/2, 2/2, 2.5/2, 3.5/2, 25/100, 35/100, .5/2, .7/2], rtol=1e-6)
    order = [0, 2, 1]
    for name in ("distance_km", "travel_time_s", "energy_kwh"):
        args[name] = args[name][np.ix_(order, order)]
    permuted = build_input_context(**args)
    np.testing.assert_array_equal(permuted["node_input_context"], nodes[order])
    np.testing.assert_array_equal(permuted["graph_input_context"], output["graph_input_context"])


def test_zero_cost_edges_are_reachable_and_unreachable_edges_are_not_in_means():
    args = _context_args()
    for key in ("distance_km", "travel_time_s", "energy_kwh"):
        args[key][0, 1] = 0.
        args[key][1, 2] = np.inf
    nodes = build_input_context(**args)["node_input_context"]
    assert np.isfinite(nodes).all()
    assert nodes[1, 8] == 1.  # zero-cost depot-to-node is legal
    assert nodes[1, 10] == .5
    np.testing.assert_allclose(np.expm1(nodes[1, 2]), 2/2)


def test_inactive_resource_dummies_do_not_change_context():
    args = _context_args()
    args.update(metadata={"charging_constraint": False, "time_window_constraint": False, "capacity_constraint": False})
    initial = build_input_context(**args)
    args.update(horizon_s=1e12, battery_capacity_kwh=1e12, cargo_capacity_cm3=np.inf,
                speed_kmh=1e12, energy_per_km=0., full_charge_time_s=1e12)
    args["travel_time_s"] = args["travel_time_s"] * 1e10
    args["energy_kwh"] = np.zeros_like(args["energy_kwh"])
    changed = build_input_context(**args)
    for key in initial:
        np.testing.assert_array_equal(initial[key], changed[key])
    np.testing.assert_array_equal(changed["graph_input_context"], np.zeros(10))
    assert changed["node_input_context"][:, 10:].min() == 1.


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("options", [dict(observation_coordinate_mode="depot_fixed"), dict(observation_input_context=True)])
def test_physical_encoding_requires_explicit_length_unit(fast, options):
    with pytest.raises(ValueError, match="explicit observation_distance_scale_km"):
        _env(fast, **options)


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf])
def test_bad_length_units_rejected(scale):
    with pytest.raises(ValueError, match="observation_distance_scale_km"):
        _env(observation_coordinate_mode="depot_fixed", observation_distance_scale_km=scale)


def test_bad_mode_rejected_and_signature_has_fixed_contract():
    with pytest.raises(ValueError, match="observation_coordinate_mode"):
        _env(observation_coordinate_mode="minmax_typo")
    cfg = {"env": {"observation_coordinate_mode": "depot_fixed", "observation_input_context": True, "observation_distance_scale_km": 5.}}
    signature = input_normalization_signature(cfg)
    assert signature["context_schema"] == "physical_input_context_v1"
    assert signature["context_base_units"] == {"time_s": 3600., "cargo_cm3": 1e6, "battery_kwh": 100.}
    assert len(signature["node_context_features"]) == 12 and len(signature["graph_context_features"]) == 10
    assert signature == input_normalization_signature({**cfg, "data": {"num_customers": 1000}})


def test_input_audit_surfaces_unmodelled_saved_travel_times():
    from scripts.audit_input_normalization import check_instance
    payload = copy.deepcopy(_instance().raw)
    payload["travel_time_matrix_s"] = payload["distance_matrix_km"].copy()
    correct, _ = check_instance(payload, "evrptw", 5., 7.)
    assert correct["pass"]
    # Heterogeneous speed data would be discarded by the present env. The audit
    # must report it rather than passing because new/legacy observations match.
    payload["travel_time_matrix_s"][1, 2] += 10.
    changed, _ = check_instance(payload, "evrptw", 5., 7.)
    assert not changed["pass"]
    saved = changed["physical_checks"]["saved_vs_environment_time_s"]
    assert not saved["pass"] and saved["entries_outside_tolerance"] == 1
    assert saved["max_abs_error"] == pytest.approx(10.)


def test_input_audit_does_not_accept_nan_or_mismatched_unreachable_edges():
    from scripts.audit_input_normalization import compare_arrays
    assert compare_arrays([0., np.inf], [0., np.inf])["pass"]
    assert not compare_arrays([0., np.inf], [0., 0.])["pass"]
    assert not compare_arrays([np.nan], [np.nan])["pass"]
