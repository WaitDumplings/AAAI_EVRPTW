from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from offline2online import trainer
from offline2online.instance_adapter import adapt_instance_payload


def instance():
    return SimpleNamespace(
        instance_id="vrptw_example", num_customers=3, num_charging_stations=0,
        demands_cm3=np.array([1., 1., 1.]),
        distance_matrix_km=np.ones((4, 4)) - np.eye(4),
        vehicle={"cargo_capacity_cm3": 2., "design_speed_kmh": 3600.},
        speed_profile={"effective_speed_kmh": 3600.},
        working_start_s=1000., working_end_s=1030.,
        tw_s=np.array([[1010., 1011.], [1016., 1020.], [1000., 1002.]]),
        service_time_s=np.array([5., 3., 2.]),
    )


def row():
    return {"routes": [[0, 1, 2, 0], [0, 3, 0]], "objective_distance_km": 5., "feasible": True}


def test_waiting_service_start_deadline_and_each_vehicle_clock_reset():
    result = trainer._validate_vrptw_eval_route(instance(), row())
    assert result["checked"] and result["valid"]
    assert result["problem_type"] == "vrptw"
    assert result["route_waiting_times_s"] == [9., 0.]
    assert result["route_return_times_s"] == [1020., 1004.]
    # Service at customer 1 ends at 1015, after its due=1011. This is allowed:
    # the customer deadline bounds service start, not service completion.
    assert result["time_windows_valid"] and result["service_completion_valid"]


def test_late_service_start_is_rejected():
    data = instance()
    data.tw_s[1] = [1010., 1015.]
    result = trainer._validate_vrptw_eval_route(data, row())
    assert not result["valid"] and not result["time_windows_valid"]
    assert result["time_window_violations"][0]["customer"] == 2
    assert result["time_window_violations"][0]["service_start_s"] == 1016.


def test_service_completion_and_return_deadlines_are_checked_separately():
    data = instance()
    data.working_end_s = 1019.
    result = trainer._validate_vrptw_eval_route(data, row())
    assert result["service_completion_valid"]
    assert not result["valid"] and not result["depot_return_valid"]
    assert result["depot_return_violations"][0]["return_time_s"] == 1020.
    data.service_time_s[1] = 4.
    result = trainer._validate_vrptw_eval_route(data, row())
    assert not result["service_completion_valid"]


def test_environment_direct_return_check_is_preserved_for_nonmetric_distances():
    data = instance()
    data.distance_matrix_km[1, 0] = 100.
    # Actual route 0->1->2->0 still ends at 1020. The action mask nevertheless
    # rejects visiting 1 because it cannot return directly before the horizon.
    result = trainer._validate_vrptw_eval_route(data, row())
    assert result["depot_return_valid"] and result["time_windows_valid"]
    assert not result["return_reachability_valid"] and not result["valid"]


@pytest.mark.parametrize("routes,objective,failed_check", [
    ([[0, 1, 2, 3, 0]], 4., "capacity_valid"),
    ([[0, 1, 1, 0], [0, 3, 0]], 4., "customer_coverage_valid"),
    ([[0, 1, 2, 0]], 3., "customer_coverage_valid"),
    ([[0, 1, 2], [0, 3, 0]], 4., "depot_endpoints_valid"),
    ([[0, 1, 2, 0], [0, 3, 0]], 4., "distance_matches"),
    ([[0, 1, 8, 0], [0, 3, 0]], 5., "indices_valid"),
])
def test_shared_spatial_checks_reject_invalid_routes(routes, objective, failed_check):
    result = trainer._validate_vrptw_eval_route(instance(), {"routes": routes, "objective_distance_km": objective})
    assert not result["valid"] and not result[failed_check]


def test_capacity_tolerance_is_not_relaxed_relative_to_environment():
    data = instance()
    data.vehicle["cargo_capacity_cm3"] = 2. - 1e-7
    assert not trainer._validate_vrptw_eval_route(data, row())["capacity_valid"]


@pytest.mark.parametrize("field", ["tw_s", "service_time_s", "distance_matrix_km"])
def test_nonfinite_temporal_inputs_are_rejected(field):
    data = instance()
    getattr(data, field).flat[0] = np.nan
    result = trainer._validate_vrptw_eval_route(data, row())
    assert not result["valid"] and not result["temporal_inputs_valid"]


def test_time_tolerance_matches_environment_and_speed_fallback():
    data = instance()
    data.speed_profile = {}  # Environment falls back to vehicle design speed.
    data.tw_s[1] = [1010., 1016. - 5e-10]
    assert trainer._validate_vrptw_eval_route(data, row())["valid"]
    data.tw_s[1, 1] = 1016. - 2e-8
    assert not trainer._validate_vrptw_eval_route(data, row())["valid"]


def test_direct_validator_agrees_with_real_fast_environment_route():
    data = instance()
    payload = dict(vars(data), problem_class="VRPTW", depot=[0., 0.],
                   customers=[[1., 0.], [0., 1.], [1., 1.]])
    adapted = adapt_instance_payload(payload, problem_type="vrptw")
    env = trainer.make_terran_env(instance=adapted, n_traj=1, use_fast_env=True, use_jit_mask=False, info_level="full")
    observation, _ = env.reset(seed=17)
    for node in [1, 2, 0, 3, 0]:
        assert observation["action_mask"][0, node]
        observation, _, terminated, truncated, info = env.step(np.array([node]))
    assert terminated[0] and not truncated[0] and info["success"][0]
    exported = trainer._select_min_median_trajectory_stats(info)
    result = trainer._validate_vrptw_eval_route(adapted, exported)
    assert result["valid"] and result["route_return_times_s"] == [1020., 1004.]
    env.close()


@pytest.mark.parametrize("environment_success,late", [(True, False), (True, True), (False, False)])
def test_eval_requires_both_environment_success_and_independent_validation(tmp_path, monkeypatch, environment_success, late):
    data = instance()
    if late:
        data.tw_s[1] = [1010., 1015.]
    monkeypatch.setattr(trainer, "_eval_instance_batches", lambda *a, **kw: iter([[data]]))
    monkeypatch.setattr(trainer, "make_terran_env", lambda **kw: SimpleNamespace())
    exported = row() | {
        "feasible": environment_success, "min_objective_distance_km": 5., "median_objective_distance_km": 5.,
        "vehicle_count": 2, "min_vehicle_count": 2, "median_vehicle_count": 2,
        "traj_feasible_rate": float(environment_success), "feasible_traj_count": int(environment_success), "runtime_s": .1,
    }
    monkeypatch.setattr(trainer, "_rollout_eval_batch_min_median", lambda *a, **kw: [exported])
    cfg = {"data": {"problem_type": "vrptw", "num_customers": 3},
           "evaluation": {"eval_path": str(tmp_path), "eval_output_dir": str(tmp_path / "eval"), "eval_save_routes": True}}
    metrics = trainer.evaluate_fixed_dataset(torch.nn.Linear(1, 1), cfg, seed=3009, epoch=1, device="cpu")
    saved = json.loads((tmp_path / "eval" / "epoch_0001.jsonl").read_text())
    assert saved["environment_feasible"] is environment_success
    assert saved["route_validation"]["valid"] is (not late)
    assert saved["feasible"] is (environment_success and not late)
    assert metrics["eval_feasible_rate"] == float(environment_success and not late)
    assert saved["feasibility_source"] == "environment_success_and_independent_vrptw_route_validation"
