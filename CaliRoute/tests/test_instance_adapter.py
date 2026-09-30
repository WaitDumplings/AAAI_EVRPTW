from __future__ import annotations

import pickle

import numpy as np
import pytest

from offline2online.instance_adapter import (
    CLASSICAL_BUNDLE_FORMAT,
    adapt_instance_payload,
    iter_adapted_instances,
    num_charging_stations_for_problem,
)

try:
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    make_terran_env = None


def _base_payload(problem_class: str = "CVRP") -> dict:
    return {
        "problem_class": problem_class,
        "instance_id": "toy_000",
        "depot": np.array([0.0, 0.0], dtype=np.float32),
        "customers": np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "distance_matrix_km": np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 2.2, 2.1],
                [2.0, 2.2, 0.0, 2.0],
                [3.0, 2.1, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "demands_cm3": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "package_counts": np.array([1, 1, 1], dtype=np.int32),
        "vehicle": {"cargo_capacity_cm3": 2.0},
    }


def test_cvrp_adapter_disables_battery_charging_and_time_windows() -> None:
    instance = adapt_instance_payload(_base_payload("CVRP"), problem_type="cvrp")

    assert instance.num_customers == 3
    assert instance.num_charging_stations == 0
    assert instance.distance_matrix_km.shape == (4, 4)
    assert instance.vehicle["consumption_kwh_per_km"] == 0.0
    assert instance.working_start_s == 0
    assert instance.working_end_s > 100_000
    assert np.all(instance.tw_s[:, 0] == 0)
    assert np.all(instance.tw_s[:, 1] == instance.working_end_s)

    if make_terran_env is None:
        pytest.skip("gymnasium is not installed in this test environment")
    env = make_terran_env(instance=instance, n_traj=2, use_fast_env=True, info_level="light")
    obs, _ = env.reset(seed=7)
    assert obs["action_mask"].shape == (2, 4)
    assert obs["rs_loc"].shape == (0, 2)
    assert np.allclose(obs["edge_energy"], 0.0)
    assert obs["action_mask"][:, 1:].all()


def test_vrptw_adapter_keeps_time_windows_and_removes_charging() -> None:
    payload = _base_payload("VRPTW")
    payload.update(
        {
            "working_start_s": 100,
            "working_end_s": 1000,
            "tw_s": np.array([[120, 500], [150, 600], [200, 900]], dtype=np.float32),
            "service_time_s": np.array([10, 20, 30], dtype=np.float32),
            "speed_profile": {"effective_speed_kmh": 3600.0},
        }
    )
    instance = adapt_instance_payload(payload, problem_type="vrptw")

    assert instance.num_charging_stations == 0
    assert instance.working_start_s == 100
    assert instance.working_end_s == 1000
    assert np.allclose(instance.tw_s, payload["tw_s"])
    assert np.allclose(instance.service_time_s, payload["service_time_s"])
    assert instance.vehicle["consumption_kwh_per_km"] == 0.0


def test_classical_bundle_stream_is_supported(tmp_path) -> None:
    path = tmp_path / "instances.pkl"
    payload = _base_payload("CVRP")
    with path.open("wb") as f:
        pickle.dump(
            {
                "format": CLASSICAL_BUNDLE_FORMAT,
                "num_instances": 2,
                "num_customers": 3,
                "dataset_metadata": {"problem_class": "CVRP"},
            },
            f,
        )
        pickle.dump(payload, f)
        payload_2 = dict(payload)
        payload_2["instance_id"] = "toy_001"
        pickle.dump(payload_2, f)

    instances = list(
        iter_adapted_instances(
            path,
            num_customers=3,
            num_charging_stations=0,
            problem_type="cvrp",
        )
    )
    assert [instance.instance_id for instance in instances] == ["toy_000", "toy_001"]


def test_classical_problem_defaults_to_zero_charging_stations() -> None:
    assert num_charging_stations_for_problem({"problem_type": "cvrp"}, "cvrp") == 0
