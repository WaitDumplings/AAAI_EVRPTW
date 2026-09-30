from __future__ import annotations

import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = CODE_ROOT / "EVRPTW_Benchmark" / "Exact" / "Gurobi_Solver"
PACKAGE = "EVRPTW_Benchmark.Exact.Gurobi_Solver"


@pytest.mark.parametrize("problem", ["CVRP", "VRPTW", "EVRPTW"])
def test_dry_run_does_not_import_gurobi(problem, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "gurobipy", None)
    destination = tmp_path / "must_not_be_created"
    monkeypatch.setattr(sys, "argv", [
        "run_range", "--split", "test", "--scale", "Cus15",
        "--start_index", "0", "--end_index", "1", "--dry-run",
        "--output_path", str(destination),
    ])
    runner = importlib.import_module(f"{PACKAGE}.{problem}.run_range")
    runner.main()
    printed = capsys.readouterr().out
    assert str(CODE_ROOT.parent / "AAAI_Dataset" / "test_release" / problem.lower() / "test" / "Cus15") in printed
    assert str(destination) in printed
    assert "--skip_completed" in printed
    assert not destination.exists()


@pytest.mark.parametrize("problem", ["CVRP", "VRPTW", "EVRPTW"])
def test_direct_and_shell_entry_points_outside_checkout(problem, tmp_path):
    destination = tmp_path / "output with spaces"
    dataset = tmp_path / "input with spaces"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    direct = [
        sys.executable, "-B", str(BENCHMARK / problem / "run_range.py"),
        "--start_index", "0", "--end_index", "1", "--dry-run",
        "--dataset_path", str(dataset), "--output_path", str(destination),
    ]
    shell = [
        "bash", str(BENCHMARK / problem / "run_gurobi_range.sh"),
        "--start", "0", "--end", "1", "--dry-run", "--detach",
        "--dataset-path", str(dataset), "--output-path", str(destination),
        "--python", sys.executable,
    ]
    for command in (direct, shell):
        result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
        assert f"Dataset path: {dataset}" in result.stdout
        assert f"Output path: {destination}" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_classical_schemas_remain_distinct_and_serializable():
    cvrp = importlib.import_module(f"{PACKAGE}.CVRP.classical_core.schema")
    vrptw = importlib.import_module(f"{PACKAGE}.VRPTW.classical_core.schema")
    payload = {
        "instance_id": "toy_000000", "demands_cm3": [1.0],
        "distance_matrix_km": [[0.0, 1.0], [1.0, 0.0]],
        "vehicle": {"cargo_capacity_cm3": 2.0},
        "travel_time_matrix_s": [[0.0, 60.0], [60.0, 0.0]],
        "tw_s": [[0.0, 3600.0]], "service_time_s": [10.0],
        "working_start_s": 0.0, "working_end_s": 3600.0,
    }
    capacity_instance = cvrp.ClassicalVRPInstance.from_dict(payload)
    time_window_instance = vrptw.ClassicalVRPInstance.from_dict(payload)
    assert type(capacity_instance) is not type(time_window_instance)
    assert not hasattr(capacity_instance, "tw_s")
    restored = pickle.loads(pickle.dumps(time_window_instance))
    assert type(restored) is vrptw.ClassicalVRPInstance
    np.testing.assert_array_equal(restored.tw_s, [[0.0, 3600.0]])
