from __future__ import annotations

from concurrent.futures import Future
import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from EVRPTW_Benchmark.Exact.Gurobi_Solver.resume import completed_instance_ids, read_completed_ids


CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "EVRPTW_Benchmark.Exact.Gurobi_Solver"


def write_summary(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_completed_ids_normalize_status_and_use_latest_record():
    rows = [
        {"instance_id": "optimal", "status_name": " optimal "},
        {"instance_id": "limited", "status_name": "", "status": 9, "feasible": False},
        {"instance_id": "numeric_optimal", "status": 2},
        {"instance_id": "infeasible", "status_name": "INFEASIBLE", "feasible": False},
        {"instance_id": "retry", "status_name": "OPTIMAL"},
        {"instance_id": "retry", "status_name": " error "},
        {"instance_id": "recovered", "status_name": "ERROR"},
        {"instance_id": "recovered", "status_name": "TIME_LIMIT"},
        {"instance_id": "", "status_name": "OPTIMAL"},
    ]
    for status in ("", "ERROR", "INVALID_INSTANCE", "INTERRUPTED", "LOADED", "INPROGRESS", 1, 11, 14):
        rows.append({"instance_id": f"unfinished_{status}", "status_name": status})
    assert completed_instance_ids(rows) == {
        "optimal", "limited", "numeric_optimal", "infeasible", "recovered",
    }


def test_resume_summary_missing_empty_and_malformed(tmp_path):
    summary = tmp_path / "gurobi_summary.csv"
    assert read_completed_ids(summary) == set()
    summary.touch()
    assert read_completed_ids(summary) == set()
    summary.write_text("status_name\nOPTIMAL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="instance_id"):
        read_completed_ids(summary)


def load_runner_without_gurobi(problem, monkeypatch):
    """Import real orchestration with its solver dependency stubbed, never a license."""
    solver_name = f"{PACKAGE}.{problem}.gurobi_solver"
    solver = ModuleType(solver_name)
    solver.GurobiSolverConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    setattr(solver, f"Gurobi{problem}Solver", SimpleNamespace(name="unused_test_solver"))
    solver.MAX_GUROBI_TIME_LIMIT_S = 7200.0
    solver.capped_time_limit_s = lambda value: min(float(value), 7200.0)
    monkeypatch.setitem(sys.modules, solver_name, solver)
    path = CODE_ROOT / "EVRPTW_Benchmark" / "Exact" / "Gurobi_Solver" / problem / "run_gurobi.py"
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{problem}._resume_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("problem", ["CVRP", "VRPTW", "EVRPTW"])
@pytest.mark.parametrize("scenario", ["mixed", "all_completed", "resume_disabled"])
def test_runners_submit_only_pending_instances(problem, scenario, monkeypatch, tmp_path, capsys):
    runner = load_runner_without_gurobi(problem, monkeypatch)
    dataset = tmp_path / "test" / "Cus15" / "instances.pkl"
    dataset.parent.mkdir(parents=True)
    # Loading real instances would fail: only the mocked iterator supplies records.
    dataset.write_bytes(b"not a pickle")
    output = tmp_path / "results"
    summary = output / "gurobi_summary.csv"
    done = ["optimal_000000", "limited_000001", "infeasible_000002"]
    pending = ["error_000003", "invalid_000004", "interrupted_000005", "blank_000006", "checkpoint_000007", "new_000008"]
    rows = [
        {"instance_id": done[0], "status_name": "OPTIMAL", "feasible": True, "objective_distance_km": 4},
        {"instance_id": done[1], "status_name": "TIME_LIMIT", "feasible": False},
        {"instance_id": done[2], "status_name": "INFEASIBLE", "feasible": False},
        {"instance_id": pending[0], "status_name": "ERROR"},
        {"instance_id": pending[1], "status_name": "INVALID_INSTANCE"},
        {"instance_id": pending[2], "status_name": "INTERRUPTED"},
        {"instance_id": pending[3], "status_name": ""},
        # This completed ID does not belong to the selected bundle.
        {"instance_id": "unrelated_999999", "status_name": "OPTIMAL"},
    ]
    write_summary(summary, rows, runner.SUMMARY_FIELDNAMES)
    original = summary.read_bytes()
    with summary.open(newline="", encoding="utf-8") as stream:
        original_rows = {row["instance_id"]: row for row in csv.DictReader(stream)}
    checkpoint = output / "solutions" / "checkpoints" / f"{pending[4]}_60s_solution.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial checkpoint is not a completed solve")

    selected_ids = done if scenario == "all_completed" else [*done, *pending]
    instances = [
        SimpleNamespace(instance_id=iid, num_customers=15, metadata={}, region_id="test_region")
        for iid in selected_ids
    ]
    monkeypatch.setattr(runner, "iter_instances", lambda path: iter(instances))
    submitted = []
    pools = []
    licenses = []

    class RecordingPool:
        def __init__(self, max_workers):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, instance, *args):
            submitted.append(instance.instance_id)
            future = Future()
            # Exercise actual coordinator CSV upserts, without executing a solver.
            future.set_result({
                "instance_id": instance.instance_id,
                "instance_file": str(dataset),
                "solution": None,
                "error": "mock solve result",
                "traceback": "",
                "summary_row": {
                    "instance_id": instance.instance_id,
                    "file": str(dataset),
                    "status_name": "ERROR",
                    "errors": "mock solve result",
                },
                "time_rows": [],
            })
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", RecordingPool)
    if problem == "EVRPTW":
        monkeypatch.setattr(runner, "preflight_gurobi_license", lambda: licenses.append(True))
        monkeypatch.setattr(runner, "gurobi_version_string", lambda: "test")
    arguments = ["--dataset_path", str(dataset), "--save_path", str(output), "--workers", "30"]
    if scenario != "resume_disabled":
        arguments.append("--skip_completed")
    runner.main(arguments)
    printed = capsys.readouterr().out
    assert f"summary={summary}" in printed

    if scenario == "all_completed":
        assert submitted == []
        assert pools == []
        assert licenses == []
        assert "Resume: enabled; skipped=3 pending=0" in printed
        assert "No pending instances" in printed
        assert summary.read_bytes() == original
    else:
        expected = selected_ids if scenario == "resume_disabled" else pending
        assert submitted == expected
        assert len(pools) == 1
        if problem == "EVRPTW":
            assert licenses == [True]
        with summary.open(newline="", encoding="utf-8") as stream:
            saved_rows = {row["instance_id"]: row for row in csv.DictReader(stream)}
        assert saved_rows["unrelated_999999"] == original_rows["unrelated_999999"]
        if scenario == "mixed":
            assert "Resume: enabled; skipped=3 pending=6" in printed
            for iid in done:
                assert saved_rows[iid] == original_rows[iid]
        else:
            assert "Resume: disabled; skipped=0 pending=9" in printed
