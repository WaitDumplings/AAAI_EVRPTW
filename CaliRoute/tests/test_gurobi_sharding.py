from __future__ import annotations

from concurrent.futures import Future
import json
from types import SimpleNamespace

import pytest

from test_gurobi_resume import load_runner_without_gurobi, write_summary


@pytest.fixture
def shard_run(monkeypatch, tmp_path):
    runner = load_runner_without_gurobi("EVRPTW", monkeypatch)
    dataset = tmp_path / "test" / "Cus100" / "instances.pkl"
    dataset.parent.mkdir(parents=True)
    # Deliberately invalid pickle; tests must use the mocked deterministic iterator.
    dataset.write_bytes(b"not a pickle")
    instances = [
        SimpleNamespace(
            instance_id=f"territory_{index // 50:02d}_instance_{(index % 50) * 17 + 3:06d}",
            num_customers=100, metadata={}, region_id=f"territory_{index // 50:02d}",
        )
        for index in range(1000)
    ]
    submitted = []
    pools = []
    licenses = []
    monkeypatch.setattr(runner, "iter_instances", lambda path: iter(instances))
    monkeypatch.setattr(runner, "preflight_gurobi_license", lambda: licenses.append(True))
    monkeypatch.setattr(runner, "gurobi_version_string", lambda: "test")

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
            future.set_result({
                "instance_id": instance.instance_id,
                "instance_file": str(dataset), "solution": None,
                "summary_row": {
                    "instance_id": instance.instance_id, "file": str(dataset),
                    "status_name": "ERROR", "errors": "mock result; no solve executed",
                },
                "time_rows": [],
            })
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", RecordingPool)

    def execute(shard, *, output=None, extra=()):
        if output is None:
            output = tmp_path / "results" / f"shard_{shard}_of_5"
        runner.main([
            "--dataset_path", str(dataset), "--save_path", str(output),
            "--test_shard", str(shard), "--workers", "30", "--threads", "1",
            "--time_limit_s", "7200", "--cs_copies", "2", "--skip_completed",
            "--no-tie_break_vehicle_count", *extra,
        ])
        return output

    return SimpleNamespace(
        runner=runner, instances=instances, submitted=submitted, pools=pools,
        licenses=licenses, execute=execute, root=tmp_path,
    )


def check_manifest(output, shard, expected_ids):
    manifest = json.loads((output / "test_shard.json").read_text())
    assert manifest["shard_id"] == shard
    assert manifest["num_shards"] == 5
    assert manifest["start_offset"] == (shard - 1) * 200
    assert manifest["end_offset"] == shard * 200
    assert manifest["num_instances"] == 200
    assert manifest["total_instances"] == 1000
    assert manifest["instance_ids"] == expected_ids
    return manifest


def test_five_shards_cover_every_bundle_position_exactly_once(shard_run):
    all_ids = [instance.instance_id for instance in shard_run.instances]
    for shard in range(1, 6):
        start = len(shard_run.submitted)
        output = shard_run.execute(shard)
        expected = all_ids[(shard - 1) * 200:shard * 200]
        assert shard_run.submitted[start:] == expected
        check_manifest(output, shard, expected)
    assert shard_run.submitted == all_ids
    assert len(set(shard_run.submitted)) == 1000
    assert shard_run.pools == [30] * 5
    assert shard_run.licenses == [True] * 5


def test_shard_resume_skips_completed_without_reassigning_positions(shard_run, capsys):
    assigned = [instance.instance_id for instance in shard_run.instances[400:600]]
    completed = [assigned[1], assigned[5], assigned[-1]]
    output = shard_run.root / "results" / "shard_3_of_5"
    write_summary(output / "gurobi_summary.csv", [
        {"instance_id": completed[0], "status_name": "OPTIMAL"},
        {"instance_id": completed[1], "status_name": "TIME_LIMIT"},
        {"instance_id": completed[2], "status_name": "INFEASIBLE"},
        {"instance_id": assigned[0], "status_name": "ERROR"},
        {"instance_id": assigned[3], "status_name": "INTERRUPTED"},
        {"instance_id": shard_run.instances[0].instance_id, "status_name": "OPTIMAL"},
    ], shard_run.runner.SUMMARY_FIELDNAMES)
    shard_run.execute(3, output=output)
    assert shard_run.submitted == [iid for iid in assigned if iid not in completed]
    assert "skipped=3 pending=197" in capsys.readouterr().out
    check_manifest(output, 3, assigned)


def test_fully_completed_shard_does_not_start_license_or_pool(shard_run, capsys):
    assigned = [instance.instance_id for instance in shard_run.instances[800:1000]]
    output = shard_run.root / "results" / "shard_5_of_5"
    summary = output / "gurobi_summary.csv"
    write_summary(summary, [
        {"instance_id": iid, "status_name": "TIME_LIMIT"} for iid in assigned
    ], shard_run.runner.SUMMARY_FIELDNAMES)
    original = summary.read_bytes()
    shard_run.execute(5, output=output)
    manifest = check_manifest(output, 5, assigned)
    # The same slice and input remain valid for repeated resume attempts.
    shard_run.execute(5, output=output)
    assert check_manifest(output, 5, assigned) == manifest
    assert shard_run.submitted == []
    assert shard_run.pools == []
    assert shard_run.licenses == []
    assert summary.read_bytes() == original
    printed = capsys.readouterr().out
    assert "skipped=200 pending=0" in printed
    assert "No pending instances" in printed


@pytest.mark.parametrize("corruption", ["short", "long", "duplicate", "empty_id", "wrong_scale"])
def test_invalid_shard_dataset_fails_before_license_or_pool(shard_run, corruption):
    if corruption == "short":
        shard_run.instances.pop()
    elif corruption == "long":
        shard_run.instances.append(SimpleNamespace(
            instance_id="additional_unique", num_customers=100, metadata={}, region_id="extra",
        ))
    elif corruption == "duplicate":
        shard_run.instances[-1].instance_id = shard_run.instances[0].instance_id
    elif corruption == "empty_id":
        shard_run.instances[-1].instance_id = "   "
    else:
        shard_run.instances[-1].num_customers = 50
    with pytest.raises(ValueError):
        shard_run.execute(1)
    assert shard_run.submitted == []
    assert shard_run.pools == []
    assert shard_run.licenses == []


@pytest.mark.parametrize("extra", [
    ("--start_index", "0"), ("--end_index", "200"), ("--limit", "200"),
    ("--scales", "Cus100"), ("--expert_summary_path", "missing.csv"),
])
def test_shards_reject_other_instance_selectors_before_loading(shard_run, monkeypatch, extra):
    def unexpected_load(path):
        pytest.fail("Conflicting selectors should fail before loading instances")
    monkeypatch.setattr(shard_run.runner, "iter_instances", unexpected_load)
    with pytest.raises((ValueError, SystemExit)):
        shard_run.execute(1, extra=extra)
    assert shard_run.submitted == []
    assert shard_run.pools == []
    assert shard_run.licenses == []


def test_existing_shard_manifest_rejects_changed_assignment(shard_run):
    output = shard_run.execute(2)
    manifest_path = output / "test_shard.json"
    original = manifest_path.read_bytes()
    summary = output / "gurobi_summary.csv"
    original_summary = summary.read_bytes()
    # Reordering input would silently move an ID between servers without a guard.
    shard_run.instances[0], shard_run.instances[200] = shard_run.instances[200], shard_run.instances[0]
    shard_run.submitted.clear()
    shard_run.pools.clear()
    shard_run.licenses.clear()
    with pytest.raises(ValueError):
        shard_run.execute(2, output=output)
    assert shard_run.submitted == []
    assert shard_run.pools == []
    assert shard_run.licenses == []
    assert manifest_path.read_bytes() == original
    assert summary.read_bytes() == original_summary
