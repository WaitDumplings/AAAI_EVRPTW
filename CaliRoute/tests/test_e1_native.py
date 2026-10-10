"""Native E1 adapters must preserve algorithm, metric, candidate budget and recovery."""
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from e1 import native
from e1.native_worker import action_routes, evaluate_runner
from offline2online.instance_adapter import adapt_instance_payload


def payload(identity="train", scale=1., customers=100):
    xy = np.stack((np.arange(customers+1), np.arange(customers+1) % 7), axis=-1).astype(np.float32)
    distance = np.abs(np.arange(customers+1)[:, None]-np.arange(customers+1)[None, :]).astype(np.float32)+1.
    distance *= np.where(np.arange(customers+1)[:, None] < np.arange(customers+1)[None, :], 1.4, 1.)
    distance *= scale
    np.fill_diagonal(distance, 0.)
    return dict(instance_id=identity, problem_type="cvrp", working_start_s=0, working_end_s=10000,
        depot=xy[0], customers=xy[1:], charging_stations=np.empty((0, 2), np.float32),
        distance_matrix_km=distance, demands_cm3=np.ones(customers, np.float32),
        package_counts=np.ones(customers, np.int32), service_time_s=np.zeros(customers, np.float32),
        tw_s=np.tile([0., 10000.], (customers, 1)), cs_time_to_depot_s=np.empty(0, np.float32),
        vehicle=dict(cargo_capacity_cm3=20.), metadata={"service_territory_id": "native_test_city"})


def bundle(path, instances):
    path.mkdir(parents=True)
    with (path / "instances.pkl").open("wb") as stream:
        pickle.dump(dict(format="classical_vrp_instance_bundle_v1", num_instances=len(instances), num_customers=100), stream)
        for instance in instances: pickle.dump(instance, stream)
    return path


def config(tmp_path, method, *, val_count=1):
    train = bundle(tmp_path / "train", [payload("train0"), payload("train1", 1.2)])
    val = bundle(tmp_path / "val", [payload(f"val{i}", 1.4+0.2*i) for i in range(val_count)])
    return dict(method=method, training_seed=3009, train_path=str(train), val_path=str(val),
                instance_exposures=2, batch_size=1, n_traj=50, eval_k=50,
                eval_interval_exposures=1, eval_batch_size=1, checkpoint_interval_updates=1,
                experiment_protocol={"smoke_only": True})


def test_converter_preserves_asymmetry_capacity_and_id_without_euclidean_cost():
    raw = payload()
    instance = adapt_instance_payload(raw, problem_type="cvrp", strict_road_metric=True)
    arrays, identities = native.convert_cvrp_instances([instance])
    np.testing.assert_array_equal(arrays["distance_matrix"][0], raw["distance_matrix_km"])
    assert arrays["distance_matrix"][0, 1, 2] != arrays["distance_matrix"][0, 2, 1]
    np.testing.assert_array_equal(arrays["demand"][0], raw["demands_cm3"])
    assert arrays["capacity"][0] == raw["vehicle"]["cargo_capacity_cm3"]
    assert identities[0]["territory_id"] == "native_test_city"
    assert identities[0]["semantic_sha256"]
    assert not any("time" in key or "energy" in key or "expert" in key for key in arrays)


@pytest.mark.parametrize("bad", ["nonzero_diagonal", "nan", "too_large_demand", "duplicate"])
def test_converter_rejects_invalid_native_contract(bad):
    raw = payload()
    if bad == "nonzero_diagonal": raw["distance_matrix_km"][1, 1] = 1
    if bad == "nan": raw["distance_matrix_km"][1, 2] = np.nan
    if bad == "too_large_demand": raw["demands_cm3"][0] = 21
    with pytest.raises(ValueError):
        instance = adapt_instance_payload(raw, problem_type="cvrp", strict_road_metric=True)
        native.convert_cvrp_instances([instance, instance] if bad == "duplicate" else [instance])


def test_native_sequence_conventions_are_explicit_and_no_missing_customer_repair():
    assert action_routes([0, 1, 2, 0, 0, 0]) == [[0, 1, 2, 0]]
    assert action_routes([1, 0, 2], implicit_final_return=True) == [[0, 1, 0], [0, 2, 0]]
    with pytest.raises(ValueError): action_routes([0, 1, 2])


def test_source_freeze_detects_changes_and_has_no_pretrained_or_datasets(tmp_path):
    for method in ("rrnco", "radar"):
        target = tmp_path / method
        manifest = native.freeze_source(method, native.DEFAULT_SOURCES[method], target)
        native.verify_files(target, manifest)
        assert manifest["upstream_provenance"]["git_head"]
        assert not any(Path(item["path"]).suffix in {".pt", ".ckpt", ".pkl", ".npz"} for item in manifest["files"])
        selected = target / native.REQUIRED[method][0]
        selected.write_text(selected.read_text()+"\n# changed\n")
        with pytest.raises(ValueError, match="fingerprint changed"): native.verify_files(target, manifest)


def test_e1_native_config_refuses_fake_distributed_execution(tmp_path):
    cfg = config(tmp_path, "radar")
    with pytest.raises(ValueError, match="masquerade as DDP"):
        native._config({**cfg, "world_size": 2})
    resolved = native._config(cfg)
    assert resolved["native_algorithm"] == "native_REINFORCE_POMO"
    assert resolved["initialization"] == "scratch" and resolved["num_augment"] == 1
    assert resolved["native_learning_rate"] == 4e-4


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor): torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray): np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): _assert_equal(a, b)
    else: assert left == right


@pytest.mark.parametrize("method", ["radar", "rrnco"])
def test_native_cpu_update_validation_fresh_process_resume_matches_continuation(tmp_path, method):
    available = native.native_status(method)
    if not available["source_available"] or not available["dependencies_available"]:
        pytest.skip(f"Native source/environment unavailable: {available}")
    cfg = config(tmp_path, method)
    interrupted, uninterrupted = tmp_path / "resumed", tmp_path / "straight"
    first = native.train(cfg, interrupted, max_updates=1)
    assert first["state"] == "paused" and first["counters"]["optimizer_updates"] == 1
    assert first["checkpoint"]["backup_verified"] and not first["training_complete"]
    # Recovery works even when the primary checkpoint is unavailable.
    (interrupted / "native_checkpoints/last.pt").unlink()
    resumed = native.train({**cfg, "resume_checkpoint": first["checkpoint"]["backup_path"]}, interrupted, resume=True)
    recovery_event = json.loads((interrupted / "native_resume.jsonl").read_text().splitlines()[0])
    assert recovery_event["checkpoint"] == first["checkpoint"]["backup_path"]
    straight = native.train(cfg, uninterrupted)
    assert resumed["training_complete"] and straight["training_complete"]
    assert resumed["counters"]["instance_exposures"] == 2
    assert resumed["counters"]["complete_rollouts"] == 100
    assert resumed["counters"]["optimizer_updates"] == 2
    left = torch.load(interrupted / "native_checkpoints/last.pt", map_location="cpu", weights_only=False)
    right = torch.load(uninterrupted / "native_checkpoints/last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "sampler", "rng", "best_metric"):
        _assert_equal(left[key], right[key])
    # Resume is configuration/identity-checked, not an actor-only checkpoint.
    with pytest.raises(ValueError, match="configuration differs"):
        native.train({**cfg, "training_seed": 3010}, interrupted, resume=True)
    selected = interrupted / "native_checkpoints/best.pt"
    report = native.evaluate(cfg, interrupted, selected)
    assert report["instances"] == 1 and report["actual_candidates"] == 50
    assert report["cost_mismatch_candidates"] == 0
    records = [json.loads(line) for line in Path(report["output"]).read_text().splitlines()]
    assert records[0]["requested_K"] == records[0]["actual_K"] == 50
    assert records[0]["num_augment"] == 1 and records[0]["split"] == "val"
    assert "encoding/SVD" in records[0]["timing_scope"]
    assert records[0]["selected_cost_matches_reported"] is True
    assert not (interrupted / "native_data/test.npz").exists()
    assert native.file_hash(selected) == native.file_hash(Path(json.loads((interrupted / "native_resolved.json").read_text())["backup_dir"]) / "best.pt")


def test_native_runtime_resume_path_and_portable_interpreter_override(tmp_path, monkeypatch):
    cfg = config(tmp_path, "rrnco")
    monkeypatch.setenv("E1_RRNCO_PYTHON", "/portable/rrnco/bin/python")
    resolved = native._config({**cfg, "resume_checkpoint": "/backup/last.pt"})
    assert resolved["native_python"] == "/portable/rrnco/bin/python"
    assert "resume_checkpoint" not in resolved
    with pytest.raises(ValueError, match="scratch training never loads"):
        native.train({**cfg, "resume_checkpoint": "/backup/last.pt"}, tmp_path / "run")
    readiness = native.native_status("rrnco")
    assert not readiness["dependencies_available"]
    assert readiness["interpreter_environment_variable"] == "E1_RRNCO_PYTHON"
    assert Path(readiness["requirements_lock"]).is_file()
    assert "Python 3.12" in readiness["setup_hint"]


def test_native_eval_limit_requires_hardware_preflight(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import e1.native_worker as worker
    cfg = config(tmp_path, "radar")
    with pytest.raises(ValueError, match="hardware_preflight_only"):
        native._config({**cfg, "eval_limit": 1})
    cfg = native._config({**cfg, "eval_limit": 1, "experiment_protocol": {"hardware_preflight_only": True}})
    dummy_instance = adapt_instance_payload(payload(), problem_type="cvrp", strict_road_metric=True)
    arrays, identities = native.convert_cvrp_instances([dummy_instance])
    metadata = {"count": 1000, "instances": identities}
    monkeypatch.setattr(worker, "load_data", lambda *args: (arrays, metadata))
    # Route failure is retained; even unsuccessful candidates count against K.
    class Runner:
        device = torch.device("cpu")
        model = torch.nn.Identity()
        def decode(self, arrays, start, stop, k):
            assert (start, stop, k) == (0, 1, 50)
            return np.zeros((1, 50, 1), dtype=int), np.ones((1, 50)), np.zeros((1, 50), dtype=bool)
    result = evaluate_runner(Runner(), cfg, tmp_path, split="val", checkpoint_id="preflight",
                             output=tmp_path / "validation.jsonl", inference_seed=42)
    assert result["instances"] == 1 and result["actual_candidates"] == 50
    with pytest.raises(ValueError, match="only for hardware preflight"):
        evaluate_runner(Runner(), cfg, tmp_path, split="test", checkpoint_id="invalid",
                        output=tmp_path / "test.jsonl", inference_seed=42)
    with pytest.raises(ValueError, match="all 1000"):
        monkeypatch.setattr(worker, "load_data", lambda *args: (arrays, {"count": 1, "instances": identities}))
        evaluate_runner(Runner(), cfg, tmp_path, split="val", checkpoint_id="invalid",
                        output=tmp_path / "invalid.jsonl", inference_seed=42)


@pytest.mark.parametrize("method", ["radar", "rrnco"])
def test_native_evaluation_batch_override_preserves_checkpoint_identity_and_k(tmp_path, method):
    available = native.native_status(method)
    if not available["source_available"] or not available["dependencies_available"]:
        pytest.skip(f"Native source/environment unavailable: {available}")
    cfg = config(tmp_path, method, val_count=2)
    cfg.update(instance_exposures=1, run_id=f"E1_UNIQUE_CAMPAIGN_{method}")
    run = tmp_path / method
    native.train(cfg, run)
    selected = run / "native_checkpoints/best.pt"
    resolved = run / "native_resolved.json"
    original_config_hash = native.file_hash(resolved)
    original_checkpoint_hash = native.file_hash(selected)
    for batch in (1, 2):
        report = native.evaluate(cfg, run, selected, eval_batch_size=batch,
                                 output_path=run / f"batch{batch}.jsonl")
        assert report["evaluation_batch_size"] == batch
        assert report["instances"] == 2 and report["actual_candidates"] == 100
        records = [json.loads(line) for line in Path(report["output"]).read_text().splitlines()]
        assert all(row["timing_batch_size"] == batch for row in records)
        assert all(row["requested_K"] == row["actual_K"] == 50 for row in records)
        assert all(row["run_id"] == cfg["run_id"] for row in records)
        assert all(row["runtime_kind"] == ("single_instance_latency" if batch == 1 else "batch_amortized") for row in records)
        assert report["cost_mismatch_candidates"] == 0
        assert native.file_hash(resolved) == original_config_hash
        assert native.file_hash(selected) == original_checkpoint_hash
    assert json.loads(resolved.read_text())["eval_batch_size"] == 1
    training_status_hash = native.file_hash(run / "native_status.json")
    with pytest.raises(RuntimeError, match="native_evaluation_status.json"):
        native.evaluate(cfg, run, selected.with_name("missing.pt"), eval_batch_size=1)
    assert native.file_hash(run / "native_status.json") == training_status_hash
    assert json.loads((run / "native_status.json").read_text())["training_complete"] is True
    assert json.loads((run / "native_evaluation_status.json").read_text())["state"] == "failed"
    with pytest.raises(ValueError, match="positive integer"):
        native.evaluate(cfg, run, selected, eval_batch_size=0)
