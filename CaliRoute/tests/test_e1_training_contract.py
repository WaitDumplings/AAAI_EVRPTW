"""Opt-in E1 cost-only SL references, recovery and actual CPU/Gloo training.

Four-customer fixtures make this a short implementation smoke, not a Cus100
performance result. Production method weights, four PPO passes and four
minibatches are retained; embeddings/trajectories/chunks are reduced here.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline2online import trainer
from caliroute.methods import method_preset

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    preset = method_preset("slppo")
    cfg = {"offline": preset.offline_config(), "advantage": preset.advantage_config(),
           "training": {"reward_norm_mode": "legacy"}, "data": {"num_customers": 4}}
    cfg["offline"].update(solution_reference_contract=trainer.E1_SOLUTION_REFERENCE_CONTRACT,
                          use_priority_sampler=False)
    trainer._apply_solution_level_aliases(cfg)
    return cfg


def _batch(costs, *, success=None, done=None, served=None):
    costs = np.asarray(costs, dtype=float)
    if costs.ndim == 1:
        costs = costs[None]
    success = np.ones_like(costs, dtype=bool) if success is None else np.asarray(success).reshape(costs.shape)
    served = np.full_like(costs, 4) if served is None else np.asarray(served).reshape(costs.shape)
    done = np.ones_like(costs, dtype=bool) if done is None else np.asarray(done).reshape(costs.shape)
    actions = torch.zeros((2, *costs.shape), dtype=torch.long)
    return SimpleNamespace(actions=actions, old_logprobs=torch.zeros_like(actions, dtype=torch.float),
        dones=torch.as_tensor(np.stack([np.zeros_like(done), done])), valid=torch.ones_like(actions, dtype=torch.bool),
        final_infos=[dict(objective_distance_km=row, success=suc, served_customers=srv)
                     for row, suc, srv in zip(costs, success, served)])


def _envs(*ids):
    return [SimpleNamespace(instance=SimpleNamespace(instance_id=identity)) for identity in ids]


class CostBuffer:
    def __init__(self, **costs):
        self.costs = costs

    def reference_objective(self, identity):
        return self.costs.get(identity)


@pytest.mark.parametrize("expert,memory,expected", [
    (100., 80., 80.), (80., 100., 80.), (None, 80., 80.),
    (100., None, 100.), (None, None, None), (float("nan"), 80., 80.),
    (float("inf"), 80., 80.), (-1., 80., 80.), (100., 0., 0.),
])
def test_reference_is_finite_minimum_of_expert_and_old_incumbent(expert, memory, expected):
    old = {"a": memory}
    buffer = CostBuffer(a=expert)
    snapshot = trainer._E1CostReferenceSnapshot(buffer, old, ["a", "a"])
    old["a"] = 1.
    buffer.costs["a"] = 2.
    assert snapshot.reference_objective("a") == expected


def test_missing_expert_keeps_policy_solution_learning_and_legacy_is_unchanged():
    cfg, batch, envs = _cfg(), _batch([110., 120.]), _envs("a")
    actual, success, info = trainer._solution_level_advantage_tensors(
        batch, cfg, envs, None, "cpu", {"a": 90.})
    expected_cfg = copy.deepcopy(cfg)
    expected_cfg["offline"].pop("solution_reference_contract")
    expected, _, _ = trainer._solution_level_advantage_tensors(
        batch, expected_cfg, envs, CostBuffer(a=90.), "cpu", {"a": 90.})
    torch.testing.assert_close(actual, expected)
    assert success.all() and torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert info["sl_group_reference_count"] == 1
    assert info["sl_reference_available_instances"] == 1
    legacy_missing, _, legacy_info = trainer._solution_level_advantage_tensors(
        batch, expected_cfg, envs, None, "cpu", {"a": 90.})
    assert legacy_info["sl_group_reference_count"] == 0
    assert not torch.allclose(legacy_missing, actual)
    assert trainer._prepare_sl_expert_candidates(None, batch, cfg, envs, None, {"a": 90.}, "cpu") == ([], {})


def test_no_references_has_finite_group_signal_and_no_reference_signal():
    advantages, _, info = trainer._solution_level_advantage_tensors(
        _batch([110., 120.]), _cfg(), _envs("a"), None, "cpu", {})
    assert torch.isfinite(advantages).all() and advantages[0, 0] > 0 > advantages[0, 1]
    assert info["ref_adv_mean"] == 0 and info["sl_group_reference_count"] == 0


def test_partial_failed_and_nonfinite_routes_cannot_train_sl_or_update_incumbent():
    batch = _batch([1., 2., np.nan, np.inf, 3., 100., 90.],
        success=[1, 0, 1, 1, 1, 1, 1], done=[0, 1, 1, 1, 1, 1, 1],
        served=[4, 4, 4, 4, 3, 4, 4])
    cfg, envs, memory = _cfg(), _envs("a"), {"a": 95.}
    with np.errstate(invalid="ignore"):
        advantages, feasible, _ = trainer._solution_level_advantage_tensors(batch, cfg, envs, None, "cpu", memory)
    assert feasible.tolist() == [[False, False, False, False, False, True, True]]
    assert advantages[0, :5].eq(0).all() and torch.isfinite(advantages).all()
    assert memory == {"a": 95.}
    trainer._update_policy_best_objectives(memory, batch, envs, cfg=cfg)
    assert memory == {"a": 90.}


def test_duplicate_batch_ids_use_old_cost_then_merge_future_costs_across_ranks():
    cfg, envs, memory = _cfg(), _envs("a", "a"), {"a": 100.}
    batch = _batch([[50., 60.], [120., 130.]])
    actual, _, _ = trainer._solution_level_advantage_tensors(batch, cfg, envs, None, "cpu", memory)
    single, _, _ = trainer._solution_level_advantage_tensors(
        _batch([120., 130.]), cfg, _envs("a"), None, "cpu", memory)
    torch.testing.assert_close(actual[1], single[0])
    distributed = SimpleNamespace(gather_objects=lambda local: [local, {"b": 70., "a": 40.}])
    trainer._update_policy_best_objectives(memory, batch, envs, cfg=cfg, distributed=distributed)
    assert memory == {"a": 40., "b": 70.}


@pytest.mark.parametrize("section,key,value", [
    ("training", "reward_norm_mode", "physical_shared_popart"),
    ("offline", "policy_replay_enabled", True),
    ("offline", "branch_exploration_enabled", True),
    ("offline", "exploration_enabled", True),
    ("offline", "method", "ppo"),
    ("advantage", "use_expert_solution_level", False),
    ("advantage", "sl_candidate_use_expert_candidate", False),
    ("advantage", "sl_use_memory_incumbent", False),
])
def test_e1_rejects_silent_objective_changes(section, key, value):
    cfg = _cfg()
    trainer._validate_e1_solution_contract(cfg)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        trainer._validate_e1_solution_contract(cfg)
    cfg["offline"].pop("solution_reference_contract")
    trainer._validate_e1_solution_contract(cfg)  # Historical paths remain allowed.


@pytest.mark.parametrize("epoch,evaluate", [(3, False), (50, False), (53, True), (100, False), (103, True), (1503, True)])
def test_dapg_validation_aligns_to_online_loops_without_moving_archives(epoch, evaluate):
    plan = trainer._training_checkpoint_plan(epoch, 1503, eval_interval=50,
        checkpoint_interval=50, latest_checkpoint_interval=5, eval_epoch_offset=3)
    assert plan.evaluate == evaluate
    assert plan.archive == (epoch % 50 == 0 or epoch == 1503)
    assert plan.latest == (epoch % 5 == 0 or epoch == 1503)


def test_checkpoint_backup_is_atomic_verified_and_primary_survives_failure(tmp_path, monkeypatch):
    agent = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(agent.parameters())
    cfg = dict(run_name="E1_BACKUP", training={"checkpoint_backup_dir": str(tmp_path / "backup")})
    path = tmp_path / "primary/checkpoint_latest.pt"
    trainer.save_checkpoint(path, agent, optimizer, cfg, 1, 3009)
    backup = tmp_path / "backup/E1_BACKUP/seed_3009/checkpoint_latest.pt"
    manifest = json.loads(backup.with_suffix(".pt.backup.json").read_text())
    assert manifest["verified"] and manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert backup.read_bytes() == path.read_bytes()
    previous = backup.read_bytes()
    def fail(source, destination, length):
        destination.write(b"partial")
        raise OSError("interrupted backup")
    monkeypatch.setattr(trainer.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="interrupted backup"):
        trainer.save_checkpoint(path, agent, optimizer, cfg, 2, 3009)
    assert torch.load(path, weights_only=False)["epoch"] == 2
    assert backup.read_bytes() == previous


def test_restore_rejects_changed_dataset_order_or_content_and_bad_costs():
    from test_aaai_core_training import _payload
    instances = [SimpleNamespace(**_payload(identity, "vrptw")) for identity in ("a", "b")]
    pool = SimpleNamespace(instances=instances)
    cfg = _cfg()
    identity = trainer._e1_incumbent_identity(pool, cfg)
    state = {"policy_incumbent_identity": identity, "policy_route_pool": None,
             "sampler": {"supported": True}, "policy_best_objectives": {"a": 5.}}
    trainer._validate_e1_resume_state(state, identity)
    changed = trainer._e1_incumbent_identity(SimpleNamespace(instances=instances[::-1]), cfg)
    with pytest.raises(ValueError, match="identity"):
        trainer._validate_e1_resume_state(state, changed)
    instances[0].distance_matrix_km[1, 2] += 1.
    with pytest.raises(ValueError, match="identity"):
        trainer._validate_e1_resume_state(state, trainer._e1_incumbent_identity(pool, cfg))
    for costs in ({"unknown": 5.}, {"a": float("nan")}, {"a": -1.}):
        state["policy_best_objectives"] = costs
        with pytest.raises(ValueError, match="invalid"):
            trainer._validate_e1_resume_state(state, identity)


def test_exact_resume_signature_binds_training_objectives_experts_and_schedule(tmp_path):
    expert = tmp_path / "experts.csv"
    expert.write_text("instance_id,objective_distance_km\na,5\n")
    cfg = _cfg()
    cfg["training"].update(epochs=1500, gamma=.99, ppo_update_epochs=4,
        num_envs_per_gpu=32, n_traj=50, track_experiment_budget=True)
    cfg["offline"]["expert_solution_path"] = str(expert)
    signature = trainer._exact_experiment_resume_signature(cfg)
    runtime = copy.deepcopy(cfg)
    runtime["training"].update(stop_after_epoch=100, checkpoint_backup_dir=str(tmp_path / "new"))
    runtime["run_name"] = "relocated_backup_recovery"
    assert trainer._exact_experiment_resume_signature(runtime) == signature
    for key, value in (("gamma", 1.), ("ppo_update_epochs", 5), ("epochs", 1600)):
        changed = copy.deepcopy(cfg)
        changed["training"][key] = value
        assert trainer._exact_experiment_resume_signature(changed) != signature
    expert.write_text("instance_id,objective_distance_km\na,4\n")
    assert trainer._exact_experiment_resume_signature(cfg) != signature


def _real_config(output, fixture, *, method="slppo", epochs=2):
    preset = method_preset(method)
    cfg = yaml.safe_load((ROOT / "configs/templates/slppo_cus100_public.yaml").read_text())
    cfg["run_name"] = "E1_CPU_" + method
    cfg["data"].update(problem_type="cvrp", num_customers=4, num_charging_stations=0,
        train_dataset_path=str(fixture / "train/instances.pkl"))
    cfg["model"].update(embedding_dim=16, n_encode_layers=1)
    cfg["env"].update(use_jit_mask=False, reward_distance_scale_mode="single_customer_repair_median",
        reward_distance_scale_km=10., observation_distance_scale_km=10.)
    cfg["training"].update(epochs=epochs, num_envs_per_gpu=4, n_traj=3, rollout_steps=32,
        ppo_step_chunk_size=4, ppo_update_epochs=4, num_minibatches=4,
        checkpoint_interval=1, latest_checkpoint_interval=1, debug=False,
        mixed_precision=False, reward_norm_mode="legacy", track_experiment_budget=True,
        checkpoint_backup_dir=str(output / "backup"), monitor_interval=1,
        monitor_gradient_components=False, monitor_output_dir=str(output / "monitor"))
    cfg["offline"] = preset.offline_config()
    cfg["offline"].update(expert_dataset_path=str(fixture / "train/instances.pkl"),
        expert_solution_path=str(fixture / "train/expert_solutions.csv"),
        use_priority_sampler=False, policy_replay_enabled=False, branch_exploration_enabled=False,
        exploration_enabled=False, bc_batch_size=8, strict_replay=True)
    cfg["advantage"] = preset.advantage_config()
    if method == "slppo":
        cfg["offline"]["solution_reference_contract"] = trainer.E1_SOLUTION_REFERENCE_CONTRACT
        cfg["advantage"]["sl_expert_logprob_chunk_size"] = 4
    cfg["evaluation"].update(eval_interval=1, eval_path=str(fixture / "val/instances.pkl"),
        eval_n_traj=3, eval_max_steps=32, eval_batch_size=2, eval_save_routes=True,
        eval_output_dir=str(output / "validation"), gurobi_summary_path=None)
    return cfg


def _fixture(path):
    from test_aaai_core_training import _payload
    from test_joint_graph_training import _bundle
    rows = [_payload(f"train_{i}", "vrptw") for i in range(8)]
    _bundle(path / "train/instances.pkl", rows)
    _bundle(path / "val/instances.pkl", [_payload(f"val_{i}", "vrptw") for i in range(2)])
    with (path / "train/expert_solutions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "feasible", "objective_distance_km", "vehicle_count", "routes_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(instance_id=row["instance_id"], feasible=True, objective_distance_km=5.,
                                 vehicle_count=1, routes_json="[[0,1,2,3,4,0]]"))


def _launch(output, cfg, mode="full", world=1):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR=str(output / "numba"))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
        str(output), mode, str(world)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout[-6000:] + completed.stderr[-6000:]
    return json.loads((output / "result_0.json").read_text())


@pytest.mark.parametrize("pause_mode", ["cut", "pause"])
def test_real_slppo_backup_fresh_process_resume_matches_uninterrupted_cpu(tmp_path, pause_mode):
    fixture = tmp_path / "fixture"
    _fixture(fixture)
    full = tmp_path / "full"
    uninterrupted = _launch(full, _real_config(full, fixture))
    split = tmp_path / "split"
    cfg = _real_config(split, fixture)
    if pause_mode == "pause":
        cfg["training"]["stop_after_epoch"] = 1
    interrupted = _launch(split, cfg, "cut" if pause_mode == "cut" else "full")
    if pause_mode == "pause":
        assert Path(interrupted["checkpoint"]).name == "checkpoint_epoch_0001.pt"
        assert not Path(interrupted["checkpoint"]).with_name("checkpoint_final.pt").exists()
    # Resume exclusively from independently copied bytes, with no best.json or
    # original results folder available in this new process/output directory.
    restored = tmp_path / "restored"
    resume_cfg = _real_config(restored, fixture)
    resume_cfg["training"]["resume_checkpoint_path"] = interrupted["backup"]
    resumed = _launch(restored, resume_cfg)
    assert resumed["rollout_calls"] == 1 and uninterrupted["rollout_calls"] == 2
    assert resumed["parameter_sha256"] == uninterrupted["parameter_sha256"]
    assert resumed["budget"] == uninterrupted["budget"]
    assert resumed["incumbents"] == uninterrupted["incumbents"]
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"] == 32
    assert resumed["best_selection"] == uninterrupted["best_selection"]
    assert resumed["budget"]["instance_exposures"] == 8
    assert resumed["budget"]["sampled_trajectories"] == 24
    assert resumed["budget"]["expert_backward_steps"] > 0
    assert resumed["maximum_unchanged_logprob_error"] < 2e-6
    assert resumed["maximum_online_sl_advantage"] > 0
    assert resumed["replay_pool_is_none"] and resumed["has_identity"]


def test_real_dual_gloo_cost_incumbents_parameters_and_budget_are_synchronized(tmp_path):
    fixture = tmp_path / "fixture"
    _fixture(fixture)
    output = tmp_path / "dual"
    first = _launch(output, _real_config(output, fixture, epochs=1), world=2)
    second = json.loads((output / "result_1.json").read_text())
    for key in ("parameter_sha256", "incumbents", "budget", "optimizer_steps"):
        assert first[key] == second[key]
    assert first["budget"]["instance_exposures"] == 8
    assert first["budget"]["optimizer_attempts"] == first["optimizer_steps"] == 16
    assert first["replay_pool_is_none"] and second["replay_pool_is_none"]


def _worker(rank, output, mode, world):
    import torch.distributed as dist
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("gloo", init_method="file://" + str(output / "gloo"), rank=rank, world_size=world)
    trainer.REPO_ROOT = output
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    errors = []
    online_advantages = []
    rollout_calls = []
    captured = {}
    original_agent = trainer.Agent
    class CapturedAgent(original_agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["agent"] = self
    trainer.Agent = CapturedAgent
    original_advantages = trainer._solution_level_advantage_tensors
    def checked_advantages(*args, **kwargs):
        result = original_advantages(*args, **kwargs)
        online_advantages.append(float(result[0].abs().sum()))
        return result
    trainer._solution_level_advantage_tensors = checked_advantages
    original_collect = trainer.collect_rollout
    def checked_collect(agent, *args, **kwargs):
        batch = original_collect(agent, *args, **kwargs)
        rollout_calls.append(1)
        with torch.no_grad():
            replay = trainer._policy_chunk_evaluations(agent, batch, np.arange(batch.actions.shape[1]), 0, batch.actions.shape[0])
            logprob = torch.stack([step[0] for step in replay])
            errors.append(float((logprob[batch.valid] - batch.old_logprobs[batch.valid]).abs().max()))
        return batch
    trainer.collect_rollout = checked_collect
    original_candidates = trainer._prepare_sl_expert_candidates
    def checked_candidates(agent, batch, config, envs, expert_buffer, incumbents, device):
        candidates, info = original_candidates(agent, batch, config, envs, expert_buffer, incumbents, device)
        if candidates:
            with torch.no_grad():
                replay = trainer._expert_route_mean_logprobs(agent, candidates, device,
                    int(config.get("advantage", {}).get("sl_expert_logprob_chunk_size", 4096)),
                    cache_static=bool(config.get("training", {}).get("cache_expert_route_encoding", False)))
                old = torch.tensor([candidate.old_mean_logprob for candidate in candidates])
                errors.append(float((replay.cpu() - old).abs().max()))
        return candidates, info
    trainer._prepare_sl_expert_candidates = checked_candidates
    original_save = trainer.save_checkpoint
    class StopAfterRecovery(Exception):
        pass
    def stop_after_recovery(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if mode == "cut" and path.name == "checkpoint_latest.pt":
            raise StopAfterRecovery(path)
    trainer.save_checkpoint = stop_after_recovery
    try:
        try:
            checkpoint = trainer.train_from_config(cfg, seed=31, device="cpu")
        except StopAfterRecovery as exc:
            checkpoint = exc.args[0]
        if world > 1:
            dist.barrier()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resume = state["training_resume_state"]
        digest = hashlib.sha256()
        for name, tensor in trainer.inference_model_state(captured["agent"]).items():
            digest.update(name.encode())
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        backup = Path(cfg["training"]["checkpoint_backup_dir"]) / cfg["run_name"] / "seed_31" / checkpoint.name
        record = dict(checkpoint=str(checkpoint), backup=str(backup), parameter_sha256=digest.hexdigest(),
            budget=resume["experiment_budget"], incumbents=resume["ranks"][rank]["policy_best_objectives"],
            optimizer_steps=resume["ranks"][rank]["optimizer_steps"],
            best_selection=resume["best_validation_selection"],
            replay_pool_is_none=resume["ranks"][rank]["policy_route_pool"] is None,
            has_identity="policy_incumbent_identity" in resume["ranks"][rank],
            maximum_unchanged_logprob_error=max(errors, default=0.),
            maximum_online_sl_advantage=max(online_advantages, default=0.), rollout_calls=len(rollout_calls))
        (output / f"result_{rank}.json").write_text(json.dumps(record))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__" and sys.argv[1] == "--worker":
    output, mode, world = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    if world > 1:
        import torch.multiprocessing as mp
        mp.spawn(_worker, args=(output, mode, world), nprocs=world, join=True)
    else:
        _worker(0, output, mode, world)
