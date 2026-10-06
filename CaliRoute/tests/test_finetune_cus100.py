"""Protect train/validation isolation and the Cus100 comparison protocol."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from offline2online.training_schedule import schedule_for_epoch


_path = Path(__file__).resolve().parents[1] / "scripts/finetune_cus100_config.py"
_spec = importlib.util.spec_from_file_location("finetune_cus100_config", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_config = _module.build_config


def config(tmp_path, problem="cvrp", phase="control", **execution_settings):
    return build_config(
        problem=problem, phase=phase, run_name=f"{problem}_{phase}",
        output_dir=tmp_path / problem / phase, data_root=tmp_path / "dataset",
        init_checkpoint=None if phase == "ppo_init" else tmp_path / problem / "ppo_init.pt",
        **execution_settings,
    )


@pytest.mark.parametrize("problem", ["cvrp", "vrptw"])
def test_ppo_init_does_not_load_experts_or_sl_objectives(tmp_path, problem):
    from offline2online.trainer import (
        _group_advantage_enabled, _init_policy_route_pool, _load_expert_buffer,
        _reference_advantage_enabled, _sl_candidate_enabled,
    )

    cfg = config(tmp_path, problem, "ppo_init")
    assert cfg["offline"]["method"] == "ppo"
    assert not _group_advantage_enabled(cfg)
    assert not _reference_advantage_enabled(cfg)
    assert not _sl_candidate_enabled(cfg)
    # No dataset or checkpoint exists in tmp_path: successful early returns
    # demonstrate that these real trainer entry points do not touch experts.
    assert _load_expert_buffer(cfg, seed=3009, debug_enabled=False, debug_file=None) is None
    assert _init_policy_route_pool(cfg, train_pool=None) is None
    assert not cfg["offline"]["use_priority_sampler"]
    assert cfg["offline"]["sl_coef"] == 0
    assert not any(key.endswith("_path") for key in cfg["offline"])
    assert cfg["training"]["ppo_update_epochs"] == 3
    assert cfg["training"]["epochs"] == 100
    assert cfg["model"]["use_rdi_v2"] and cfg["model"]["use_agda_v2"]


@pytest.mark.parametrize("problem", ["cvrp", "vrptw"])
@pytest.mark.parametrize("phase", ["ppo_init", "control", "candidate", "long_control", "long_candidate"])
def test_only_task_matched_train_and_full_validation_are_configured(tmp_path, problem, phase):
    cfg = config(tmp_path, problem, phase)
    train = tmp_path / "dataset" / problem / "train" / "Cus100"
    val = tmp_path / "dataset" / problem / "val" / "Cus100"
    assert cfg["data"]["train_dataset_path"] == str(train)
    assert cfg["data"]["problem_type"] == problem
    assert cfg["data"]["num_customers"] == 100
    assert cfg["data"]["num_charging_stations"] == 0
    evaluation = cfg["evaluation"]
    assert evaluation["eval_path"] == str(val)
    assert evaluation["gurobi_summary_path"] == str(val / "gurobi_summary.csv")
    assert "eval_limit" not in evaluation and "eval_num_batches" not in evaluation
    assert evaluation["eval_n_traj"] == 50
    assert evaluation["eval_batch_size"] == 32
    assert evaluation["eval_interval"] == 20
    assert evaluation["eval_seed"] == 17003009
    assert evaluation["eval_before_training"] and evaluation["eval_save_routes"]
    assert evaluation["eval_max_steps"] == cfg["training"]["rollout_steps"] == 201
    assert cfg["training"]["ppo_step_chunk_size"] == 8
    assert cfg["experiment_protocol"]["ppo_step_chunk_size"] == 8
    assert cfg["experiment_protocol"]["eval_batch_size"] == 32
    if phase != "ppo_init":
        assert cfg["offline"]["expert_dataset_path"] == str(train)
        assert cfg["offline"]["expert_solution_path"] == str(train / "expert_solutions.csv")
        assert cfg["offline"]["init_checkpoint_strict"] is True
        assert "resume_checkpoint_path" not in cfg["offline"]
    assert evaluation["eval_output_dir"] == str(tmp_path / problem / phase / "evaluations")
    assert cfg["training"]["monitor_output_dir"] == str(tmp_path / problem / phase / "monitoring")


@pytest.mark.parametrize("problem", ["cvrp", "vrptw"])
@pytest.mark.parametrize("phase", ["ppo_init", "control", "candidate", "long_control", "long_candidate"])
def test_memory_settings_apply_to_every_phase_without_changing_algorithm(tmp_path, problem, phase):
    original = config(tmp_path, problem, phase)
    expanded = config(tmp_path, problem, phase, ppo_step_chunk_size=32, eval_batch_size=64)
    assert expanded["training"]["ppo_step_chunk_size"] == 32
    assert expanded["evaluation"]["eval_batch_size"] == 64
    assert expanded["experiment_protocol"]["ppo_step_chunk_size"] == 32
    assert expanded["experiment_protocol"]["eval_batch_size"] == 64
    for cfg in (original, expanded):
        assert cfg["experiment_protocol"]["global_instances_per_rollout"] == 64
        assert cfg["experiment_protocol"]["global_trajectories_per_rollout"] == 3200
        assert cfg["training"]["num_envs_per_gpu"] == 32
        assert cfg["training"]["num_minibatches"] == 4
        assert cfg["training"]["n_traj"] == 50
        assert cfg["training"]["learning_rate"] == 5e-5
        cfg["training"].pop("ppo_step_chunk_size")
        cfg["evaluation"].pop("eval_batch_size")
        cfg["experiment_protocol"].pop("ppo_step_chunk_size")
        cfg["experiment_protocol"].pop("eval_batch_size")
    # Includes update passes, SL coefficients, all schedule settings and model
    # flags, so memory tuning cannot silently alter the experiment budget.
    assert original == expanded


@pytest.mark.parametrize("chunk", [1, 201])
def test_chunk_boundary_values_are_accepted(tmp_path, chunk):
    cfg = config(tmp_path, ppo_step_chunk_size=chunk, eval_batch_size=1)
    assert cfg["training"]["ppo_step_chunk_size"] == chunk
    assert cfg["evaluation"]["eval_batch_size"] == 1


@pytest.mark.parametrize("key,value", [
    ("ppo_step_chunk_size", value) for value in (False, True, 0, -1, 202, 8.0, "8", None)
] + [
    ("eval_batch_size", value) for value in (False, True, 0, -1, 32.0, "32", None)
])
def test_execution_settings_require_bounded_positive_integers(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        config(tmp_path, **{key: value})


@pytest.mark.parametrize("prefix", ["", "long_"])
def test_control_candidate_have_only_requested_training_differences(tmp_path, prefix):
    control = config(tmp_path, phase=f"{prefix}control")
    candidate = config(tmp_path, phase=f"{prefix}candidate")
    assert control["model"] == candidate["model"] == config(tmp_path, phase="ppo_init")["model"]
    assert control["offline"]["init_checkpoint_path"] == candidate["offline"]["init_checkpoint_path"]
    assert control["training"]["ppo_update_epochs"] == 4
    assert candidate["training"]["ppo_update_epochs"] == 3
    assert control["offline"]["sl_coef"] == .5
    assert candidate["offline"]["sl_coef"] == .35
    for cfg in (control, candidate):
        assert cfg["offline"]["policy_replay_enabled"]
        assert cfg["offline"]["policy_replay_weight"] == .1
        assert cfg["offline"]["policy_replay_warmup_epochs"] == 25
        assert cfg["offline"]["policy_replay_ramp_epochs"] == 75
        assert cfg["training"]["num_envs_per_gpu"] == 32
        assert cfg["training"]["num_minibatches"] == 4
        assert cfg["training"]["n_traj"] == 50
        assert cfg["training"]["latest_checkpoint_interval"] == 5
        assert cfg["training"]["monitor_interval"] == 10
        assert cfg["training"]["monitor_gradient_components"]
        assert cfg["training"]["amp_init_scale"] == 4096
    left, right = deepcopy(control), deepcopy(candidate)
    for cfg in (left, right):
        cfg.pop("run_name")
        cfg["experiment_protocol"].pop("phase")
        cfg["training"].pop("monitor_output_dir")
        cfg["training"].pop("ppo_update_epochs")
        cfg["evaluation"].pop("eval_output_dir")
        cfg["offline"].pop("sl_coef")
    assert left == right


@pytest.mark.parametrize("phase", ["control", "candidate"])
def test_screen_is_explicitly_a_constant_schedule_local_probe(tmp_path, phase):
    cfg = config(tmp_path, phase=phase)
    assert cfg["training"]["epochs"] == cfg["training"]["checkpoint_interval"] == 40
    assert "local_probe" in cfg["experiment_protocol"]["schedule_scope"]
    for epoch in (1, 5, 20, 40):
        assert schedule_for_epoch(cfg, epoch) == {"learning_rate": 5e-5, "ent_coef": .01}


@pytest.mark.parametrize("phase,epochs,warmup", [("ppo_init", 100, 10), ("long_control", 1000, 20), ("long_candidate", 1000, 20)])
def test_full_runs_have_the_requested_schedule_endpoints(tmp_path, phase, epochs, warmup):
    cfg = config(tmp_path, phase=phase)
    assert cfg["training"]["epochs"] == epochs
    assert cfg["training"]["lr_warmup_epochs"] == warmup
    assert cfg["training"]["checkpoint_interval"] == 50
    assert schedule_for_epoch(cfg, 1)["learning_rate"] == pytest.approx(5e-5 / warmup)
    assert schedule_for_epoch(cfg, warmup)["learning_rate"] == 5e-5
    assert schedule_for_epoch(cfg, epochs) == {"learning_rate": 1e-5, "ent_coef": .002}


@pytest.mark.parametrize("kwargs", [
    {"problem": "evrptw"}, {"phase": "unknown"},
    {"data_root": "/data/AAAI_Dataset/test_release"},
    {"data_root": "/data/AAAI_Dataset/test"},
    {"phase": "control", "init_checkpoint": None},
    {"phase": "ppo_init", "init_checkpoint": "/data/existing.pt"},
])
def test_invalid_or_test_referencing_protocol_is_rejected(tmp_path, kwargs):
    args = dict(problem="cvrp", phase="control", run_name="example", output_dir=tmp_path / "outputs",
                data_root=tmp_path / "dataset", init_checkpoint=tmp_path / "init.pt")
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_config(**args)


def test_builds_are_independent_and_do_not_create_files(tmp_path):
    a = config(tmp_path)
    a["model"]["use_rdi_v2"] = False
    a["advantage"]["sl_relative_std_floor"] = 100
    b = config(tmp_path)
    assert b["model"]["use_rdi_v2"]
    assert b["advantage"]["sl_relative_std_floor"] == .01
    assert list(tmp_path.iterdir()) == []
