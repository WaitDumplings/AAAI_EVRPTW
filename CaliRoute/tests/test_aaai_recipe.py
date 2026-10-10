"""Canonical recipes preserve the recorded algorithm, independent of storage/GPU shape.

These are configuration checks only. They must never probe CUDA, create runs,
read dataset payloads, or silently turn unsupported tasks into validated recipes.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from caliroute.recipes import build_recipe_config


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads(
    (ROOT / "docs/experiments/graph_rdi100_20261008_provenance.json").read_text()
)


def _reference(problem, encoder):
    batch = 40 if problem == "vrptw" else 32
    name = f"{encoder.upper()}_{problem.upper()}100_S3011_B{batch}_E1500_20261008_r2"
    return deepcopy(PROVENANCE["runs"][name]["effective_config"])


def _build(tmp_path, *, problem="vrptw", encoder="graph", **overrides):
    args = dict(
        problem=problem, customers=100, encoder=encoder, seed=3011, epochs=1500,
        data_root=tmp_path / "dataset_storage", output_dir=tmp_path / "outputs",
        run_name="RECIPE_CONFIG_TEST", world_size=1, hardware="rtx48_single",
    )
    args.update(overrides)
    return build_recipe_config(**args)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            path: leaf
            for key, child in value.items()
            for path, leaf in _flatten(child, f"{prefix}.{key}" if prefix else key).items()
        }
    return {prefix: value}


# Compare every effective source field, rather than asserting a small favorable
# subset. Metadata may explain the canonical API, but cannot hide training drift.
PATH_FIELDS = {
    "run_name", "data.train_dataset_path", "offline.expert_dataset_path",
    "offline.expert_solution_path", "evaluation.eval_path",
    "evaluation.gurobi_summary_path", "evaluation.eval_output_dir",
    "training.monitor_output_dir",
}
HARDWARE_FIELDS = {
    "training.num_envs_per_gpu", "training.ppo_step_chunk_size",
    "offline.sl_expert_logprob_chunk_size", "advantage.sl_expert_logprob_chunk_size",
    "offline.exploration_instances", "offline.policy_replay_max_new_routes",
}


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
@pytest.mark.parametrize("encoder", ["graph", "current"])
@pytest.mark.parametrize("hardware,world_size", [("rtx48_single", 1), ("2080ti_dual", 2)])
def test_entire_algorithm_matches_recorded_effective_config(tmp_path, problem, encoder, hardware, world_size):
    cfg = _build(tmp_path, problem=problem, encoder=encoder, hardware=hardware, world_size=world_size)
    expected = _flatten(_reference(problem, encoder))
    actual = _flatten(cfg)
    permitted = PATH_FIELDS | HARDWARE_FIELDS
    missing = object()
    drift = {
        key: (expected.get(key, missing), actual.get(key, missing))
        for key in expected.keys() | actual.keys()
        if not key.startswith("experiment_protocol.")
        and key not in permitted
        and expected.get(key, missing) != actual.get(key, missing)
    }
    assert not drift, drift
    # Configuration construction must work before datasets have been downloaded.
    assert not (tmp_path / "dataset_storage").exists()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("problem,global_batch", [("vrptw", 40), ("evrptw", 32)])
@pytest.mark.parametrize("hardware,world_size,chunk,expert_chunk", [
    ("rtx48_single", 1, 120, 128), ("2080ti_dual", 2, 48, 64),
])
def test_default_main_search_and_archive_budgets_are_global(tmp_path, problem, global_batch, hardware, world_size, chunk, expert_chunk):
    cfg = _build(tmp_path, problem=problem, hardware=hardware, world_size=world_size)
    train, offline = cfg["training"], cfg["offline"]
    assert train["num_envs_per_gpu"] * world_size == global_batch
    assert train["n_traj"] == 50
    assert train["ppo_update_epochs"] == 5 and train["num_minibatches"] == 4
    assert train["gradient_accumulation_steps"] == 1
    assert train["ppo_step_chunk_size"] == chunk
    assert offline["sl_expert_logprob_chunk_size"] == expert_chunk
    assert cfg["advantage"]["sl_expert_logprob_chunk_size"] == expert_chunk
    assert train["learning_rate"] == 1e-4 and train["target_kl"] is None
    assert train.get("ppo_warmup_epochs", 0) == 0 and offline["method"] == "sl_ppo"
    assert offline["exploration_instances"] * world_size == 8
    assert offline["exploration_trajectories"] == 8
    assert offline["exploration_interval"] == 5
    assert offline["policy_replay_max_new_routes"] * world_size == 32
    assert offline["policy_replay_capacity"] == 3
    assert offline["policy_replay_exploration_capacity"] == 4
    assert offline["policy_replay_fraction"] == .25
    assert min(offline["policy_replay_max_candidates"], train["num_envs_per_gpu"] // 4) * world_size == global_batch // 4
    evaluation = cfg["evaluation"]
    assert evaluation["eval_batch_size"] == 16 and evaluation["eval_n_traj"] == 50
    assert evaluation["eval_interval"] == 50 and evaluation["eval_before_training"]
    assert evaluation.get("eval_limit") is None and evaluation.get("eval_num_batches") is None
    protocol = cfg["experiment_protocol"]
    assert protocol["world_size"] == world_size
    assert protocol["global_instances_per_rollout"] == global_batch
    assert protocol["global_trajectories_per_rollout"] == global_batch * 50
    assert protocol["global_instances_per_optimizer_step"] == global_batch // 4


@pytest.mark.parametrize("problem", ["vrptw", "evrptw"])
def test_current_control_changes_exactly_the_recorded_encoder_fields(tmp_path, problem):
    graph = _build(tmp_path, problem=problem)
    current = _build(tmp_path, problem=problem, encoder="current")
    for section in ("data", "env", "training", "offline", "advantage", "evaluation", "pbrs", "critic"):
        assert graph[section] == current[section], section
    changed = {
        key for key in graph["model"].keys() | current["model"].keys()
        if graph["model"].get(key) != current["model"].get(key)
    }
    assert changed == {
        "use_joint_graph_encoder", "use_edge_relation_encoder",
        "joint_graph_edge_dim", "joint_graph_dropout",
    }


@pytest.mark.parametrize("seed", [0, 3009, 3010, 3012])
def test_seed_override_changes_rng_settings_without_changing_algorithm(tmp_path, seed):
    original = _build(tmp_path)
    actual = _build(tmp_path, seed=seed)
    assert actual["training"]["post_init_seed"] == seed
    assert actual["evaluation"]["eval_seed"] == 17_000_000 + seed
    before, after = _flatten(original), _flatten(actual)
    permitted = {"training.post_init_seed", "evaluation.eval_seed"}
    drift = {
        key: (before.get(key), after.get(key))
        for key in before.keys() | after.keys()
        if not key.startswith("experiment_protocol.") and key not in permitted
        and before.get(key) != after.get(key)
    }
    assert not drift, drift


def test_explicit_memory_overrides_preserve_losses_and_actual_update_budget(tmp_path):
    reference = _build(tmp_path, hardware="2080ti_dual", world_size=2)
    cfg = _build(tmp_path, hardware="2080ti_dual", world_size=2,
                 ppo_chunk_size=24, expert_chunk_size=32, eval_batch_size=8)
    before, after = _flatten(reference), _flatten(cfg)
    permitted = {
        "training.ppo_step_chunk_size", "offline.sl_expert_logprob_chunk_size",
        "advantage.sl_expert_logprob_chunk_size", "evaluation.eval_batch_size",
    }
    assert cfg["training"]["ppo_step_chunk_size"] == 24
    assert cfg["offline"]["sl_expert_logprob_chunk_size"] == 32
    assert cfg["evaluation"]["eval_batch_size"] == 8
    drift = {
        key: (before.get(key), after.get(key))
        for key in before.keys() | after.keys()
        if not key.startswith("experiment_protocol.") and key not in permitted
        and before.get(key) != after.get(key)
    }
    assert not drift, drift


@pytest.mark.parametrize("global_batch", [32, 40, 48, 64])
def test_explicit_global_batch_has_same_candidate_search_and_intake_budget_on_either_topology(tmp_path, global_batch):
    single = _build(tmp_path, global_batch=global_batch)
    dual = _build(tmp_path, global_batch=global_batch, hardware="2080ti_dual", world_size=2)
    budgets = []
    for cfg, world in [(single, 1), (dual, 2)]:
        train, offline = cfg["training"], cfg["offline"]
        assert train["num_envs_per_gpu"] * world == global_batch
        assert train["learning_rate"] == 1e-4
        assert offline["policy_replay_max_candidates"] == 16
        budgets.append((
            world * min(offline["policy_replay_max_candidates"], train["num_envs_per_gpu"] // 4),
            world * offline["policy_replay_max_new_routes"],
            world * offline["exploration_instances"] * offline["exploration_trajectories"],
        ))
    assert budgets[0] == budgets[1] == (global_batch // 4, 32, 64)


@pytest.mark.parametrize("overrides", [
    {"problem": "cvrp"}, {"problem": "tsp"}, {"customers": 15}, {"customers": 50},
    {"customers": 1000}, {"encoder": "legacy"}, {"hardware": "unknown"},
    {"hardware": "rtx48_single", "world_size": 2},
    {"hardware": "2080ti_dual", "world_size": 1},
    {"world_size": 0}, {"world_size": True}, {"global_batch": 0},
    {"global_batch": 33}, {"global_batch": 128}, {"seed": -1}, {"epochs": 0},
    {"ppo_chunk_size": 0}, {"ppo_chunk_size": 202},
    {"expert_chunk_size": 0}, {"eval_batch_size": 0},
])
def test_invalid_or_unvalidated_recipe_requests_fail_explicitly(tmp_path, overrides):
    with pytest.raises(ValueError):
        _build(tmp_path, **overrides)


def test_returned_config_is_independent_and_has_no_implicit_checkpoint(tmp_path):
    first = _build(tmp_path)
    second = _build(tmp_path)
    first["model"]["embedding_dim"] = -1
    first["offline"]["exploration_prefix_fractions"].append(.99)
    assert second["model"]["embedding_dim"] == 256
    assert second["offline"]["exploration_prefix_fractions"] == [0., .1, .25, .5]
    flat = _flatten(second)
    for key, value in flat.items():
        if key.endswith(("init_checkpoint", "resume_checkpoint", "reference_checkpoint")):
            assert not value, (key, value)
