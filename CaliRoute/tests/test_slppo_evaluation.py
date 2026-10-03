from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from caliroute.cli import parse_args
from caliroute.config import build_training_config
from offline2online import trainer


def _eval_row(fr, distance):
    return {"eval_status": "ok", "eval_feasible_rate": fr, "eval_avg_objective_distance_km": distance}


def test_checkpoint_selection_prioritizes_feasibility():
    assert not trainer._is_better_eval_result(_eval_row(0.5, 1), 1.0, 100)
    assert trainer._is_better_eval_result(_eval_row(1.0, 100), 0.5, 1)
    assert trainer._is_better_eval_result(_eval_row(1.0, 90), 1.0, 100)
    assert not trainer._is_better_eval_result(_eval_row(0.0, 1), -1.0, float("inf"))
    assert not trainer._is_better_eval_result(_eval_row(1.0, float("nan")), -1.0, float("inf"))


def test_failed_partial_route_is_excluded_from_reference_gap():
    failed = trainer._select_min_median_trajectory_stats({
        "objective_distance_km": [1.0], "success": [False],
        "served_customers": [1], "vehicle_count": [1],
    })
    failed["instance_id"] = "partial"
    success = dict(failed, instance_id="complete", feasible=True, objective_distance_km=12.0)
    refs = {i: {"objective_distance_km": 10.0, "vehicle_count": 1} for i in ("partial", "complete")}
    assert trainer._tail_gap_stats([failed, success], refs)["eval_gap_mean"] == 2.0
    assert np.isnan(trainer._tail_gap_stats([failed], refs)["eval_gap_mean"])


def test_eval_rng_repeats_without_consuming_training_randomness():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    draws = []
    for _ in range(2):
        with trainer._isolated_eval_rng(901, "cpu"):
            draws.append((random.random(), float(np.random.random()), torch.rand(3)))
    assert draws[0][:2] == draws[1][:2]
    assert torch.equal(draws[0][2], draws[1][2])
    actual = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_eval_rng_restores_after_exception():
    torch.manual_seed(99)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with trainer._isolated_eval_rng(100, "cpu"):
            torch.rand(5)
            raise RuntimeError("evaluation failed")
    assert torch.equal(torch.random.get_rng_state(), state)


def test_eval_exports_selected_routes_and_uses_epoch_independent_seed(tmp_path, monkeypatch):
    instance = SimpleNamespace(
        instance_id="test_instance", num_customers=2, demands_cm3=np.array([1.0, 1.0]),
        distance_matrix_km=np.array([[0., 4., 3.], [4., 0., 3.], [3., 3., 0.]]),
        vehicle={"cargo_capacity_cm3": 2.0},
    )
    monkeypatch.setattr(trainer, "_eval_instance_batches", lambda *args, **kwargs: iter([[instance]]))
    monkeypatch.setattr(trainer, "make_terran_env", lambda **kwargs: SimpleNamespace())
    seeds = []
    def rollout(*args, seed, **kwargs):
        seeds.append(seed)
        draw = float(torch.rand(()))
        return [trainer._select_min_median_trajectory_stats({
            "objective_distance_km": [20 + draw, 10.0], "success": [True, True],
            "served_customers": [2, 2], "vehicle_count": [1, 1], "invalid_action": [False, False],
            "routes": [[[0, 1, 2, 0]], [[0, 2, 1, 0]]],
            "route_sequence": [[0, 1, 2, 0], [0, 2, 1, 0]],
        }) | {"runtime_s": 0.1}]
    monkeypatch.setattr(trainer, "_rollout_eval_batch_min_median", rollout)
    cfg = {
        "data": {"problem_type": "cvrp", "num_customers": 2},
        "evaluation": {"eval_path": str(tmp_path), "eval_output_dir": str(tmp_path / "export"),
                       "eval_save_routes": True, "eval_seed": 42},
    }
    agent = torch.nn.Linear(1, 1)
    first = trainer.evaluate_fixed_dataset(agent, cfg, seed=3009, epoch=1, device="cpu")
    second = trainer.evaluate_fixed_dataset(agent, cfg, seed=3009, epoch=20, device="cpu")
    assert first["eval_avg_objective_distance_km"] == second["eval_avg_objective_distance_km"]
    assert seeds == [42, 42]
    assert agent.training
    row = json.loads((tmp_path / "export" / "epoch_0020.jsonl").read_text())
    assert row["instance_id"] == "test_instance"
    assert row["feasible"] is True
    assert row["routes"] == [[0, 2, 1, 0]]
    assert row["selected_trajectory_index"] == 1
    assert row["route_validation"]["valid"] is True
    assert row["route_validation"]["recomputed_distance_km"] == 10.0
    assert row["reference_objective_distance_km"] is None
    assert row["relative_gap_pct"] is None


def _agent_with_new_module():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.legacy = torch.nn.Linear(2, 2)
    model.backbone.residual_edge_bias = torch.nn.Linear(2, 1)
    return model


def test_initial_checkpoint_only_allows_missing_new_adapter_keys():
    model = _agent_with_new_module()
    state = {k: v.clone() for k, v in model.state_dict().items() if ".residual_edge_bias." not in k}
    result = trainer._load_initial_model_state(model, state, strict=False)
    assert result.missing_keys == ["backbone.residual_edge_bias.weight", "backbone.residual_edge_bias.bias"]
    with pytest.raises(RuntimeError, match="Incompatible initialization"):
        trainer._load_initial_model_state(model, {k: v for k, v in state.items() if not k.endswith("legacy.bias")}, strict=False)
    with pytest.raises(RuntimeError, match="Incompatible initialization"):
        trainer._load_initial_model_state(model, dict(state, unknown=torch.ones(1)), strict=False)
    with pytest.raises(RuntimeError):
        trainer._load_initial_model_state(model, state, strict=True)


def test_initial_checkpoint_rejects_shape_mismatch_even_for_new_modules():
    model = _agent_with_new_module()
    state = dict(model.state_dict())
    state["backbone.residual_edge_bias.weight"] = torch.ones(5, 5)
    with pytest.raises(RuntimeError, match="size mismatch"):
        trainer._load_initial_model_state(model, state, strict=False)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.6])
def test_public_expert_weight_updates_loss_and_advantage(tmp_path, monkeypatch, weight):
    expert = tmp_path / "expert_solutions.csv"
    expert.write_text("instance_id\n")
    monkeypatch.setattr("sys.argv", ["train.py",
        "--problem", "cvrp", "--customers", "50", "--offline-method", "slppo",
        "--expert-solution", str(expert), "--sl-expert-candidate-weight", str(weight),
    ])
    cfg = build_training_config(parse_args())
    assert cfg["offline"]["sl_expert_candidate_weight"] == weight
    assert cfg["advantage"]["sl_expert_candidate_weight"] == weight


def _cvrp_validation_instance():
    return SimpleNamespace(
        num_customers=3, demands_cm3=np.array([1.0, 1.0, 1.0]),
        distance_matrix_km=np.array([[0., 1., 1., 1.], [1., 0., 1., 1.],
                                     [1., 1., 0., 1.], [1., 1., 1., 0.]]),
        vehicle={"cargo_capacity_cm3": 2.0},
    )


def test_independent_cvrp_route_validation_accepts_complete_capacitated_solution():
    result = trainer._validate_cvrp_eval_route(
        _cvrp_validation_instance(), {"routes": [[0, 1, 2, 0], [0, 3, 0]], "objective_distance_km": 5.0},
    )
    assert result["valid"]
    assert result["route_loads_cm3"] == [2.0, 1.0]


@pytest.mark.parametrize("routes,objective,failed_check", [
    ([[0, 1, 1, 0], [0, 3, 0]], 4.0, "customer_coverage_valid"),
    ([[0, 1, 2, 3, 0]], 4.0, "capacity_valid"),
    ([[0, 1, 2], [0, 3, 0]], 4.0, "depot_endpoints_valid"),
    ([[0, 1, 9, 0], [0, 2, 3, 0]], 5.0, "indices_valid"),
    ([[0, 1, 2, 0], [0, 3, 0]], 4.0, "distance_matches"),
])
def test_independent_cvrp_route_validation_rejects_invalid_solutions(routes, objective, failed_check):
    result = trainer._validate_cvrp_eval_route(
        _cvrp_validation_instance(), {"routes": routes, "objective_distance_km": objective},
    )
    assert not result["valid"]
    assert not result[failed_check]


def test_checkpoint_route_pool_is_restored_only_for_resume(tmp_path):
    agent = torch.nn.Linear(2, 1)
    agent.policy_route_pool = SimpleNamespace(state_dict=lambda: {"version": 1, "routes": [0, 1, 0]})
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, agent, optimizer, {}, epoch=12, seed=3009)
    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-5)
    info = trainer._load_training_checkpoint(resumed, resumed_optimizer, path, "cpu")
    assert info["epoch"] == 12
    assert info["optimizer_loaded"]
    assert resumed._pending_policy_route_pool_state == {"version": 1, "routes": [0, 1, 0]}
    finetuned = torch.nn.Linear(2, 1)
    trainer._load_agent_checkpoint(finetuned, path, "cpu")
    assert not hasattr(finetuned, "_pending_policy_route_pool_state")
    for expected, actual in zip(agent.parameters(), finetuned.parameters()):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("enabled,resume,expected_calls", [
    (False, None, 0), (True, "checkpoint.pt", 0), (True, None, 1),
])
def test_initial_evaluation_is_optional_epoch_zero_and_skipped_on_resume(monkeypatch, enabled, resume, expected_calls):
    calls = []
    expected = _eval_row(1.0, 10.0)
    def evaluate(agent, cfg, seed, epoch, device):
        calls.append((seed, epoch, device))
        return expected
    monkeypatch.setattr(trainer, "evaluate_fixed_dataset", evaluate)
    result = trainer._evaluate_before_training(
        torch.nn.Linear(1, 1), {"evaluation": {"eval_before_training": enabled}},
        3009, "cpu", resume_checkpoint_path=resume,
    )
    assert len(calls) == expected_calls
    if expected_calls:
        assert calls == [(3009, 0, "cpu")]
        assert result is expected
    else:
        assert result is None
