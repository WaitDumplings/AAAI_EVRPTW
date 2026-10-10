"""Real CPU/Gloo training for the slim core, including expert/archive losses.

This smoke test uses four-customer fixtures and small embeddings/trajectories.
It preserves five PPO passes and four minibatches (20 optimizer attempts), but
sets replay warmup/ramp to zero and seeds *independently verified* archive routes
to exercise replay immediately. Those are test-only changes, not recipe defaults.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PREFIX = "backbone.joint_graph_encoder."


def _payload(identity, task):
    size = 6 if task == "evrptw" else 5
    distance = np.full((size, size), 2.8, dtype=np.float32)
    np.fill_diagonal(distance, 0.)
    for start, end in zip([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]):
        distance[start, end] = 1.
    if task == "evrptw":
        distance[5, :5] = distance[:5, 5] = 1.8
    travel = distance.copy()
    travel[1, 2] += .3
    energy = distance * .5
    energy[2, 0] += .2
    return dict(instance_id=identity, working_start_s=0, working_end_s=100,
        depot=np.array([0., 0.]), customers=np.array([[.01, 0.], [.02, 0.], [.02, .01], [0., .01]]),
        charging_stations=np.array([[.01, .01]]) if task == "evrptw" else np.empty((0, 2)),
        distance_matrix_km=distance, travel_time_matrix_s=travel, energy_matrix_kwh=energy,
        demands_cm3=np.ones(4), package_counts=np.ones(4, dtype=np.int32),
        service_time_s=np.ones(4), tw_s=np.array([[0., 100.]] * 4),
        cs_time_to_depot_s=np.array([1.8]) if task == "evrptw" else np.empty(0),
        vehicle=dict(cargo_capacity_cm3=4., battery_capacity_kwh=10.,
                     consumption_kwh_per_km=.5, full_charge_time_s=2.),
        speed_profile=dict(effective_speed_kmh=3600.), metadata={})


def _worker(rank, rendezvous, output, world_size):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    from offline2online import trainer, branch_exploration
    from test_joint_graph_training import _worker as graph_worker

    original_init = trainer._init_policy_route_pool

    def init_verified_archive(cfg, pool):
        archive = original_init(cfg, pool)
        assert archive is not None and archive.structure_enabled
        for instance_id in archive.instances:
            admission = archive.ingest(instance_id, [1, 2, 3, 4, 0], 5., epoch=0)
            assert admission["verified"] and admission["elite_added"], admission
        return archive

    def search_must_remain_disabled(*args, **kwargs):
        raise AssertionError("The core recipe unexpectedly executed independent search")

    trainer._init_policy_route_pool = init_verified_archive
    branch_exploration.run_branch_exploration = search_must_remain_disabled
    # Reuse the existing worker's real Agent gradient hooks, full-state SHA256,
    # checkpoint capture and Gloo setup; no training loss/rollout is replaced.
    graph_worker(rank, rendezvous, output, world_size)


@pytest.mark.parametrize("task", ["vrptw", "evrptw"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_core_real_training_preserves_graph_sl_archive_and_rank_sync(tmp_path, task, world_size):
    from caliroute.recipes import build_recipe_config
    from test_joint_graph_training import _bundle

    rank_rows = [[_payload(f"{task}_rank{rank}_{i}", task) for i in range(4)] for rank in range(world_size)]
    for rank, rows in enumerate(rank_rows):
        _bundle(tmp_path / f"train/rank{rank}/instances.pkl", rows)
    experts = [item for rows in rank_rows for item in rows]
    _bundle(tmp_path / "experts/instances.pkl", experts)
    validation = [rank_rows[0][0], rank_rows[-1][1]]
    _bundle(tmp_path / "val/instances.pkl", validation)
    expert_csv = tmp_path / "experts/expert_solutions.csv"
    with expert_csv.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "instance_id", "feasible", "objective_distance_km", "vehicle_count", "routes_json"])
        writer.writeheader()
        for item in experts:
            writer.writerow(dict(instance_id=item["instance_id"], feasible=True,
                objective_distance_km=5., vehicle_count=1, routes_json="[[0, 1, 2, 3, 4, 0]]"))

    for rank in range(world_size):
        cfg = build_recipe_config(problem=task, preset="core", seed=31, epochs=1,
            data_root=tmp_path, output_dir=tmp_path / "arm", run_name=f"CORE_{task}_{world_size}_CPU",
            hardware="2080ti_dual" if world_size == 2 else "rtx48_single", world_size=world_size,
            global_batch=4 * world_size, ppo_chunk_size=2, expert_chunk_size=4, eval_batch_size=2)
        assert cfg["model"]["use_joint_graph_encoder"]
        for flag in ("use_rdi_v2", "use_typed_static_fusion", "use_agda_v2", "use_resource_decoder"):
            assert not cfg["model"][flag], flag
        assert cfg["model"]["decoder_observation_mode"] == "feasible"
        assert cfg["offline"]["policy_replay_enabled"]
        assert cfg["offline"]["policy_replay_selection"] == "structural"
        assert not cfg["offline"]["branch_exploration_enabled"]
        assert not cfg["offline"]["exploration_enabled"]
        assert cfg["offline"]["policy_replay_warmup_epochs"] == 25
        assert cfg["offline"]["policy_replay_ramp_epochs"] == 75
        cfg["data"].update(num_customers=4, num_charging_stations=1 if task == "evrptw" else 0,
            train_dataset_path=str(tmp_path / f"train/rank{rank}/instances.pkl"))
        cfg["model"].update(embedding_dim=16, n_encode_layers=2)
        cfg["env"].update(use_jit_mask=False, reward_distance_scale_km=10.,
            observation_distance_scale_km=10., max_steps_factor=4)
        cfg["training"].update(n_traj=2, rollout_steps=32, checkpoint_interval=1,
            mixed_precision=False, debug=False, monitor_interval=1,
            monitor_gradient_components=False, post_update_kl_interval=1)
        cfg["offline"].update(expert_dataset_path=str(tmp_path / "experts/instances.pkl"),
            expert_solution_path=str(expert_csv), policy_replay_warmup_epochs=0,
            policy_replay_ramp_epochs=0)
        cfg["evaluation"].update(eval_interval=1, eval_path=str(tmp_path / "val/instances.pkl"),
            eval_max_steps=32, eval_n_traj=2, eval_batch_size=2, eval_limit=None,
            gurobi_summary_path=str(expert_csv), eval_output_dir=str(tmp_path / "evaluations"))
        (tmp_path / f"config_rank{rank}.yaml").write_text(yaml.safe_dump(cfg))

    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR=str(tmp_path / "numba"))
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-workers", str(tmp_path), str(world_size)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout[-9000:] + completed.stderr[-9000:]
    records = [json.loads((tmp_path / f"rank{rank}_completed.json").read_text()) for rank in range(world_size)]
    for record in records:
        assert record["world_size"] == world_size
        assert record["optimizer_steps"] == 20
        assert record["finite_weights"] and record["finite_gradients"]
        for suffix in ("input_road_projection.0.weight", "layers.0.qkv.weight",
                       "layers.0.edge_value", "layers.0.edge_update.weight", "layers.1.edge_update.weight"):
            name = GRAPH_PREFIX + suffix
            assert record["gradient_max_abs"][name] > 0, name
            assert record["graph_parameter_max_change"][name] > 0, name
    if world_size == 2:
        for key in ("parameter_sha256", "actor_state", "critic_state", "graph_parameter_max_change"):
            assert records[0][key] == records[1][key], key

    for rank in range(world_size):
        rows = [json.loads(line) for line in
            (tmp_path / "arm/monitoring" / f"monitor_rank_{rank}.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        monitor = rows[0]
        assert monitor["optimizer_steps_epoch"] == 20 and monitor["amp_skipped_steps_epoch"] == 0
        assert not monitor["exploration"].get("branch_search_sampled_trajectories", 0)
        assert monitor["advantage"]["sl_advantage_mode"] == "leave_one_out_physical_cost_shared_RMS"
        assert monitor["slppo"]["sl_num_routes_used"] > 0
        assert monitor["slppo"]["sl_candidate_expert_num_routes"] > 0
        assert monitor["replay"]["policy_replay_active_fraction"] > 0
        assert monitor["replay_loss_diagnostics"]["sl_candidate_expert_num_routes"] > 0
        assert monitor["reward_normalization"]["reward_identity_max_abs_error"] < 1e-4
        assert monitor["reward_normalization"]["normalizer_world_size"] == world_size
    checkpoint = torch.load(records[0]["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1 and "reward_normalization_state" in checkpoint
    assert checkpoint["model_integration_signature"]["use_joint_graph_encoder"]
    assert not checkpoint["model_integration_signature"]["use_resource_decoder"]
    for epoch in (0, 1):
        evaluated = [json.loads(line) for line in
            (tmp_path / "evaluations" / f"epoch_{epoch:04d}.jsonl").read_text().splitlines()]
        assert len(evaluated) == len(validation)
        for item in evaluated:
            assert item["feasible"] and item["route_validation"]["checked"] and item["route_validation"]["valid"]
            assert item["route_validation"]["travel_time_source"] == "provided_travel_time_matrix_s"
            assert item["feasibility_source"] == f"environment_success_and_independent_{task}_route_validation"
            if task == "evrptw":
                assert item["route_validation"]["energy_source"] == "provided_energy_matrix_kwh"


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--run-workers":
    location, world_size = Path(sys.argv[2]), int(sys.argv[3])
    arguments = (str(location / "gloo_init"), str(location), world_size)
    if world_size == 1:
        _worker(0, *arguments)
    else:
        mp.spawn(_worker, args=arguments, nprocs=world_size, join=True)
