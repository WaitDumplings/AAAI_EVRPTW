"""Real CPU/Gloo graph-policy PPO/SL updates, search, physical norms and evaluation.

Small routing fixtures retain the production five PPO passes/four minibatches.
No CUDA context or long-lived training process is created by these tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PREFIX = 'backbone.joint_graph_encoder.'


def _payload(identity, task):
    size = 4 if task == 'evrptw' else 3
    distance = np.array([[0., 1., 1.4, 1.1], [1.2, 0., .8, .6],
                         [1.1, .9, 0., .7], [1.3, .6, .8, 0.]], dtype=np.float32)[:size, :size]
    travel = distance.copy()
    travel[1, 2] += .3  # Explicit time need not be a fixed multiple of distance.
    energy = distance * .5
    energy[2, 0] += .2
    return dict(instance_id=identity, working_start_s=0, working_end_s=100,
        depot=np.array([0., 0.]), customers=np.array([[.01, 0.], [0., .01]]),
        charging_stations=np.array([[.01, .01]]) if task == 'evrptw' else np.empty((0, 2)),
        distance_matrix_km=distance, travel_time_matrix_s=travel, energy_matrix_kwh=energy,
        demands_cm3=np.ones(2), package_counts=np.ones(2, dtype=np.int32),
        service_time_s=np.ones(2), tw_s=np.array([[0., 100.], [0., 100.]]),
        cs_time_to_depot_s=np.array([1.3]) if task == 'evrptw' else np.empty(0),
        vehicle=dict(cargo_capacity_cm3=2., battery_capacity_kwh=5.,
                     consumption_kwh_per_km=.5, full_charge_time_s=2.),
        speed_profile=dict(effective_speed_kmh=3600.), metadata={})


def _bundle(path, instances):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump({'instances': instances}, handle)


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        sys.path.insert(0, str(ROOT))
        from offline2online import trainer
        trainer.REPO_ROOT = Path(output)
        cfg = yaml.safe_load((Path(output) / f'config_rank{rank}.yaml').read_text())
        captured = {'gradient_max_abs': {}, 'finite_gradients': True}
        original_agent = trainer.Agent

        class CapturedAgent(original_agent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured['agent'] = self
                captured['initial_graph'] = {name: value.detach().clone()
                    for name, value in self.named_parameters() if name.startswith(GRAPH_PREFIX)}
                assert captured['initial_graph'], 'The real trainer did not instantiate the graph backend'
                for name, parameter in self.named_parameters():
                    if name.startswith(GRAPH_PREFIX):
                        def record_gradient(gradient, key=name):
                            captured['finite_gradients'] &= bool(torch.isfinite(gradient).all())
                            peak = float(gradient.detach().abs().max())
                            captured['gradient_max_abs'][key] = max(captured['gradient_max_abs'].get(key, 0.), peak)
                        parameter.register_hook(record_gradient)

        trainer.Agent = CapturedAgent
        checkpoint = trainer.train_from_config(cfg, seed=31, device='cpu')
        agent = captured['agent']
        normalizer = agent._reward_normalization
        fingerprint = hashlib.sha256()
        finite_weights = True
        for name, value in agent.state_dict().items():
            finite_weights &= bool(torch.isfinite(value).all())
            fingerprint.update(name.encode())
            fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
        graph_change = {name: float((value.detach() - captured['initial_graph'][name]).abs().max())
                       for name, value in agent.named_parameters() if name in captured['initial_graph']}
        record = dict(rank=rank, parameter_sha256=fingerprint.hexdigest(), checkpoint=str(checkpoint),
            optimizer_steps=agent._distributed_context.optimizer_steps, finite_weights=finite_weights,
            finite_gradients=captured['finite_gradients'], gradient_max_abs=captured['gradient_max_abs'],
            graph_parameter_max_change=graph_change,
            actor_updates=int(normalizer.actor.update_count.item()),
            actor_state={key: value.detach().cpu().tolist() for key, value in normalizer.actor.state_dict().items()},
            critic_state={key: value.detach().cpu().tolist() for key, value in normalizer.critic.state_dict().items()})
        (Path(output) / f'rank{rank}_completed.json').write_text(json.dumps(record, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_real_joint_graph_dual_trainer_updates_search_and_physical_evaluation(tmp_path, task):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_evrptw_dual_scratch as launch
    rows = [[_payload(f'{task}_rank{rank}_{index}', task) for index in range(4)] for rank in range(2)]
    for rank, instances in enumerate(rows):
        _bundle(tmp_path / f'train/rank{rank}/instances.pkl', instances)
    experts = rows[0] + rows[1]
    _bundle(tmp_path / 'experts/instances.pkl', experts)
    _bundle(tmp_path / 'val/instances.pkl', [rows[0][0], rows[1][0]])
    expert_path = tmp_path / 'experts/expert_solutions.csv'
    with expert_path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['instance_id', 'feasible', 'objective_distance_km', 'vehicle_count', 'routes_json'])
        writer.writeheader()
        for instance in experts:
            writer.writerow(dict(instance_id=instance['instance_id'], feasible=True,
                objective_distance_km=2.9, vehicle_count=1, routes_json='[[0, 1, 2, 0]]'))
    base = yaml.safe_load((ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml').read_text())
    for rank in range(2):
        cfg = launch.build_config(base, variant='optimized', encoder_variant='graph', task=task,
            output=tmp_path / 'arm', run_name=f'GRAPH_{task.upper()}_DUAL_CPU', data_root=tmp_path,
            seed=31, epochs=1, eval_interval=1, batch_per_gpu=4, chunk_size=2, expert_chunk_size=4)
        cfg['data'].update(num_customers=2, num_charging_stations=1 if task == 'evrptw' else 0,
            train_dataset_path=str(tmp_path / f'train/rank{rank}/instances.pkl'))
        cfg['model'].update(embedding_dim=16, n_encode_layers=2)
        cfg['env'].update(use_jit_mask=False, reward_distance_scale_km=10.,
            observation_distance_scale_km=10., max_steps_factor=4)
        cfg['training'].update(num_envs_per_gpu=4, n_traj=2, rollout_steps=20,
            checkpoint_interval=1, mixed_precision=False, debug=False,
            monitor_interval=1, monitor_gradient_components=False, post_update_kl_interval=1)
        cfg['offline'].update(expert_dataset_path=str(tmp_path / 'experts/instances.pkl'),
            expert_solution_path=str(expert_path), exploration_interval=1,
            exploration_instances=1, exploration_trajectories=2, exploration_max_prefix_steps=2)
        cfg['evaluation'].update(eval_interval=1, eval_path=str(tmp_path / 'val/instances.pkl'),
            eval_max_steps=20, eval_n_traj=2, eval_batch_size=2, eval_limit=None,
            gurobi_summary_path=str(expert_path), eval_output_dir=str(tmp_path / 'evaluations'))
        launch.scratch.assert_scratch(cfg)
        (tmp_path / f'config_rank{rank}.yaml').write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    executed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run-workers', str(tmp_path)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    assert executed.returncode == 0, executed.stdout[-9000:] + executed.stderr[-9000:]
    records = [json.loads((tmp_path / f'rank{rank}_completed.json').read_text()) for rank in range(2)]
    for record in records:
        assert record['optimizer_steps'] == 20  # Five complete passes, four minibatches.
        assert record['finite_weights'] and record['finite_gradients'] and record['actor_updates'] > 0
        for suffix in ('input_road_projection.0.weight', 'layers.0.qkv.weight',
                       'layers.0.edge_value', 'layers.0.edge_update.weight', 'layers.1.edge_update.weight'):
            name = GRAPH_PREFIX + suffix
            assert record['gradient_max_abs'][name] > 0, name
            assert record['graph_parameter_max_change'][name] > 0, name
    for key in ('parameter_sha256', 'actor_state', 'critic_state', 'graph_parameter_max_change'):
        assert records[0][key] == records[1][key], key
    checkpoint = torch.load(records[0]['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 1 and 'reward_normalization_state' in checkpoint
    signature = checkpoint['model_integration_signature']
    assert signature['use_joint_graph_encoder'] and signature['joint_graph_edge_dim'] == 32
    assert signature['joint_graph_dropout'] == 0.0 and not signature['use_edge_relation_encoder']
    assert any(name.startswith(GRAPH_PREFIX) for name in checkpoint['model_state_dict'])
    for rank in range(2):
        monitors = [json.loads(line) for line in (tmp_path / 'arm/monitoring' / f'monitor_rank_{rank}.jsonl').read_text().splitlines()]
        assert len(monitors) == 1
        monitor = monitors[0]
        assert monitor['optimizer_steps_epoch'] == 20 and monitor['amp_skipped_steps_epoch'] == 0
        assert monitor['model_integration']['use_joint_graph_encoder']
        assert monitor['exploration']['branch_search_sampled_trajectories'] == 2
        assert monitor['exploration']['branch_search_mask_violations'] == 0
        assert monitor['exploration']['branch_search_completed_trajectories'] == 2
    for epoch in (0, 1):
        evaluated = [json.loads(line) for line in (tmp_path / 'evaluations' / f'epoch_{epoch:04d}.jsonl').read_text().splitlines()]
        assert len(evaluated) == 2
        for item in evaluated:
            assert item['feasible'] and item['route_validation']['checked'] and item['route_validation']['valid']
            assert item['route_validation']['travel_time_source'] == 'provided_travel_time_matrix_s'
            assert item['feasibility_source'] == f'environment_success_and_independent_{task}_route_validation'
            if task == 'evrptw':
                assert item['route_validation']['energy_source'] == 'provided_energy_matrix_kwh'


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--run-workers':
    location = Path(sys.argv[2])
    mp.spawn(_worker, args=(str(location / 'gloo_init'), str(location)), nprocs=2, join=True)
