"""Real two-rank CPU/Gloo EVRPTW training with failed episodes and shared eval."""
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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _payload(identity, *, impossible=False):
    distance = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    travel = distance.copy()
    travel[1, 3] = 1.5  # Authoritative edge seconds are deliberately nonuniform.
    return dict(instance_id=identity, working_start_s=0, working_end_s=100,
        depot=np.array([0., 0.]), customers=np.array([[.01, 0.], [0., .01]]),
        charging_stations=np.array([[.01, .01]]), distance_matrix_km=distance,
        travel_time_matrix_s=travel, energy_matrix_kwh=distance*.5,
        demands_cm3=np.array([1., 3. if impossible else 1.]),
        package_counts=np.ones(2, dtype=np.int32), service_time_s=np.ones(2),
        tw_s=np.array([[0., 100.], [0., 100.]]), cs_time_to_depot_s=np.array([1.]),
        vehicle=dict(cargo_capacity_cm3=2., battery_capacity_kwh=5.,
                     consumption_kwh_per_km=.5, full_charge_time_s=2.),
        speed_profile=dict(effective_speed_kmh=3600.), metadata={})


def _bundle(path, payloads):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump({'instances': payloads}, handle)


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        sys.path.insert(0, str(ROOT))
        from offline2online import trainer
        trainer.REPO_ROOT = Path(output)
        cfg = yaml.safe_load((Path(output)/f'config_rank{rank}.yaml').read_text())
        captured = {}
        original_agent = trainer.Agent
        original_collect = trainer.collect_rollout

        class CapturedAgent(original_agent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured['agent'] = self

        def collect(*args, **kwargs):
            batch = original_collect(*args, **kwargs)
            if 'first_rollout' not in captured:
                captured['first_rollout'] = dict(
                    success=np.stack([item['success'] for item in batch.final_infos]),
                    all_done=bool(batch.dones[-1].all()),
                    rewards_sum=batch.rewards.sum(0).detach().cpu(),
                    objective=np.stack([item['objective_distance_km'] for item in batch.final_infos]))
            return batch

        trainer.Agent = CapturedAgent
        trainer.collect_rollout = collect
        checkpoint = trainer.train_from_config(cfg, seed=31, device='cpu')
        agent = captured['agent']
        normalizer = agent._reward_normalization
        fingerprint = hashlib.sha256()
        for name, value in agent.state_dict().items():
            fingerprint.update(name.encode())
            fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
        rollout = captured['first_rollout']
        failures = ~rollout['success']
        physical_error = np.abs(rollout['rewards_sum'].numpy() + rollout['objective']/10. + failures*100.).max()
        record = dict(rank=rank, parameter_sha256=fingerprint.hexdigest(), checkpoint=str(checkpoint),
            all_episodes_done=rollout['all_done'], feasible_trajectories=int(rollout['success'].sum()),
            failed_trajectories=int(failures.sum()), physical_error=float(abs(physical_error)),
            optimizer_steps=agent._distributed_context.optimizer_steps,
            actor_updates=int(normalizer.actor.update_count.item()),
            actor_state={key:value.detach().cpu().tolist() for key,value in normalizer.actor.state_dict().items()},
            critic_state={key:value.detach().cpu().tolist() for key,value in normalizer.critic.state_dict().items()})
        (Path(output)/f'rank{rank}_completed.json').write_text(json.dumps(record, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_evrptw_failed_rollouts_shared_norm_and_independent_eval(tmp_path):
    sys.path.insert(0, str(ROOT/'scripts'))
    import run_evrptw_dual_scratch as launch
    bad = [_payload('bad0', impossible=True), _payload('bad1', impossible=True)]
    good = [_payload('good0'), _payload('good1')]
    for rank, rows in enumerate((bad, good)):
        _bundle(tmp_path/f'train/rank{rank}/instances.pkl', rows)
    _bundle(tmp_path/'experts/instances.pkl', bad + good)
    _bundle(tmp_path/'val/instances.pkl', [good[0], bad[0]])
    expert_path = tmp_path/'experts/expert_solutions.csv'
    with expert_path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['instance_id','feasible','objective_distance_km','vehicle_count','routes_json'])
        writer.writeheader()
        # Partial expert coverage, as in the real EVRPTW release.
        writer.writerow(dict(instance_id='good0', feasible=True, objective_distance_km=3.,
                             vehicle_count=1, routes_json='[[0, 1, 2, 0]]'))
    base = yaml.safe_load((ROOT/'configs/experiments/physics_exploration_vrptw100.yaml').read_text())
    for rank in range(2):
        cfg = launch.build_config(base, variant='optimized', output=tmp_path/'arm',
            run_name='EVRPTW_DUAL_CPU', data_root=tmp_path, seed=31, epochs=1,
            eval_interval=1, batch_per_gpu=4, chunk_size=2, expert_chunk_size=4)
        cfg['data'].update(num_customers=2, num_charging_stations=1,
                           train_dataset_path=str(tmp_path/f'train/rank{rank}/instances.pkl'))
        cfg['model'].update(embedding_dim=16, n_encode_layers=1)
        cfg['env'].update(use_jit_mask=False, reward_distance_scale_km=10.,
                          observation_distance_scale_km=10., max_steps_factor=4)
        cfg['training'].update(epochs=1, num_envs_per_gpu=2, n_traj=2,
            rollout_steps=20, ppo_update_epochs=2, num_minibatches=2,
            checkpoint_interval=1, mixed_precision=False, debug=False,
            monitor_interval=1, monitor_gradient_components=False, post_update_kl_interval=1)
        cfg['offline'].update(expert_dataset_path=str(tmp_path/'experts/instances.pkl'),
            expert_solution_path=str(expert_path), exploration_interval=1,
            exploration_instances=1, exploration_trajectories=2,
            exploration_max_prefix_steps=2)
        cfg['evaluation'].update(eval_interval=1, eval_path=str(tmp_path/'val/instances.pkl'),
            eval_max_steps=20, eval_n_traj=2, eval_batch_size=2, eval_limit=None,
            gurobi_summary_path=str(expert_path), eval_output_dir=str(tmp_path/'evaluations'))
        (tmp_path/f'config_rank{rank}.yaml').write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR=str(tmp_path/'numba'))
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(key, None)
    executed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run-workers', str(tmp_path)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert executed.returncode == 0, executed.stdout[-7000:] + executed.stderr[-7000:]
    records = [json.loads((tmp_path/f'rank{rank}_completed.json').read_text()) for rank in range(2)]
    assert records[0]['feasible_trajectories'] == 0 and records[0]['failed_trajectories'] == 4
    assert records[1]['feasible_trajectories'] == 4 and records[1]['failed_trajectories'] == 0
    for record in records:
        assert record['all_episodes_done'] and record['physical_error'] < 1e-4
        assert record['optimizer_steps'] == 4
        assert record['actor_updates'] > 0
    for key in ('parameter_sha256','actor_state','critic_state'):
        assert records[0][key] == records[1][key]
    checkpoint = torch.load(records[0]['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 1 and 'reward_normalization_state' in checkpoint
    for epoch in (0, 1):
        evaluated = [json.loads(line) for line in (tmp_path/'evaluations'/f'epoch_{epoch:04d}.jsonl').read_text().splitlines()]
        assert len(evaluated) == 2
        by_id = {item['instance_id']: item for item in evaluated}
        assert by_id['good0']['feasible'] and not by_id['bad0']['feasible']
        missing_reference = by_id['bad0'].get('reference_objective_distance_km')
        assert missing_reference is None or not np.isfinite(missing_reference)
        for item in evaluated:
            assert item['route_validation']['checked']
            assert item['route_validation']['problem_type'] == 'evrptw'
            assert item['route_validation']['travel_time_source'] == 'provided_travel_time_matrix_s'
            assert item['route_validation']['energy_source'] == 'provided_energy_matrix_kwh'
            assert item['feasibility_source'] == 'environment_success_and_independent_evrptw_route_validation'


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--run-workers':
    location = Path(sys.argv[2])
    mp.spawn(_worker, args=(str(location/'gloo_init'), str(location)), nprocs=2, join=True)
