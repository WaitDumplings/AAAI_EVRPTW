"""Pure PPO warmup is a real optimizer-continuous stage, not zero-weight SL."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppo_warmup import PPOWarmupSchedule


@pytest.mark.parametrize('value', [-1, 1.5, '100', True])
def test_invalid_warmup_fails(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        PPOWarmupSchedule({'training': {'ppo_warmup_epochs': value}})


def test_schedule_boundaries_and_no_flag_legacy_behavior():
    assert PPOWarmupSchedule({'offline': {'method': 'sl_ppo'}}).fields(1)['effective_offline_method'] == 'sl_ppo'
    schedule = PPOWarmupSchedule({'training': {'ppo_warmup_epochs': 100}, 'offline': {'method': 'sl_ppo'}})
    assert schedule.fields(100)['effective_offline_method'] == 'ppo'
    assert schedule.fields(101)['effective_offline_method'] == 'sl_ppo'
    assert schedule.fields(101)['phase_epoch'] == 1
    assert schedule.checkpoint_metadata(100)['next_effective_offline_method'] == 'sl_ppo'
    assert schedule.checkpoint_metadata(100)['warmup_boundary_checkpoint']


@pytest.mark.parametrize('section,key,value', [('offline', 'use_priority_sampler', True),
    ('training', 'use_oracle_ordering_hint', True), ('offline', 'bc_warmup_epochs', 2),
    ('offline', 'hard_ref_kl_coef', .1)])
def test_warmup_rejects_hidden_expert_channels(section, key, value):
    cfg = {'training': {'ppo_warmup_epochs': 1}, 'offline': {'method': 'sl_ppo'}}
    cfg[section][key] = value
    with pytest.raises(ValueError, match='Pure PPO warmup'):
        PPOWarmupSchedule(cfg)


def _fixture(tmp_path, *, method='sl_ppo', warmup=1, epochs=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    distance = np.array([[0., 1., 1.4], [1.2, 0., .8], [1.1, .9, 0.]])
    payload = dict(working_start_s=0., working_end_s=100., depot=np.array([0., 0.]),
        customers=np.array([[.01, 0.], [0., .01]]), charging_stations=np.empty((0, 2)),
        distance_matrix_km=distance, travel_time_matrix_s=distance,
        demands_cm3=np.ones(2), package_counts=np.ones(2, dtype=np.int32),
        service_time_s=np.zeros(2), tw_s=np.array([[0., 100.], [0., 100.]]),
        vehicle=dict(cargo_capacity_cm3=2., design_speed_kmh=3600.),
        speed_profile=dict(effective_speed_kmh=3600.), metadata={})
    dataset = tmp_path/'train/instances.pkl'
    dataset.parent.mkdir()
    with dataset.open('wb') as handle:
        pickle.dump({'instances': [dict(payload, instance_id=f'toy_{i}') for i in range(8)]}, handle)
    experts = tmp_path/'experts.csv'
    with experts.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['instance_id', 'objective_distance_km', 'vehicle_count', 'routes_json'])
        writer.writeheader()
        for i in range(8):
            writer.writerow(dict(instance_id=f'toy_{i}', objective_distance_km=2.9,
                                 vehicle_count=1, routes_json='[[0,1,2,0]]'))
    cfg = dict(run_name=tmp_path.name,
        data=dict(problem_type='vrptw', num_customers=2, num_charging_stations=0,
                  train_dataset_path=str(dataset), fixed_dataset_sample_mode='shuffle_cycle'),
        env=dict(use_fast_env=True, use_jit_mask=False, info_level='light', normalize_reward=True,
                 reward_distance_scale_km=10.),
        model=dict(embedding_dim=16, n_encode_layers=1, use_decomposed_critic=False),
        critic=dict(use_decomposed_critic=False, advantage_mode='total'),
        training=dict(epochs=epochs, ppo_warmup_epochs=warmup, num_envs_per_gpu=2, n_traj=4,
                      rollout_steps=12, ppo_update_epochs=3, ppo_step_chunk_size=4,
                      num_minibatches=1, checkpoint_interval=50, learning_rate=1e-4,
                      mixed_precision=False, debug=False, monitor_interval=1),
        evaluation=dict(eval_interval=2, eval_path=str(dataset), eval_n_traj=2,
                        eval_batch_size=2, eval_max_steps=12, eval_decode_mode='sample',
                        eval_seed=17000031, eval_before_training=False,
                        eval_output_dir=str(tmp_path/'evaluations')),
        offline=dict(method=method, use_priority_sampler=False, expert_solution_path=str(experts),
                     expert_dataset_path=str(dataset), sl_coef=.5, sl_expert_candidate_weight=.6,
                     strict_replay=True, original_share_static_expert_observations=True),
        advantage=dict(use_group_advantage=True, group_adv_coef=1., use_reference_advantage=True,
                       reference_adv_coef=.5, use_expert_solution_level=True,
                       sl_expert_candidate_weight=.6, sl_expert_logprob_chunk_size=8))
    if method == 'ppo':
        cfg['advantage'] = dict(use_group_advantage=False, use_reference_advantage=False,
                                use_expert_solution_level=False)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return config


def _run(directory, config, source, *, original, world_size):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR=str(directory/'numba'))
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    prefix = [sys.executable]
    if world_size > 1:
        prefix += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '--max-restarts=0']
    if original:
        cmd = prefix + [str(ROOT/'scripts/run_original_scratch.py'), '--source-root', str(source),
                        '--config', str(config), '--seed', '31', '--device', 'cpu']
    else:
        cmd = prefix + [str(Path(__file__).resolve()), '--worker', str(config), str(source)]
    result = subprocess.run(cmd, env=env, cwd=directory, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout[-5000:]+result.stderr[-7000:]
    return result


@pytest.mark.parametrize('original', [False, True])
def test_real_two_rank_ppo_warmup_then_slppo_without_optimizer_reset(tmp_path, original):
    directory = tmp_path/'scheduled'
    config = _fixture(directory)
    if original:
        source = tmp_path/'archive/CaliRoute'
        archived = subprocess.run(['git', 'archive', '--format=tar', 'f388343', 'CaliRoute'],
                                  cwd=ROOT.parent, check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
            archive.extractall(source.parent, filter='data')
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
    else:
        source = tmp_path/'modern'
    result = _run(directory, config, source, original=original, world_size=2)
    logs = source/'results/logs/Cus_2_CS_0/scheduled/seed_31/train_log.csv'
    rows = list(csv.DictReader(logs.open()))
    assert [r['train_mode'] for r in rows] == ['ppo', 'sl_ppo']
    assert [r['training_phase'] for r in rows] == ['ppo_warmup', 'slppo']
    assert rows[0].get('sl_num_routes_used', '') in ('', '0')
    assert float(rows[1]['sl_num_routes_used']) > 0
    assert float(rows[1]['sl_coef']) == .5
    checkpoints = source/'results/checkpoints/Cus_2_CS_0/scheduled/seed_31'
    boundary = torch.load(checkpoints/'checkpoint_epoch_0001.pt', map_location='cpu', weights_only=False)
    final = torch.load(checkpoints/'checkpoint_final.pt', map_location='cpu', weights_only=False)
    phase = boundary['config']['training_stage'] if original else boundary['training_phase']
    assert phase['warmup_boundary_checkpoint'] and phase['next_effective_offline_method'] == 'sl_ppo'
    assert boundary['config']['offline']['sl_coef'] == .5
    assert {int(state['step']) for state in boundary['optimizer_state_dict']['state'].values()} == {3}
    assert {int(state['step']) for state in final['optimizer_state_dict']['state'].values()} == {6}
    assert '[TrainingPhase]' in result.stdout
    if original:
        assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
        metadata = json.loads((directory/'original_adapter.json').read_text())
        peer = json.loads((directory/'original_adapter_rank_1.json').read_text())
        assert metadata['optimizer_steps'] == peer['optimizer_steps'] == 6
        assert metadata['final_parameter_max_abs_difference'] == 0.
        assert (checkpoints/'checkpoint_epoch_0001.phase.json').is_file()
    else:
        monitors = [json.loads(line) for line in (logs.parent/'monitor_rank_0.jsonl').read_text().splitlines()]
        assert monitors[0]['slppo'] == {}
        assert monitors[0]['exploration'] == {} and monitors[0]['replay'] == {}


@pytest.mark.parametrize('original', [False, True])
def test_warmup_first_update_matches_standalone_pure_ppo(tmp_path, original):
    scheduled = tmp_path/'scheduled'
    pure = tmp_path/'pure'
    warm_config = _fixture(scheduled, epochs=1)
    pure_config = _fixture(pure, method='ppo', warmup=0, epochs=1)
    if original:
        source = tmp_path/'archive/CaliRoute'
        archived = subprocess.run(['git', 'archive', '--format=tar', 'f388343', 'CaliRoute'],
                                  cwd=ROOT.parent, check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
            archive.extractall(source.parent, filter='data')
    else:
        source = tmp_path/'modern'
    _run(scheduled, warm_config, source, original=original, world_size=1)
    _run(pure, pure_config, source, original=original, world_size=1)
    def load(name):
        return torch.load(source/f'results/checkpoints/Cus_2_CS_0/{name}/seed_31/checkpoint_final.pt',
                          map_location='cpu', weights_only=False)['model_state_dict']
    left, right = load('scheduled'), load('pure')
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left), 'Warmup must equal pure PPO, not auxiliary-loss PPO'


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--worker':
    sys.path.insert(0, str(ROOT))
    from offline2online import trainer
    torch.set_num_threads(1)
    trainer.REPO_ROOT = Path(sys.argv[3])
    cfg = yaml.safe_load(Path(sys.argv[2]).read_text())
    trainer.train_from_config(cfg, seed=31, device='cpu')
