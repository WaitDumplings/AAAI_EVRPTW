"""Original source isolation, scratch initialization, and unified measurements."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_scratch_adapter_test', ROOT / 'scripts/run_original_scratch.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
validation = adapter._validation


def fixture_instance():
    return SimpleNamespace(
        instance_id='toy', num_customers=3, num_charging_stations=0,
        demands_cm3=np.array([1., 1., 1.]),
        distance_matrix_km=np.ones((4, 4)) - np.eye(4),
        vehicle={'cargo_capacity_cm3': 2., 'design_speed_kmh': 3600.},
        speed_profile={'effective_speed_kmh': 3600.},
        working_start_s=1000., working_end_s=1030.,
        tw_s=np.array([[1010., 1011.], [1016., 1020.], [1000., 1002.]]),
        service_time_s=np.array([5., 3., 2.]),
    )


def fixture_row():
    return {'routes': [[0, 1, 2, 0], [0, 3, 0]], 'objective_distance_km': 5., 'feasible': True}


@pytest.mark.parametrize('change', ['valid', 'capacity', 'window', 'no_return', 'direct_return', 'explicit_time', 'directed_distance', 'duplicate'])
def test_independent_numpy_validator_matches_modern(change):
    from offline2online.trainer import _validate_vrptw_eval_route
    data, row = fixture_instance(), fixture_row()
    if change == 'capacity':
        data.vehicle['cargo_capacity_cm3'] = 1.9
    elif change == 'window':
        data.tw_s[1, 1] = 1015.
    elif change == 'no_return':
        row['routes'][0].pop()
        row['objective_distance_km'] -= 1.
    elif change == 'direct_return':
        data.distance_matrix_km[1, 0] = 100.
    elif change == 'explicit_time':
        data.raw = {'travel_time_matrix_s': data.distance_matrix_km.copy()}
        data.raw['travel_time_matrix_s'][1, 2] = 10.
    elif change == 'directed_distance':
        data.distance_matrix_km[0, 1] = 4.
        row['objective_distance_km'] += 3.
    elif change == 'duplicate':
        row['routes'][0][2] = 1
    actual = validation.validate_vrptw_route(data, row)
    expected = _validate_vrptw_eval_route(data, row, prefer_explicit_edge_matrices=True)
    for key in ('valid', 'indices_valid', 'depot_endpoints_valid', 'customer_coverage_valid', 'capacity_valid',
                'distance_matches', 'time_windows_valid', 'service_completion_valid',
                'depot_return_valid', 'return_reachability_valid', 'travel_time_source'):
        assert actual[key] == expected[key], (change, key)
    assert actual['valid'] is (change in ('valid', 'directed_distance'))


def test_raw_time_sidecar_recovers_explicit_edges_dropped_by_original_adapter():
    data, row = fixture_instance(), fixture_row()
    travel = data.distance_matrix_km.copy()
    travel[1, 2] = 10.
    assert validation.validate_vrptw_route(data, row)['valid']
    checked = validation.validate_vrptw_route(data, row, raw_payload={'travel_time_matrix_s': travel})
    assert not checked['valid'] and checked['travel_time_source'] == 'provided_travel_time_matrix_s'


@pytest.mark.parametrize('key', ['init_checkpoint', 'resume_checkpoint', 'reference_checkpoint', 'pretrained_path', 'nested_checkpoint_path'])
def test_scratch_rejects_every_checkpoint_path(key):
    with pytest.raises(ValueError, match='forbids'):
        adapter.assert_scratch_config({'training': {key: '/does/not/exist.pt'}})
    adapter.assert_scratch_config({'training': {key: None}, 'resume_start_epoch': 1})


def test_rng_context_restores_torch_python_numpy_even_on_error():
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with pytest.raises(RuntimeError):
        with adapter.isolated_evaluation_rng(17, 'cpu'):
            torch.rand(5); random.random(); np.random.rand(5)
            raise RuntimeError('abort')
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]


def test_original_best_selection_keeps_success_and_original_tie_order():
    select = validation.select_original_best_index
    assert select([4., 2., 3.], [True, False, True], [3, 2, 3]) == 2
    assert select([4., 2., 3.], [False]*3, [2, 1, 2]) == 2
    assert select([3., 3., 3.], [True]*3, [3]*3) == int(np.argsort(np.array([3., 3., 3.]))[0])
    assert select([float('nan')], [False], [0]) is None


@pytest.fixture(scope='module')
def original_archive(tmp_path_factory):
    target = tmp_path_factory.mktemp('original_archive')
    archived = subprocess.run(['git', 'archive', '--format=tar', 'f388343', 'CaliRoute'],
                              cwd=ROOT.parent, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archived)) as source:
        source.extractall(target, filter='data')
    return target / 'CaliRoute'


def _run_original(tmp_path, source_root, *, initial_eval, hostile_time=False):
    tmp_path.mkdir(parents=True)
    data = fixture_instance()
    # Broader windows make random policies successful while preserving capacity.
    payload = dict(vars(data), problem_class='VRPTW', depot=[0., 0.],
                   customers=[[1., 0.], [0., 1.], [1., 1.]],
                   working_start_s=0., working_end_s=100.,
                   tw_s=np.array([[0., 100.]]*3), service_time_s=np.zeros(3),
                   travel_time_matrix_s=data.distance_matrix_km * (200. if hostile_time else 1.))
    payload.pop('num_customers'); payload.pop('num_charging_stations')
    bundle = tmp_path / 'instances.pkl'
    with bundle.open('wb') as handle:
        pickle.dump({'instances': [dict(payload, instance_id=f'toy_{i}') for i in range(2)]}, handle)
    cfg = {
        'run_name': tmp_path.name,
        'data': {'problem_type': 'vrptw', 'num_customers': 3, 'num_charging_stations': 0,
                 'train_dataset_path': str(bundle), 'fixed_dataset_sample_mode': 'shuffle_cycle'},
        'env': {'use_fast_env': True, 'use_jit_mask': False, 'info_level': 'light',
                'normalize_reward': True, 'reward_distance_scale_km': 10.},
        'model': {'embedding_dim': 16, 'n_encode_layers': 1, 'use_decomposed_critic': False,
                  'graph_token': True, 'dynamic_decision': True},
        'critic': {'use_decomposed_critic': False, 'advantage_mode': 'total'},
        'training': {'epochs': 1, 'num_envs_per_gpu': 2, 'n_traj': 2, 'rollout_steps': 12,
                     'ppo_update_epochs': 1, 'ppo_chunk_steps': 4, 'num_minibatches': 1,
                     'learning_rate': 1e-4, 'mixed_precision': False, 'debug': False},
        'evaluation': {'eval_interval': 1, 'eval_path': str(bundle), 'eval_n_traj': 2,
                       'eval_batch_size': 2, 'eval_max_steps': 12, 'eval_decode_mode': 'sample',
                       'eval_seed': 17000031, 'eval_before_training': initial_eval,
                       'eval_output_dir': str(tmp_path / 'evaluations')},
        'offline': {'method': 'ppo', 'use_priority_sampler': False},
    }
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/run_original_scratch.py'),
                                '--source-root', str(source_root), '--config', str(config),
                                '--seed', '31', '--device', 'cpu'], cwd=tmp_path,
                               env=environment, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-7000:]
    metadata = json.loads((tmp_path / 'original_adapter.json').read_text())
    logs = source_root / 'results/logs/Cus_3_CS_0' / tmp_path.name / 'seed_31'
    evals = list(csv.DictReader((logs / 'eval_log.csv').open()))
    return metadata, evals


def test_real_original_archive_runs_without_checkpoint_and_e0_preserves_training(tmp_path, original_archive):
    before = {str(p.relative_to(original_archive)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in original_archive.rglob('*.py')}
    formal, rows = _run_original(tmp_path / 'formal', original_archive, initial_eval=True)
    preflight, rows_without_zero = _run_original(tmp_path / 'preflight', original_archive, initial_eval=False)
    assert [int(row['epoch']) for row in rows] == [0, 1]
    assert [int(row['epoch']) for row in rows_without_zero] == [1]
    assert formal['state'] == preflight['state'] == 'completed'
    assert formal['independent_route_validation'] and formal['per_instance_route_export']
    assert formal['initialization'] == 'random; checkpoint loading forbidden'
    for source in formal['source_modules'].values():
        assert Path(source).is_relative_to(original_archive)
    after = {str(p.relative_to(original_archive)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in original_archive.rglob('*.py')}
    assert before == after, 'Measurement instrumentation must not edit original sources'
    left = torch.load(formal['checkpoint_final'], map_location='cpu', weights_only=False)
    right = torch.load(preflight['checkpoint_final'], map_location='cpu', weights_only=False)
    model_key = 'agent_state_dict' if 'agent_state_dict' in left else 'model_state_dict'
    assert left[model_key].keys() == right[model_key].keys()
    assert all(torch.equal(left[model_key][key], right[model_key][key]) for key in left[model_key])
    for epoch in (0, 1):
        per_instance = [json.loads(line) for line in (tmp_path / 'formal/evaluations' / f'epoch_{epoch:04d}.jsonl').read_text().splitlines()]
        assert len(per_instance) == 2
        assert all(row['eval_seed'] == 17000031 and row['epoch'] == epoch for row in per_instance)
        assert all(row['feasible'] and row['environment_feasible'] and row['independently_valid'] for row in per_instance)
        assert all(row['route_validation']['travel_time_source'] == 'provided_travel_time_matrix_s' for row in per_instance)


def test_original_training_mask_difference_is_exposed_by_unified_eval(tmp_path, original_archive):
    metadata, rows = _run_original(tmp_path / 'hostile', original_archive, initial_eval=True, hostile_time=True)
    assert float(rows[0]['eval_environment_feasible_rate']) == 1.
    assert float(rows[0]['eval_independent_valid_rate']) == 0.
    assert float(rows[0]['eval_feasible_rate']) == 0.
    assert metadata['state'] == 'completed'
    exports = [json.loads(line) for line in (tmp_path / 'hostile/evaluations/epoch_0000.jsonl').read_text().splitlines()]
    assert all(item['environment_feasible'] and not item['feasible'] for item in exports)
