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


def _run_original(tmp_path, source_root, *, initial_eval, hostile_time=False, world_size=1, problem='vrptw', hostile_energy=False, method='ppo', share_static=False):
    tmp_path.mkdir(parents=True)
    data = fixture_instance()
    # Broader windows make random policies successful while preserving capacity.
    payload = dict(vars(data), problem_class='VRPTW', depot=[0., 0.],
                   customers=[[1., 0.], [0., 1.], [1., 1.]],
                   working_start_s=0., working_end_s=100.,
                   tw_s=np.array([[0., 100.]]*3), service_time_s=np.zeros(3),
                   travel_time_matrix_s=data.distance_matrix_km * (200. if hostile_time else 1.))
    payload.pop('num_customers'); payload.pop('num_charging_stations')
    if problem == 'evrptw':
        distance = np.ones((5, 5)) - np.eye(5)
        payload.update(problem_class='EVRPTW', charging_stations=[[.5, .5]],
                       package_counts=np.ones(3, dtype=np.int32), cs_time_to_depot_s=np.array([1.]),
                       distance_matrix_km=distance, travel_time_matrix_s=distance,
                       energy_matrix_kwh=distance * (200. if hostile_energy else .1),
                       vehicle=dict(data.vehicle, battery_capacity_kwh=100.,
                                    consumption_kwh_per_km=.1, full_charge_time_s=1.))
    bundle = tmp_path / 'instances.pkl'
    with bundle.open('wb') as handle:
        pickle.dump({'instances': [dict(payload, instance_id=f'toy_{i}') for i in range(8 if world_size > 1 else 2)]}, handle)
    cfg = {
        'run_name': tmp_path.name,
        'data': {'problem_type': problem, 'num_customers': 3, 'num_charging_stations': int(problem == 'evrptw'),
                 'train_dataset_path': str(bundle), 'fixed_dataset_sample_mode': 'shuffle_cycle'},
        'env': {'use_fast_env': True, 'use_jit_mask': False, 'info_level': 'light',
                'normalize_reward': True, 'reward_distance_scale_km': 10.},
        'model': {'embedding_dim': 16, 'n_encode_layers': 1, 'use_decomposed_critic': False,
                  'graph_token': True, 'dynamic_decision': True},
        'critic': {'use_decomposed_critic': False, 'advantage_mode': 'total'},
        'training': {'epochs': 1, 'num_envs_per_gpu': 2, 'n_traj': 2, 'rollout_steps': 12,
                     'ppo_update_epochs': 1, 'ppo_step_chunk_size': 4, 'num_minibatches': 1,
                     'learning_rate': 1e-4, 'mixed_precision': False, 'debug': False},
        'evaluation': {'eval_interval': 1, 'eval_path': str(bundle), 'eval_n_traj': 2,
                       'eval_batch_size': 2, 'eval_max_steps': 12, 'eval_decode_mode': 'sample',
                       'eval_seed': 17000031, 'eval_before_training': initial_eval,
                       'eval_output_dir': str(tmp_path / 'evaluations')},
        'offline': {'method': method, 'use_priority_sampler': False,
                    'original_share_static_expert_observations': share_static},
    }
    if method == 'sl_ppo':
        cfg['training'].update(num_envs_per_gpu=4, ppo_update_epochs=5, num_minibatches=4)
        expert_path = tmp_path / 'experts.csv'
        with expert_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['instance_id', 'objective_distance_km', 'vehicle_count', 'routes_json'])
            writer.writeheader()
            for index in range(8 if world_size > 1 else 2):
                writer.writerow(dict(instance_id=f'toy_{index}', objective_distance_km=5., vehicle_count=2,
                                     routes_json=json.dumps([[0, 1, 2, 0], [0, 3, 0]])))
        cfg['offline'].update(expert_solution_path=str(expert_path), expert_dataset_path=str(bundle),
                              sl_coef=.5, sl_expert_candidate_weight=.6, strict_replay=True)
        cfg['advantage'] = dict(use_group_advantage=True, group_adv_coef=1., use_reference_advantage=True,
                                reference_adv_coef=.5, use_expert_solution_level=True,
                                sl_expert_candidate_weight=.6, sl_expert_logprob_chunk_size=8)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
    prefix = [sys.executable]
    if world_size > 1:
        prefix += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f'--nproc-per-node={world_size}', '--max-restarts=0']
    completed = subprocess.run(prefix + [str(ROOT / 'scripts/run_original_scratch.py'),
                                '--source-root', str(source_root), '--config', str(config),
                                '--seed', '31', '--device', 'cpu'], cwd=tmp_path,
                               env=environment, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-7000:]
    metadata = json.loads((tmp_path / 'original_adapter.json').read_text())
    logs = source_root / f'results/logs/Cus_3_CS_{int(problem == "evrptw")}' / tmp_path.name / 'seed_31'
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


@pytest.mark.parametrize('problem', ['vrptw', 'evrptw'])
def test_real_original_archive_two_rank_gloo_training_and_rank0_eval(tmp_path, original_archive, problem):
    directory = tmp_path / ('dual_' + problem)
    metadata, rows = _run_original(directory, original_archive, initial_eval=True, world_size=2, problem=problem)
    second = json.loads((directory / 'original_adapter_rank_1.json').read_text())
    assert metadata['state'] == second['state'] == 'completed'
    assert metadata['final_parameter_max_abs_difference'] == second['final_parameter_max_abs_difference'] == 0.
    assert metadata['optimizer_steps'] == second['optimizer_steps'] == 1
    assert metadata['distributed_runtime']['sampling_seed'] != second['distributed_runtime']['sampling_seed']
    assert [int(row['epoch']) for row in rows] == [0, 1]
    assert second['evaluation_executed_on_rank'] == 0
    assert metadata['checkpoint_final'] == second['checkpoint_final']
    assert Path(metadata['checkpoint_final']).is_file()
    sidecar = torch.load(Path(metadata['checkpoint_final']).with_suffix('.distributed.pt'), weights_only=False)
    assert sidecar['world_size'] == 2 and not sidecar['exact_resume_supported']
    assert [row['rank'] for row in sidecar['ranks']] == [0, 1]
    assert sidecar['ranks'][0]['sampled_instance_ids'] != sidecar['ranks'][1]['sampled_instance_ids']
    assert all(row['optimizer_steps'] == 1 for row in sidecar['ranks'])
    assert all(row['pool']['class'] == 'RankedPool' for row in sidecar['ranks'])
    rank_directory = Path(metadata['checkpoint_final']).parents[1] / 'rank_1/seed_31'
    assert not list(rank_directory.glob('*.pt')), 'Only primary rank writes model checkpoints'
    for rank in (0, 1):
        records = [json.loads(line) for line in (directory / 'monitoring' / f'monitor_rank_{rank}.jsonl').read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record['global_num_envs'] == 4 and record['global_trajectories_per_rollout'] == 8
        assert record['optimizer_attempts_epoch'] == 1 and record['optimizer_steps_epoch'] == 1
        assert record['parameter_sync_checksum_max_diff'] == record['amp_scale_rank_max_diff'] == 0
    for epoch in (0, 1):
        exports = [json.loads(line) for line in (directory / 'evaluations' / f'epoch_{epoch:04d}.jsonl').read_text().splitlines()]
        assert len(exports) == 8
        assert all(row['route_validation']['problem_type'] == problem for row in exports)
        assert all(row['feasibility_source'] == f'environment_success_and_independent_{problem}_route_validation' for row in exports)


def test_original_evrptw_eval_exposes_authoritative_energy_discrepancy(tmp_path, original_archive):
    directory = tmp_path / 'ev_energy'
    metadata, rows = _run_original(directory, original_archive, initial_eval=True, problem='evrptw', hostile_energy=True)
    assert metadata['state'] == 'completed'
    assert float(rows[0]['eval_environment_feasible_rate']) > 0.
    assert float(rows[0]['eval_feasible_rate']) == 0.
    exports = [json.loads(line) for line in (directory / 'evaluations/epoch_0000.jsonl').read_text().splitlines()]
    assert any(row['environment_feasible'] and not row['independently_valid'] for row in exports)
    assert all(row['route_validation']['energy_source'] == 'provided_energy_matrix_kwh' for row in exports)


class _SyncFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2.]))
        self.optional = torch.nn.Parameter(torch.tensor([.5]))
        self.unused = torch.nn.Parameter(torch.tensor([3.]))


def _original_sync_worker(rank, rendezvous, output, source_root):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    context = adapter._distributed.support.DistributedContext(rank=rank, world_size=2, device='cpu')
    runtime = adapter._distributed.OriginalDistributedRuntime(
        {'data': {'train_dataset_path': '/only-for-protocol-test'}, 'training': {}}, 31, 'cpu', output, context=context)
    original, _ = adapter._import_original(Path(source_root))
    model = _SyncFixture()
    with torch.no_grad():
        model.weight.add_(rank * 10.)
    context.broadcast_module(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    loss = model.weight.sum() if rank == 0 else model.weight.square().sum() + model.optional.square().sum()
    loss.backward()
    expected = _SyncFixture()
    ((expected.weight.sum() + expected.weight.square().sum() + expected.optional.square().sum()) / 2).backward()
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=.01, weight_decay=0.)
    original._optimizer_step(expected_optimizer, expected, 1., None, False)
    runtime.optimizer_step(original._optimizer_step, optimizer, model, 1., None, False)
    for parameter, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(parameter, reference, rtol=0., atol=0.)
    assert model.unused.grad is None
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    before = model.weight.detach().clone()
    if rank == 0:
        scaler.scale(model.weight.sum() * float('inf')).backward()
    runtime.optimizer_step(original._optimizer_step, optimizer, model, 1., scaler, True)
    assert torch.equal(model.weight, before) and scaler.get_scale() == 64.
    assert context.amp_skipped_steps == 1
    optimizer.zero_grad(set_to_none=True)
    if rank == 1:
        scaler.scale(model.weight.sum()).backward()
    runtime.optimizer_step(original._optimizer_step, optimizer, model, 1., scaler, True)
    weights = [torch.zeros_like(model.weight) for _ in range(2)]
    dist.all_gather(weights, model.weight.detach())
    assert torch.equal(weights[0], weights[1])
    assert context.optimizer_steps == 2
    optimizer.zero_grad(set_to_none=True)
    runtime.optimizer_step(original._optimizer_step, optimizer, model, 1., scaler, True)
    assert runtime.empty_gradient_steps == 1
    Path(output, f'rank{rank}.ok').write_text('ok')
    dist.destroy_process_group()


def test_original_optimizer_sync_unused_gradients_and_amp_overflow(tmp_path, original_archive):
    torch.multiprocessing.spawn(_original_sync_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path), str(original_archive)), nprocs=2, join=True)
    assert (tmp_path / 'rank0.ok').exists() and (tmp_path / 'rank1.ok').exists()


def test_original_evrptw_slppo_expert_and_incumbent_state_are_rank_local(tmp_path, original_archive):
    directory = tmp_path / 'dual_ev_slppo'
    metadata, rows = _run_original(directory, original_archive, initial_eval=True, world_size=2,
                                   problem='evrptw', method='sl_ppo')
    assert metadata['final_parameter_max_abs_difference'] == 0.
    assert metadata['optimizer_steps'] == 20 and [int(row['epoch']) for row in rows] == [0, 1]
    state = torch.load(Path(metadata['checkpoint_final']).with_suffix('.distributed.pt'), weights_only=False)
    assert all('expert_rng' in rank and rank['policy_best_objectives'] is not None for rank in state['ranks'])
    assert state['ranks'][0]['expert_rng'] != state['ranks'][1]['expert_rng']
    assert all(rank['sampled_instance_ids'] for rank in state['ranks'])
    assert all(rank['attempted_optimizer_steps'] == rank['optimizer_steps'] == 20 for rank in state['ranks'])
    assert all(rank['last_epoch_metrics']['optimizer_attempts_epoch'] == 20 for rank in state['ranks'])


def test_static_expert_storage_shares_only_equal_same_instance_readonly_arrays():
    storage = adapter._distributed._storage.StaticExpertStorage()
    first = {'edge_distance': np.eye(3), 'current_load': np.array([1.])}
    storage.observe('one', first)
    saved_first = {key: storage.asarray(value).copy() for key, value in first.items()}
    second = {'edge_distance': np.eye(3), 'current_load': np.array([1.])}
    storage.observe('one', second)
    saved_second = {key: storage.asarray(value).copy() for key, value in second.items()}
    assert saved_first['edge_distance'] is saved_second['edge_distance']
    assert not saved_first['edge_distance'].flags.writeable
    assert not np.shares_memory(saved_first['current_load'], saved_second['current_load'])
    assert saved_first['current_load'].flags.writeable
    second['edge_distance'][0, 0] = 8.
    assert saved_first['edge_distance'][0, 0] == 1.
    # A nominally static feature changes: preserve values and fall back to copying.
    changed = storage.asarray(second['edge_distance']).copy()
    assert changed[0, 0] == 8. and changed.flags.writeable
    assert storage.metrics()['changed_static_fields_preserved_without_sharing'] == 1
    storage.observe('two', first)
    other = storage.asarray(first['edge_distance']).copy()
    assert other is not saved_first['edge_distance']
    assert len(storage.current) == 1, 'Only the current static input is retained, never every expert step'
    assert storage.metrics()['saved_static_bytes'] > 0


def test_original_dual_slppo_static_storage_preserves_all_weights_and_losses(tmp_path, original_archive):
    original_dir, shared_dir = tmp_path / 'storage_original', tmp_path / 'storage_shared'
    original, original_rows = _run_original(original_dir, original_archive, initial_eval=True,
        world_size=2, problem='evrptw', method='sl_ppo', share_static=False)
    shared, shared_rows = _run_original(shared_dir, original_archive, initial_eval=True,
        world_size=2, problem='evrptw', method='sl_ppo', share_static=True)
    left = torch.load(original['checkpoint_final'], map_location='cpu', weights_only=False)
    right = torch.load(shared['checkpoint_final'], map_location='cpu', weights_only=False)
    assert left['model_state_dict'].keys() == right['model_state_dict'].keys()
    assert all(torch.equal(left['model_state_dict'][key], right['model_state_dict'][key]) for key in left['model_state_dict'])
    assert original['optimizer_steps'] == shared['optimizer_steps'] == 20
    assert shared['expert_observation_storage']['saved_static_bytes'] > 0
    assert shared['expert_observation_storage']['expert_samples_removed'] == 0
    assert not shared['expert_observation_storage']['forward_encoding_cache']
    assert shared['expert_observation_storage']['changed_static_fields_preserved_without_sharing'] == 0
    def logs(directory, rank):
        root = original_archive / 'results/logs/Cus_3_CS_1' / directory.name
        if rank:
            root = root / f'rank_{rank}'
        return list(csv.DictReader((root / 'seed_31/train_log.csv').open()))
    for rank in (0, 1):
        row_left, row_right = logs(original_dir, rank)[0], logs(shared_dir, rank)[0]
        for key in row_left:
            if ('loss' in key or key in ('entropy', 'approx_kl', 'clip_fraction', 'reward_mean', 'optimizer_steps')):
                assert row_left[key] == row_right[key], (rank, key)
    for row_left, row_right in zip(original_rows, shared_rows):
        for key in ('eval_feasible_rate', 'eval_avg_objective_distance_km', 'eval_environment_feasible_rate', 'eval_independent_valid_rate'):
            assert row_left[key] == row_right[key]
