"""Two-rank scratch budgets and pair ownership, with no GPU jobs or weight loads."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_evrptw_dual_scratch as launch
import run_reward_norm_comparison as shared
from test_scratch_comparison import prepared as scratch_prepared
from test_scratch_comparison import base_config


def cfg(tmp_path, variant):
    return launch.build_config(base_config(), variant=variant, output=tmp_path / variant,
                               run_name='DUAL_' + variant.upper(), data_root=tmp_path / 'AAAI_Dataset')


@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_two_rank_defaults_have_same_global_budget_and_complete_evaluation(tmp_path, variant):
    config = cfg(tmp_path, variant)
    train, evaluation, protocol = config['training'], config['evaluation'], config['experiment_protocol']
    assert config['data']['problem_type'] == 'evrptw'
    assert config['data']['num_customers'] == 100 and config['data']['num_charging_stations'] == 20
    assert train['num_envs_per_gpu'] == 32 and protocol['world_size'] == 2
    assert protocol['global_instances_per_rollout'] == 64
    assert train['n_traj'] == 50 and protocol['global_trajectories_per_rollout'] == 3200
    assert train['ppo_update_epochs'] == 5 and train['num_minibatches'] == 4
    assert train['gradient_accumulation_steps'] == 1
    assert protocol['global_instances_per_optimizer_step'] == 16
    assert train['learning_rate'] == 1e-4 and train['ent_coef'] == .01
    assert config['offline']['sl_coef'] == .5
    assert train['rollout_steps'] == evaluation['eval_max_steps'] == 512
    assert train['rollout_steps'] >= config['env']['max_steps_factor'] * 121
    assert train['epochs'] == 1500 and evaluation['eval_interval'] == 50
    assert evaluation['eval_before_training'] and evaluation['eval_n_traj'] == 50
    assert evaluation['eval_batch_size'] == 16
    assert 'eval_limit' not in evaluation and 'eval_num_batches' not in evaluation
    assert '/dataset/evrptw/val/Cus100' in evaluation['eval_path']
    assert '/dataset/evrptw/train/Cus100' in config['offline']['expert_solution_path']
    assert not config['offline']['use_priority_sampler']
    for section in ('offline', 'training'):
        assert not any(value and 'checkpoint' in key and any(s in key for s in ('init', 'resume', 'reference')) and not key.endswith('_strict') for key, value in config[section].items())
    if variant == 'optimized':
        assert train['require_complete_feasible_rollouts'] is False
        assert train['reward_norm_mode'] == 'physical_shared_popart'
        assert config['env']['reward_contract'] == 'strict_distance'
        assert config['offline']['exploration_instances'] == 4
        assert protocol['search_budget']['max_global_trajectories'] == 64
    else:
        assert 'reward_contract' not in config['env']
        assert config['offline']['original_share_static_expert_observations'] is True
        assert 'reward_norm_mode' not in train
        assert protocol['original_source']['commit'] == launch.scratch.ORIGINAL_COMMIT
    preflight = launch.preflight_config(config, tmp_path / 'preflight')
    for key in ('num_envs_per_gpu', 'n_traj', 'rollout_steps', 'ppo_update_epochs', 'num_minibatches', 'ppo_step_chunk_size'):
        assert preflight['training'][key] == train[key]
    assert preflight['training']['epochs'] == 2
    assert preflight['evaluation']['eval_limit'] == 16
    assert preflight['evaluation']['eval_batch_size'] == evaluation['eval_batch_size']
    assert preflight['evaluation']['eval_n_traj'] == evaluation['eval_n_traj']


@pytest.fixture
def dual(scratch_prepared, monkeypatch, request):
    state = scratch_prepared
    variant = getattr(request, 'param', 'optimized')
    root = state.root
    data = root.parent / 'AAAI_Dataset'
    for split, count in [('train', 5000), ('val', 1000)]:
        folder = data / 'dataset/evrptw' / split / 'Cus100'
        folder.mkdir(parents=True)
        (folder / 'instances.pkl').write_bytes(b'not loaded during preparation')
        (folder / 'metadata.json').write_text(json.dumps(dict(num_instances=count, num_customers=100, num_charging_stations=20)))
        (folder / ('expert_solutions.csv' if split == 'train' else 'gurobi_summary.csv')).write_text('instance_id\nfixture_0\nfixture_1\n')
    monkeypatch.setattr(launch, 'CODE_ROOT', root)
    args = launch.make_parser().parse_args(['--variant', variant, '--prepare-only', '--run-id', 'DUAL_FIXTURE',
        '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml'), '--data-root', str(data)])
    run = launch.prepare(args)
    return SimpleNamespace(run=run, manifest=json.loads((run / 'manifest.json').read_text()),
                           variant=variant, parent=state, args=args)


@pytest.mark.parametrize('dual', ['original', 'optimized'], indirect=True)
def test_cpu_prepare_never_loads_weights_or_launches_and_freezes_torchrun_command(dual):
    # Reused fixture makes torch.load/load_initialization fail if called and
    # reports no hardware. No dataset pickle or large assets are copied.
    manifest = dual.manifest
    assert manifest['hardware_at_prepare'] is None
    assert not dual.parent.launches
    assert manifest['gpus'] == [0, 1]
    assert manifest['protocol']['global_batch'] == 64
    assert manifest['protocol']['n_traj'] == 50
    assert manifest['protocol']['ppo_update_epochs'] == 5
    assert manifest['protocol']['world_size_per_arm'] == 2
    assert manifest['protocol']['expert_rows_at_prepare'] == 2
    frozen = Path(manifest['code_root'])
    assert not (frozen / 'assets').exists()
    spec = manifest['arms'][dual.variant]
    assert spec['required_validation_epochs'] == list(range(0, 1501, 50))
    assert spec['preflight']['required_validation_epochs'] == [2]
    for stage in (spec, spec['preflight']):
        command = stage['command']
        assert 'torch.distributed.run' in command
        assert '--nproc-per-node=2' in command and '--max-restarts=0' in command
        assert '/Cus_100_CS_20/' in stage['checkpoint_dir']
        assert stage['validation_instances'] == (16 if stage is spec['preflight'] else 1000)
        if dual.variant == 'original':
            source = manifest['additional_sources']['original']
            assert source['source']['git_commit'] == launch.scratch.ORIGINAL_COMMIT
            assert stage['code_root'] == source['code_root']
            assert str(frozen / 'scripts/run_original_scratch.py') in command
            assert '--source-root' in command and source['code_root'] in command
        else:
            assert stage['code_root'] == str(frozen)
            assert '--module' in command and 'offline2online.train' in command
    shared.verify_manifest(manifest)


class Lock:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True


def cards():
    return {index: dict(index=index, uuid=f'GPU-{index}', name='fixture2080', has_compute_process=False,
                        used_mib=0, total_mib=11264, utilization=0) for index in (0, 1)}


@pytest.mark.parametrize('failure', ['busy', 'exception'])
def test_second_lock_failure_releases_first_without_holding_half_pair(monkeypatch, failure):
    first = Lock(); calls = []
    def acquire(uuid):
        calls.append(uuid)
        if len(calls) == 1:
            return first
        if failure == 'exception':
            raise OSError('lock failure')
        return None
    monkeypatch.setattr(shared, 'acquire_gpu_lock', acquire)
    if failure == 'exception':
        with pytest.raises(OSError):
            launch.lock_gpu_pair(cards(), [1, 0])
    else:
        assert launch.lock_gpu_pair(cards(), [1, 0]) is None
    assert calls == ['GPU-0', 'GPU-1']
    assert first.closed


def supervisor_mocks(monkeypatch, *, returncode=1):
    owned = [Lock(), Lock()]
    launches = []
    monkeypatch.setattr(shared, 'gpu_snapshot', cards)
    monkeypatch.setattr(launch, 'lock_gpu_pair', lambda *args: owned)
    monkeypatch.setattr(launch.time, 'sleep', lambda _: None)
    monkeypatch.setattr(shared, 'refresh_progress', lambda *args, **kwargs: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: launches.append((args, kwargs)) or SimpleNamespace(pid=4321, poll=lambda: returncode))
    return owned, launches


def test_failed_preflight_stops_both_rank_job_without_starting_formal_training(dual, monkeypatch):
    owned, launches = supervisor_mocks(monkeypatch, returncode=1)
    launch.supervise(dual.run)
    state = json.loads((dual.run / 'status.json').read_text())
    assert len(launches) == 1
    assert state['state'] == state['arms'][dual.variant]['state'] == 'failed'
    assert state['arms'][dual.variant]['stage'] == 'preflight'
    assert all(lock.closed for lock in owned)
    assert launches[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert launches[0][1]['start_new_session']


def test_pair_recheck_error_releases_both_locks_and_starts_no_process(dual, monkeypatch):
    owned, launches = supervisor_mocks(monkeypatch)
    polls = 0
    def snapshot():
        nonlocal polls
        polls += 1
        if polls > dual.manifest['idle_checks']:
            raise OSError('GPU recheck failed after pair acquisition')
        return cards()
    monkeypatch.setattr(shared, 'gpu_snapshot', snapshot)
    with pytest.raises(OSError, match='GPU recheck'):
        launch.supervise(dual.run)
    assert not launches
    assert all(lock.closed for lock in owned)


def test_supervisor_exception_waits_for_job_exit_before_releasing_pair(dual, monkeypatch):
    owned, launches = supervisor_mocks(monkeypatch)
    state = {'spawned': False, 'alive': True, 'terminate_calls': 0}
    def spawn(*args, **kwargs):
        state['spawned'] = True
        launches.append((args, kwargs))
        return SimpleNamespace(pid=4321, poll=lambda: None if state['alive'] else -15)
    monkeypatch.setattr(launch.subprocess, 'Popen', spawn)
    def progress(*args, **kwargs):
        raise RuntimeError('fatal progress failure')
    monkeypatch.setattr(shared, 'refresh_progress', progress)
    def terminate(process):
        assert not any(lock.closed for lock in owned), 'Released GPU ownership while workers remain alive'
        state['terminate_calls'] += 1
        # SIGTERM is asynchronous: the helper does not wait. The supervisor
        # must continue polling, not release ownership after the first signal.
        if state['terminate_calls'] >= 2:
            state['alive'] = False
    monkeypatch.setattr(shared, 'terminate_group', terminate)
    with pytest.raises(RuntimeError, match='fatal progress failure'):
        launch.supervise(dual.run)
    assert not state['alive']
    assert all(lock.closed for lock in owned)


@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_shell_defaults_launch_and_prepare_only_overrides(tmp_path, variant):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/env python3\nimport sys\nprint("\\n".join(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake))
    for name in ('SEED', 'GPUS', 'RUN_ID', 'DATA_ROOT'):
        env.pop(name, None)
    script = ROOT / f'scripts/run_evrptw100_{variant}_dual.sh'
    command = subprocess.check_output(['bash', str(script)], env=env, text=True).splitlines()
    assert '--launch' in command
    assert command[command.index('--variant') + 1] == variant
    prepared = subprocess.check_output(['bash', str(script), '--prepare-only', '--gpus', '2,3'], env=env, text=True).splitlines()
    assert '--launch' not in prepared and '--prepare-only' in prepared
    assert prepared[-2:] == ['--gpus', '2,3']


def test_busy_second_gpu_does_not_reserve_idle_first_gpu(dual, monkeypatch):
    occupied = cards()
    occupied[1]['has_compute_process'] = True
    handlers = {}
    monkeypatch.setattr(shared, 'gpu_snapshot', lambda: occupied)
    monkeypatch.setattr(launch, 'lock_gpu_pair', lambda *a: pytest.fail('Reserved an incomplete idle pair'))
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('Started while one GPU is busy'))
    monkeypatch.setattr(launch.signal, 'signal', lambda signum, callback: handlers.__setitem__(signum, callback))
    monkeypatch.setattr(launch.time, 'sleep', lambda _: handlers[launch.signal.SIGTERM]())
    launch.supervise(dual.run)
    state = json.loads((dual.run / 'status.json').read_text())
    assert state['state'] == 'interrupted'


def test_transient_gpu_query_does_not_terminate_live_torchrun(dual, monkeypatch):
    owned, launches = supervisor_mocks(monkeypatch)
    state = {'spawned': False, 'queries_after_spawn': 0, 'polls': 0}
    def spawn(*args, **kwargs):
        launches.append((args, kwargs)); state['spawned'] = True
        def poll():
            state['polls'] += 1
            return 1 if state['polls'] >= 3 else None
        return SimpleNamespace(pid=4321, poll=poll)
    monkeypatch.setattr(launch.subprocess, 'Popen', spawn)
    def snapshot():
        if state['spawned']:
            state['queries_after_spawn'] += 1
            if state['queries_after_spawn'] == 1:
                raise OSError('temporary nvidia-smi query error')
        return cards()
    monkeypatch.setattr(shared, 'gpu_snapshot', snapshot)
    monkeypatch.setattr(shared, 'terminate_group', lambda *a: pytest.fail('Query error killed a live job'))
    launch.supervise(dual.run)
    assert state['queries_after_spawn'] >= 2
    assert len(launches) == 1 and all(lock.closed for lock in owned)


@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_vrptw_dual_uses_requested_task_seed_and_fixed_five_passes(tmp_path, variant):
    config = launch.build_config(base_config(), variant=variant, output=tmp_path / variant,
        run_name='VRPTW_DUAL', data_root=tmp_path / 'AAAI_Dataset', task='vrptw', seed=3011,
        chunk_size=32, expert_chunk_size=128)
    train, ev, protocol = config['training'], config['evaluation'], config['experiment_protocol']
    assert config['data']['problem_type'] == 'vrptw'
    assert config['data']['num_charging_stations'] == 0
    assert '/dataset/vrptw/train/Cus100' in config['data']['train_dataset_path']
    assert '/dataset/vrptw/val/Cus100' in ev['eval_path']
    assert train['rollout_steps'] == ev['eval_max_steps'] == 201
    assert train['ppo_update_epochs'] == 5 and train['target_kl'] is None
    assert train['num_envs_per_gpu'] == 32 and train['n_traj'] == 50
    assert train['epochs'] == 1500 and ev['eval_seed'] == 17003011
    assert protocol['task'] == 'vrptw100' and protocol['global_instances_per_rollout'] == 64
    assert protocol['implementation'] == ('legacy' if variant == 'original' else 'explore')
    launch.scratch.assert_scratch(config)
    if variant == 'optimized':
        assert train['post_init_seed'] == 3011
        assert config['offline']['branch_exploration_enabled']
        assert config['offline']['exploration_interval'] == 5
        # The short preflight must exercise search, even before epoch five.
        preflight = launch.preflight_config(config, tmp_path / 'preflight')
        assert preflight['offline']['exploration_interval'] == 1
        assert train['require_complete_feasible_rollouts']


@pytest.mark.parametrize(('task', 'horizon'), [('evrptw', 512), ('vrptw', 201)])
@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_chunk_tuning_is_bounded_by_the_requested_task_horizon(tmp_path, task, horizon, variant):
    arguments = dict(variant=variant, output=tmp_path, run_name='CHUNK_TEST',
                     data_root=tmp_path, task=task)
    config = launch.build_config(base_config(), chunk_size=horizon, **arguments)
    assert config['training']['ppo_step_chunk_size'] == horizon
    assert config['experiment_protocol']['ppo_chunk_size'] == horizon
    if variant == 'original':
        override = next(item for item in config['experiment_protocol']['protocol_overrides']
                        if item['parameter'] == 'training.ppo_step_chunk_size')
        assert override['used'] == horizon
    with pytest.raises(ValueError, match=f'chunk-size must be <={horizon}'):
        launch.build_config(base_config(), chunk_size=horizon + 1, **arguments)


def test_dataset_default_discovers_both_supported_layouts_with_explicit_override(tmp_path, monkeypatch):
    root = tmp_path / 'repository' / 'CaliRoute'
    monkeypatch.setattr(launch, 'CODE_ROOT', root)
    nested = root.parent / 'AAAI_Dataset'
    adjacent = root.parent.parent / 'AAAI_Dataset'
    assert launch.make_parser().parse_args([]).data_root == nested
    (adjacent / 'dataset').mkdir(parents=True)
    assert launch.make_parser().parse_args([]).data_root == adjacent
    (nested / 'dataset').mkdir(parents=True)
    assert launch.make_parser().parse_args([]).data_root == nested
    assert launch.make_parser().parse_args(['--data-root', str(tmp_path / 'custom')]).data_root == tmp_path / 'custom'


def add_prerequisite(dual, state='running'):
    previous = dual.run.parent / 'PREVIOUS'
    previous.mkdir()
    shared.write_json(previous / 'manifest.json', {})
    shared.write_json(previous / 'status.json', dict(state=state))
    manifest = dict(dual.manifest, after_runs=[str(previous)])
    shared.write_json(dual.run / 'manifest.json', manifest)
    return previous


def test_prerequisite_is_frozen_during_preparation(dual):
    dual.args.after_run = [dual.run]
    dual.args.run_id = 'FOLLOWUP'
    followup = launch.prepare(dual.args)
    manifest = json.loads((followup / 'manifest.json').read_text())
    assert manifest['after_runs'] == [str(dual.run)]
    assert manifest['protocol']['seed'] == dual.manifest['protocol']['seed']
    assert manifest['protocol']['global_batch'] == dual.manifest['protocol']['global_batch']
    assert not dual.parent.launches


def test_sequential_comparison_waits_without_reserving_idle_gpus(dual, monkeypatch):
    previous = add_prerequisite(dual)
    owned, launches = supervisor_mocks(monkeypatch)
    def acquire(*args):
        assert json.loads((previous / 'status.json').read_text())['state'] == 'completed'
        return owned
    monkeypatch.setattr(launch, 'lock_gpu_pair', acquire)
    waiting_observed = []
    def finish_previous(_):
        status = json.loads((dual.run / 'status.json').read_text())
        if status['state'] == 'waiting_dependency':
            waiting_observed.append(status['dependency_wait_reason'])
            assert not launches
            shared.write_json(previous / 'status.json', dict(state='completed'))
    monkeypatch.setattr(launch.time, 'sleep', finish_previous)
    launch.supervise(dual.run)
    assert len(waiting_observed) == len(launches) == 1
    assert all(lock.closed for lock in owned)


@pytest.mark.parametrize('state', ['failed', 'interrupted'])
def test_failed_prerequisite_never_launches_or_acquires_gpus(dual, monkeypatch, state):
    previous = add_prerequisite(dual)
    monkeypatch.setattr(shared, 'gpu_snapshot', cards)
    monkeypatch.setattr(launch, 'lock_gpu_pair', lambda *a: pytest.fail('Acquired GPU for dependent failure'))
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('Launched after dependent failure'))
    monkeypatch.setattr(launch.time, 'sleep',
                        lambda _: shared.write_json(previous / 'status.json', dict(state=state)))
    launch.supervise(dual.run)
    result = json.loads((dual.run / 'status.json').read_text())
    assert result['state'] == 'failed'
    assert f'state={state}' in result['error']
