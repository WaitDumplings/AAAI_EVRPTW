#!/usr/bin/env python3
"""Portable, frozen-source VRPTW100 reward/normalization 2x2 comparison.

Python prepares by default; --launch detaches a supervisor which waits for idle
GPUs. The shell wrapper launches by default. Existing jobs are never stopped.
Only validation is run. Every arm uses one GPU and identical rollout batches.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
from run_cus100_finetune import refresh_progress
from run_plugin_comparison import terminate_group
from run_slppo_comparison import digest, now, read_csv, write_json
from reward_norm_initialization import load_initialization
from reward_norm_extension import positive_integer, validation_epochs

SOURCE_RUN = 'VRPTW100_UPDATES3456_S3009_E500_20261006'
DEFAULT_CHECKPOINT = Path('assets/reward_norm/vrptw100_update5_epoch0300.pt')
ARMS = {
    'baseline': (.99, 'legacy'),
    'reward': (1., 'legacy'),
    'normalization': (.99, 'physical_shared_popart'),
    'combined': (1., 'physical_shared_popart'),
}
TERMINAL_STATES = {'completed', 'failed', 'interrupted'}


def parse_gpus(value):
    try:
        gpus = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise ValueError('GPU IDs must be comma-separated integer indices') from exc
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Choose one to four distinct nonnegative GPU indices')
    return gpus


def checkpoint_units(payload):
    env = payload.get('config', {}).get('env', {})
    reward = env.get('reward_distance_scale_km')
    observation = env.get('observation_distance_scale_km', reward)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (reward, observation)):
        raise ValueError('Initialization must record positive reward_distance_scale_km; observation scale defaults to that saved reward unit')
    return dict(reward_distance_scale_km=float(reward), observation_distance_scale_km=float(observation))


def build_arm(base, *, arm, output, run_name, init_checkpoint, data_root, seed, units, epochs=80, chunk_size=18, eval_interval=50):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    if base['data']['problem_type'] != 'vrptw' or int(base['data']['num_customers']) != 100:
        raise ValueError('This comparison requires a VRPTW100 source configuration')
    positive_integer(epochs, 'epochs')
    positive_integer(eval_interval, 'eval_interval')
    if not 1 <= chunk_size <= 201:
        raise ValueError('chunk_size must be between 1 and 201')
    cfg = copy.deepcopy(base)
    cfg.pop('normalization', None)
    cfg['run_name'] = run_name
    cfg['env'].update(units)
    mode = cfg['env'].get('reward_distance_scale_mode', '')
    if mode.startswith('dataset_'):
        cfg['env']['reward_distance_scale_mode'] = mode[len('dataset_'):]
    train_path = Path(data_root) / 'dataset/vrptw/train/Cus100'
    val_path = Path(data_root) / 'dataset/vrptw/val/Cus100'
    cfg['data'].update(train_dataset_path=str(train_path), num_charging_stations=0,
                       train_sample_mode='shuffle_cycle', async_instance_prefetch=False)
    for section in ('training', 'offline'):
        for key in list(cfg[section]):
            if key.startswith('resume_'):
                del cfg[section][key]
    gamma, mode = ARMS[arm]
    cfg['training'].update(epochs=epochs, num_envs_per_gpu=64, n_traj=50,
        rollout_steps=201, num_minibatches=4, gradient_accumulation_steps=1,
        ppo_update_epochs=5, ppo_step_chunk_size=chunk_size, gamma=gamma,
        reward_norm_mode=mode, learning_rate=1e-5, lr_schedule='constant',
        lr_warmup_epochs=0, lr_min=1e-5, ent_coef=.002,
        entropy_initial_coef=.002, entropy_final_coef=.002,
        amp_init_scale=1024., post_init_seed=seed,
        monitor_output_dir=str(output / 'monitoring'))
    cfg['offline'].update(init_checkpoint_path=str(init_checkpoint), init_checkpoint_strict=True,
        expert_dataset_path=str(train_path), expert_solution_path=str(train_path / 'expert_solutions.csv'),
        sl_coef=.35, use_priority_sampler=False)
    cfg['evaluation'].update(eval_path=str(val_path), gurobi_summary_path=str(val_path / 'gurobi_summary.csv'),
        eval_interval=eval_interval, eval_before_training=True, eval_n_traj=50, eval_batch_size=32,
        eval_max_steps=201, eval_decode_mode='sample', eval_save_routes=True,
        eval_seed=17000000 + seed, eval_output_dir=str(output / 'evaluations'))
    # A partial validation limit inherited from another launch must not survive.
    for key in ('eval_limit', 'eval_num_batches'):
        cfg['evaluation'].pop(key, None)
    cfg['experiment_protocol'] = dict(phase='reward_normalization_2x2', arm=arm,
        seed=seed, epochs=epochs, eval_interval=eval_interval, world_size=1, global_instances_per_rollout=64,
        global_trajectories_per_rollout=3200, global_instances_per_optimizer_step=16,
        ppo_update_epochs=5, attempted_optimizer_steps_per_epoch=20,
        initialization='shared mature checkpoint weights; fresh optimizer and replay',
        validation_instances=1000, selection_split='val', test_enabled=False,
        input_units='unchanged source graph inputs; reward and critic normalization are separate',
        sampling='uniform shuffle_cycle, common across arms; not an exact continuation of the priority-sampled old long run',
        comparison_scope='single-scale local fine-tuning screen, not full retraining evidence')
    return cfg


def build_preflight(cfg, output):
    result = copy.deepcopy(cfg)
    result['run_name'] += '_PREFLIGHT'
    result['training'].update(epochs=2, num_envs_per_gpu=4, n_traj=4, ppo_step_chunk_size=4,
        monitor_interval=1, monitor_output_dir=str(output / 'monitoring'))
    result['evaluation'].update(eval_before_training=False, eval_interval=2, eval_n_traj=4,
        eval_batch_size=4, eval_limit=4, eval_output_dir=str(output / 'evaluations'))
    result['experiment_protocol'].update(phase='gpu_preflight', epochs=2, eval_interval=2,
        global_instances_per_rollout=4, global_trajectories_per_rollout=16,
        global_instances_per_optimizer_step=1, validation_instances=4,
        comparison_scope='Pipeline and GPU smoke check only; excluded from formal comparison')
    return result


def successful_training(spec, detail):
    epochs = [int(row['epoch']) for row in read_csv(Path(spec['log_dir']) / 'train_log.csv')]
    evaluation = detail.get('latest_validation') or {}
    target = spec['epochs']
    expected_count = spec.get('validation_instances', 1000)
    validation_rows = read_csv(Path(spec['log_dir']) / 'eval_log.csv')
    completed_evaluations = {int(row['epoch']) for row in validation_rows
        if row.get('eval_status') == 'ok' and int(float(row.get('eval_num_instances', 0))) == expected_count}
    required = set(spec.get('required_validation_epochs', []))
    return (required <= completed_evaluations and epochs == list(range(1, target + 1))
        and int(evaluation.get('epoch', -1)) == target
        and evaluation.get('eval_status') == 'ok'
        and int(float(evaluation.get('eval_num_instances', 0))) == spec.get('validation_instances', 1000)
        and (Path(spec['checkpoint_dir']) / 'checkpoint_final.pt').is_file())


def gpu_snapshot():
    def query(options):
        result = subprocess.run(['nvidia-smi', *options, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=10)
        return [[v.strip() for v in row] for row in csv.reader(result.stdout.splitlines()) if row]
    cards = query(['--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total'])
    computes = query(['--query-compute-apps=gpu_uuid,pid'])
    busy = {row[0] for row in computes if len(row) >= 2}
    return {int(row[0]): dict(index=int(row[0]), uuid=row[1], name=row[2],
        utilization=int(row[3]), used_mib=int(row[4]), total_mib=int(row[5]),
        has_compute_process=row[1] in busy) for row in cards}


def validate_requested_gpus(gpus, cards):
    missing = sorted(set(gpus) - set(cards))
    if missing:
        raise ValueError(f'Requested GPU indices do not exist on this host: {missing}; available={sorted(cards)}')
    if len({cards[g]['name'] for g in gpus}) != 1:
        raise ValueError('Choose GPUs of one model per hardware comparison block')


def probe_requested_gpus(gpus):
    """Validate readable hardware now, but allow preparation on a CPU-only host."""
    try:
        cards = gpu_snapshot()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    validate_requested_gpus(gpus, cards)
    return cards


def idle_gpu(card, max_memory_mib=512, max_utilization=5):
    return (not card['has_compute_process'] and card['used_mib'] <= max_memory_mib
            and card['utilization'] <= max_utilization)


def prerequisite_busy(paths, gpu):
    """Include a previous experiment's test stage, not just its last training row."""
    for path in paths:
        try:
            status = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            return f'Cannot verify prerequisite {path}: {exc}'
        arms = status.get('arms', {})
        for detail in arms.values():
            if detail.get('gpu') == gpu and detail.get('state') not in TERMINAL_STATES:
                return f'Previous experiment still owns GPU {gpu}: {path}'
        if not arms and status.get('state') not in TERMINAL_STATES:
            return f'Previous experiment has not assigned its GPUs yet: {path}'
    return None


def local_live_prerequisite(path):
    """A copied status file on another host must not reserve this host's GPU."""
    try:
        record = json.loads(Path(path).read_text())
        pid = int(record['supervisor_pid'])
        if pid <= 0 or record.get('state') in TERMINAL_STATES:
            return False
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        tokens = [part.decode() for part in command if part]
        if not any(Path(token).name == 'run_vrptw_update_sweep.py' for token in tokens):
            return False
        index = tokens.index('--supervise')
        return Path(tokens[index+1]).resolve() == Path(path).resolve().parent
    except (OSError, ValueError, KeyError, IndexError, UnicodeError):
        return False


def acquire_gpu_lock(uuid):
    directory = Path(tempfile.gettempdir()) / f'caliroute-gpu-locks-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    lock = (directory / (uuid + '.lock')).open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def source_snapshot(destination, *, include_initialization_assets=True):
    """Copy code bytes; later git pulls cannot change this experiment's source."""
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=CODE_ROOT, text=True).strip())
    relative = CODE_ROOT.relative_to(repo)
    tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', str(relative)], cwd=repo)
    files = sorted(set(x.decode() for x in tracked.split(b'\0') if x))
    hashes = {}
    for name in files:
        src = repo / name
        rel = src.relative_to(CODE_ROOT)
        if rel.parts[0] == 'results':
            continue
        if not include_initialization_assets and (rel.parts[0] == 'assets' or src.suffix.lower() in {'.pt', '.pth', '.ckpt'}):
            continue
        if not src.is_file():
            raise ValueError(f'Missing source file: {src}')
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = digest(src)
        shutil.copy2(src, dst)
        if digest(src) != before or digest(dst) != before:
            raise ValueError(f'Source changed during snapshot: {src}')
        hashes[str(rel)] = before
    (destination / 'results').symlink_to(CODE_ROOT / 'results', target_is_directory=True)
    return dict(files=hashes, content_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True))


def verify_manifest(manifest):
    source = Path(manifest['code_root'])
    for relative, checksum in manifest['source']['files'].items():
        if digest(source / relative) != checksum:
            raise ValueError(f'Frozen source changed: {relative}')
    if manifest.get('initialization_mode') == 'scratch':
        if any(manifest.get(key) for key in ('init_checkpoint', 'init_checkpoint_sha256', 'source_init_checkpoint', 'source_init_epoch')):
            raise ValueError('A scratch manifest cannot contain a learned initialization')
    elif digest(Path(manifest['init_checkpoint'])) != manifest['init_checkpoint_sha256']:
        raise ValueError('Frozen initialization changed')
    # Independent historical code must be verified too, not just the launcher.
    roots = {str(source.resolve())}
    for name, item in manifest.get('additional_sources', {}).items():
        root = Path(item['code_root']).resolve()
        roots.add(str(root))
        for relative, checksum in item['source']['files'].items():
            if digest(root / relative) != checksum:
                raise ValueError(f'Frozen additional source changed: {name}/{relative}')
    for name, spec in manifest['arms'].items():
        for stage in (spec, spec.get('preflight', {})):
            if stage and str(Path(stage.get('code_root', spec.get('code_root', source))).resolve()) not in roots:
                raise ValueError(f'Unverified execution source for {name}')
    for key, item in manifest['inputs'].items():
        if digest(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Dataset/reference changed: {key}')
    for arm, spec in manifest['arms'].items():
        if spec.get('resume_checkpoint') and digest(Path(spec['resume_checkpoint'])) != spec['resume_checkpoint_sha256']:
            raise ValueError(f'Frozen resume checkpoint changed: {arm}')
        for stage in (spec, spec.get('preflight', {})):
            if stage and digest(Path(stage['config'])) != stage['config_sha256']:
                raise ValueError(f'Configuration changed: {arm}')
            if stage and manifest.get('initialization_mode') == 'scratch':
                if stage.get('resume_checkpoint'):
                    raise ValueError('Scratch execution cannot load a resume checkpoint')
                cfg = yaml.safe_load(Path(stage['config']).read_text())
                def check_scratch_mapping(node, prefix=''):
                    if isinstance(node, dict):
                        for key, value in node.items():
                            name = str(key)
                            if value and ('checkpoint' in name or name == 'pretrained_path') and any(token in name for token in ('init', 'initial', 'resume', 'reference', 'pretrained')) and not name.endswith('_strict'):
                                raise ValueError(f'Scratch execution forbids {prefix}{name}')
                            check_scratch_mapping(value, prefix + name + '.')
                    elif isinstance(node, list):
                        for value in node:
                            check_scratch_mapping(value, prefix)
                check_scratch_mapping(cfg)


def comparison_report(manifest, status):
    evaluations = {}
    for arm, spec in manifest['arms'].items():
        try:
            evaluations[arm] = {int(r['epoch']): r for r in read_csv(Path(spec['log_dir']) / 'eval_log.csv')
                                if r.get('eval_status') == 'ok'}
        except (OSError, ValueError, KeyError):
            evaluations[arm] = {}
    common = sorted(set.intersection(*(set(rows) for rows in evaluations.values())))
    metrics = ('eval_avg_objective_distance_km', 'eval_feasible_rate', 'eval_num_instances')
    aligned = {str(epoch): {arm: {key: row.get(key) for key in metrics}
                           for arm, rows in evaluations.items() for row in [rows[epoch]]} for epoch in common}
    epoch0 = []
    for rows in evaluations.values():
        try:
            value = float(rows[0]['eval_avg_objective_distance_km'])
            if math.isfinite(value):
                epoch0.append(value)
        except (KeyError, ValueError, TypeError):
            pass
    report = dict(updated_at_utc=now(), state=status['state'], protocol=manifest['protocol'],
        arms=status['arms'], matched_validation_epochs=aligned,
        initial_evaluation_consistent=(max(epoch0)-min(epoch0) <= 1e-4) if len(epoch0) == len(evaluations) and len(epoch0)>1 else None,
        scope='Compare raw mean km and feasibility at matching epochs; timing only within the same GPU model.')
    pairs = manifest['protocol'].get('initial_evaluation_pairs')
    if pairs is not None:
        initial = {}
        for arm, rows in evaluations.items():
            row = rows.get(0, {})
            try:
                distance = float(row['eval_avg_objective_distance_km'])
                feasibility = float(row['eval_feasible_rate'])
                count = int(float(row['eval_num_instances']))
                if math.isfinite(distance) and math.isfinite(feasibility):
                    initial[arm] = dict(distance_km=distance, feasible_rate=feasibility, num_instances=count)
            except (KeyError, ValueError, TypeError, OverflowError):
                pass
        checks = {}
        expected_count = manifest['protocol'].get('validation_instances', 1000)
        for left, right in pairs:
            key = f'{left}__{right}'
            a, b = initial.get(left), initial.get(right)
            checks[key] = (abs(a['distance_km'] - b['distance_km']) <= 1e-4
                and abs(a['feasible_rate'] - b['feasible_rate']) <= 1e-8
                and a['num_instances'] == b['num_instances'] == expected_count) if a and b else None
        report.update(initial_evaluation_consistency_scope=manifest['protocol'].get('initial_evaluation_consistency_scope', 'within_coordinate_mode_pairs_only'),
            initial_evaluation_pairs=checks, initial_validation_by_arm=initial,
            initial_evaluation_consistent=all(checks.values()) if checks and all(v is not None for v in checks.values()) else None,
            initialization_caveat=manifest['protocol'].get('initialization_caveat', 'Different coordinate modes may have different epoch-zero quality despite shared weights; compare absolute results and report migration cost.'))
        if 'legacy' in initial:
            base_distance = initial['legacy']['distance_km']
            report['initial_distance_delta_from_legacy_km'] = {arm: row['distance_km'] - base_distance for arm, row in initial.items()}
    return report


def supervise(experiment, *, stop_requested=None):
    manifest = json.loads((experiment / 'manifest.json').read_text())
    # Prevent launching two supervisors for one prepared experiment.
    supervisor_lock = (experiment / 'supervisor.lock').open('a+')
    fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((experiment / 'status.json').read_text())
    if previous['state'] != 'prepared':
        raise ValueError('Supervisor requires a fresh prepared experiment; automatic partial reruns are disabled')
    try:
        verify_manifest(manifest)
    except Exception as exc:
        write_json(experiment / 'status.json', dict(state='failed', error=f'Input verification: {exc}', finished_at_utc=now()))
        supervisor_lock.close()
        raise
    status = dict(state='waiting_gpu', started_at_utc=now(), supervisor_pid=os.getpid(),
        arms={arm: dict(state='queued', target_epochs=spec['epochs'], completed_training_epochs=0) for arm, spec in manifest['arms'].items()})
    active, streams, locks, idle_counts = {}, {}, {}, {gpu: 0 for gpu in manifest['gpus']}
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    def spawn_stage(arm, stage, gpu, card):
        spec = manifest['arms'][arm]
        if stage == 'preflight':
            spec = spec['preflight']
        stream = (Path(spec['output_dir']) / 'console.log').open('a', buffering=1)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(Path(spec['output_dir']) / 'numba_cache'))
        env.pop('EVRPTW_DB_ROOT', None)
        # Inherited PYTHONPATH must not import another checkout into a frozen arm.
        env.pop('PYTHONPATH', None)
        code_root = spec.get('code_root', manifest['arms'][arm].get('code_root', manifest['code_root']))
        try:
            process = subprocess.Popen(spec['command'], cwd=code_root, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            stream.close()
            raise
        active[arm], streams[arm] = process, stream
        detail = status['arms'][arm]
        detail.update(state='running', stage=stage, pid=process.pid, gpu=gpu,
            gpu_uuid=card['uuid'], gpu_model=card['name'], stage_started_at_utc=now())
        detail.setdefault('started_at_utc', now())
        print(f'{now()} {arm} {stage} GPU{gpu} pid={process.pid}', flush=True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            if stop_requested is not None and stop_requested():
                stopped = True
            for arm, process in list(active.items()):
                detail, formal = status['arms'][arm], manifest['arms'][arm]
                is_preflight = detail['stage'] == 'preflight'
                spec = formal['preflight'] if is_preflight else formal
                progress = detail.setdefault('preflight_progress', {}) if is_preflight else detail
                refresh_progress(spec, progress)
                progress['rank_monitors'] = {'0': progress.get('rank_monitors', {}).get('0')}
                code = process.poll()
                if code is None:
                    if stopped:
                        terminate_group(process)
                    continue
                try:
                    refresh_progress(spec, progress, finished=(code == 0))
                    success = code == 0 and successful_training(spec, progress)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    success = False
                    progress['completion_read_error'] = str(exc)
                progress.update(exit_code=code, finished_at_utc=now())
                streams.pop(arm).close()
                active.pop(arm)
                if is_preflight and success and not stopped:
                    progress['state'] = 'completed'
                    # Formal training starts from the common archive again, never preflight weights.
                    spawn_stage(arm, 'training', detail['gpu'], dict(uuid=detail['gpu_uuid'], name=detail['gpu_model']))
                    continue
                detail.update(exit_code=code, finished_at_utc=now())
                if stopped:
                    detail['state'] = 'interrupted'
                elif not success:
                    detail.update(state='failed', error=f'{detail["stage"]} exit={code}; require every epoch and full configured final validation')
                else:
                    detail['state'] = 'completed'
                if is_preflight:
                    progress['state'] = detail['state']
                locks.pop(arm).close()
            if stopped:
                for detail in status['arms'].values():
                    if detail['state'] == 'queued':
                        detail['state'] = 'interrupted'
            # Record hardware on every poll, including when all arms are running.
            # Reuse this sample for scheduling; only lock acquisition needs a recheck.
            cards = None
            try:
                cards = gpu_snapshot()
                validate_requested_gpus(manifest['gpus'], cards)
                status.pop('gpu_poll_warning', None)
                with (experiment / 'hardware.jsonl').open('a') as log:
                    log.write(json.dumps(dict(time_utc=now(), gpus=list(cards.values()),
                        active_stages={a:status['arms'][a]['stage'] for a in active}))+'\n')
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
                cards = None
                status['gpu_poll_warning'] = f'{type(exc).__name__}: {exc}'
            queued = [arm for arm, detail in status['arms'].items() if detail['state'] == 'queued']
            if queued and not stopped and cards is not None:
                try:
                    occupied = {status['arms'][arm]['gpu'] for arm in active}
                    status['gpu_availability'] = {}
                    for gpu in manifest['gpus']:
                        blocked = prerequisite_busy(manifest['wait_for_experiments'], gpu)
                        available = gpu not in occupied and not blocked and idle_gpu(cards[gpu])
                        idle_counts[gpu] = idle_counts[gpu]+1 if available else 0
                        status['gpu_availability'][str(gpu)] = dict(card=cards[gpu], prerequisite=blocked,
                            consecutive_idle_checks=idle_counts[gpu], required_idle_checks=manifest['idle_checks'])
                        if not queued or idle_counts[gpu] < manifest['idle_checks']:
                            continue
                        lock = acquire_gpu_lock(cards[gpu]['uuid'])
                        if lock is None:
                            continue
                        # Recheck after acquiring our cooperative lock. External jobs are never killed.
                        if not idle_gpu(gpu_snapshot()[gpu]):
                            lock.close(); idle_counts[gpu] = 0; continue
                        arm = queued.pop(0)
                        locks[arm] = lock
                        try:
                            stage = 'training' if manifest['arms'][arm].get('restored_through_epoch') else 'preflight'
                            spawn_stage(arm, stage, gpu, cards[gpu])
                        except BaseException:
                            locks.pop(arm).close()
                            raise
                except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                    status['gpu_poll_warning'] = f'{type(exc).__name__}: {exc}'
            done = all(detail['state'] in TERMINAL_STATES for detail in status['arms'].values())
            status['state'] = ('interrupted' if stopped else 'completed' if all(d['state']=='completed' for d in status['arms'].values())
                else 'failed') if done else 'running' if active else 'waiting_gpu'
            status['updated_at_utc'] = now()
            if done:
                status['finished_at_utc'] = now()
            write_json(experiment / 'status.json', status)
            write_json(experiment / 'comparison.json', comparison_report(manifest, status))
            if done:
                break
            time.sleep(manifest['poll_seconds'])
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}', finished_at_utc=now())
        write_json(experiment / 'status.json', status)
        raise
    finally:
        while any(process.poll() is None for process in active.values()):
            for process in active.values():
                terminate_group(process)
            time.sleep(.25)
        for stream in streams.values():
            stream.close()
        for lock in locks.values():
            lock.close()
        supervisor_lock.close()


def prepare(args, *, arm_definitions=None, arm_builder=None, default_checkpoint=None,
            protocol_overrides=None, prerequisite_source_run=SOURCE_RUN, preflight_builder=None):
    """Prepare one frozen experiment; optional hooks support input-only screens.

    Default arguments preserve the historical reward/norm launcher protocol.
    Supervision is manifest driven and shared by all supported factorizations.
    """
    definitions = ARMS if arm_definitions is None else arm_definitions
    builder = build_arm if arm_builder is None else arm_builder
    preflight_factory = build_preflight if preflight_builder is None else preflight_builder
    bundled_checkpoint = DEFAULT_CHECKPOINT if default_checkpoint is None else default_checkpoint
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or not arms or any(arm not in definitions for arm in arms):
        raise ValueError('arms must be distinct names from: '+','.join(definitions))
    gpus = parse_gpus(args.gpus)
    hardware_at_prepare = probe_requested_gpus(gpus)
    positive_integer(args.epochs, 'epochs')
    positive_integer(args.eval_interval, 'eval_interval')
    if not 1 <= args.poll_seconds <= 60 or not 2 <= args.idle_checks <= 10:
        raise ValueError('poll_seconds must be 1..60 and idle_checks 2..10')
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('run-id must be a fresh directory name')
    base = yaml.safe_load(args.base_config.resolve().read_text())
    checkpoint = args.init_checkpoint.resolve()
    bundled = checkpoint == (CODE_ROOT / bundled_checkpoint).resolve()
    payload, initialization = load_initialization(checkpoint, args.expected_init_epoch,
        metadata_path=checkpoint.with_suffix('.json') if bundled else None)
    units = checkpoint_units(payload)
    del payload
    data_root = args.data_root.resolve()
    required_inputs = {}
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')), ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        for name in names:
            path = data_root / 'dataset/vrptw' / split / 'Cus100' / name
            required_inputs[f'vrptw/{split}/Cus100/{name}'] = path
    missing = [str(path) for path in required_inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing local VRPTW100 data/reference files (datasets are not in Git):\n  '
            + '\n  '.join(missing) + '\nPlace AAAI_Dataset beside CaliRoute or pass --data-root /path/to/AAAI_Dataset.')
    for split in ('train', 'val'):
        path = data_root / 'dataset/vrptw' / split / 'Cus100/metadata.json'
        if path.is_file():
            required_inputs[f'vrptw/{split}/Cus100/metadata.json'] = path
    inputs = {key: dict(path=str(path), sha256=digest(path)) for key, path in required_inputs.items()}
    experiment = CODE_ROOT / 'results/optimization' / args.run_id
    experiment.mkdir(parents=True, exist_ok=False)
    shared = experiment / f'inputs/init_epoch_{args.expected_init_epoch:04d}.pt'
    shared.parent.mkdir()
    before = initialization['sha256']
    shutil.copy2(checkpoint, shared)
    if digest(checkpoint) != before or digest(shared) != before:
        raise ValueError('Initialization changed while copying; use a fixed archive')
    frozen = experiment / 'source/CaliRoute'
    source = source_snapshot(frozen)
    specs = {}
    for arm in arms:
        output = experiment / arm
        output.mkdir()
        name = args.run_id + '_' + arm.upper()
        cfg = builder(base, arm=arm, output=output, run_name=name, init_checkpoint=shared,
            data_root=data_root, seed=args.seed, units=units, epochs=args.epochs, chunk_size=args.chunk_size, eval_interval=args.eval_interval)
        path = output / 'config.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        specs[arm] = dict(config=str(path), config_sha256=digest(path), output_dir=str(output), epochs=args.epochs,
            required_validation_epochs=validation_epochs(args.epochs, args.eval_interval),
            log_dir=str(CODE_ROOT / 'results/logs/Cus_100_CS_0' / name / f'seed_{args.seed}'),
            checkpoint_dir=str(CODE_ROOT / 'results/checkpoints/Cus_100_CS_0' / name / f'seed_{args.seed}'),
            command=[sys.executable, '-B', '-u', '-m', 'offline2online.train', '--config', str(path), '--seed', str(args.seed), '--device', 'cuda:0'])
        preflight_output = output / 'preflight'
        preflight_output.mkdir()
        preflight_cfg = preflight_factory(cfg, preflight_output)
        preflight_path = preflight_output / 'config.yaml'
        preflight_path.write_text(yaml.safe_dump(preflight_cfg, sort_keys=False))
        specs[arm]['preflight'] = dict(config=str(preflight_path), config_sha256=digest(preflight_path),
            output_dir=str(preflight_output), epochs=2, validation_instances=4,
            log_dir=str(CODE_ROOT / 'results/logs/Cus_100_CS_0' / (name+'_PREFLIGHT') / f'seed_{args.seed}'),
            checkpoint_dir=str(CODE_ROOT / 'results/checkpoints/Cus_100_CS_0' / (name+'_PREFLIGHT') / f'seed_{args.seed}'),
            command=[sys.executable, '-B', '-u', '-m', 'offline2online.train', '--config', str(preflight_path), '--seed', str(args.seed), '--device', 'cuda:0'])
    prerequisites = [str(p.resolve()) for p in args.wait_for_experiment]
    if prerequisite_source_run is not None:
        previous = CODE_ROOT / 'results/optimization' / prerequisite_source_run / 'status.json'
        if local_live_prerequisite(previous) and str(previous.resolve()) not in prerequisites:
            prerequisites.append(str(previous.resolve()))
    manifest = dict(created_at_utc=now(), code_root=str(frozen), source=source,
        init_checkpoint=str(shared), init_checkpoint_sha256=before, source_init_checkpoint=str(checkpoint),
        initialization_provenance=initialization,
        source_init_epoch=args.expected_init_epoch, frozen_units=units, inputs=inputs, arms=specs, gpus=gpus,
        hardware_at_prepare=list(hardware_at_prepare.values()) if hardware_at_prepare is not None else None,
        wait_for_experiments=prerequisites, idle_checks=args.idle_checks, poll_seconds=args.poll_seconds,
        protocol=dict(task='vrptw100', epochs=args.epochs, seed=args.seed, variants={a:dict(gamma=definitions[a][0], reward_norm_mode=definitions[a][1]) for a in arms},
            world_size_per_arm=1, global_batch=64, n_traj=50, num_minibatches=4,
            ppo_update_epochs=5, ppo_step_chunk_size=args.chunk_size, learning_rate=1e-5,
            lr_schedule='constant', entropy_coef=.002, sl_coef=.35,
            initialization='same fixed epoch checkpoint weights, fresh optimizer/replay; preflight weights discarded',
            gpu_preflight=dict(epochs=2, instances_per_rollout=4, n_traj=4, chunk_size=4, validation_instances=4),
            validation_instances=1000, eval_interval=args.eval_interval, eval_n_traj=50, eval_seed=17000000+args.seed,
            eval_batch_size=32, test_enabled=False, objective='raw mean distance km among feasible solutions; report coverage separately',
            hardware_comparison='one GPU model per block; do not compare A6000 and 2080Ti timing as an algorithm effect',
            sampling='uniform shuffle_cycle, no priority sampler; common new fine-tuning protocol',
            normalization_bundle='historical actor RMS + PopArt + physical-cost SL leave-one-out without expert group mixing; expert/replay auxiliary losses unchanged',
            stage='80-epoch local screen unless explicitly overridden; requires full retraining before broad accuracy claims'))
    if protocol_overrides is not None:
        manifest['protocol'].update(copy.deepcopy(protocol_overrides))
    write_json(experiment / 'manifest.json', manifest)
    write_json(experiment / 'status.json', dict(state='prepared', arms={a:dict(state='prepared') for a in arms}))
    if args.launch:
        with (experiment / 'supervisor.log').open('a', buffering=1) as log:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(frozen / 'scripts/run_reward_norm_comparison.py'), '--supervise', str(experiment)],
                cwd=frozen, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (experiment / 'supervisor.pid').write_text(str(process.pid)+'\n')
    print(experiment, flush=True)
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, default=CODE_ROOT / 'configs/experiments/reward_norm_vrptw100.yaml')
    parser.add_argument('--init-checkpoint', type=Path, default=CODE_ROOT / DEFAULT_CHECKPOINT,
        help='Defaults to the shared epoch-300 weights included in this experiment branch')
    parser.add_argument('--expected-init-epoch', type=int, default=300)
    parser.add_argument('--data-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset')
    parser.add_argument('--run-id')
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--eval-interval', type=int, default=50,
        help='Validate every N epochs, plus epoch 0 and the final epoch (default: 50)')
    parser.add_argument('--chunk-size', type=int, default=18)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--wait-for-experiment', action='append', type=Path, default=[], help='Prior status.json whose assigned GPU jobs, including test, must finish')
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--idle-checks', type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervise', type=Path)
    args = parser.parse_args()
    if args.supervise:
        supervise(args.supervise.resolve())
    else:
        if args.run_id is None:
            args.run_id = f'REWARD_NORM_VRPTW100_S{args.seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        prepare(args)


if __name__ == '__main__':
    main()
