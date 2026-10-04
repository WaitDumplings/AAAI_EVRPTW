#!/usr/bin/env python3
"""Two synchronous GPUs per model, frozen-source long-run plug-in comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caliroute.optimization import apply_optimization_profile
from offline2online.training_schedule import schedule_for_epoch
from run_slppo_comparison import (build_configs, digest, finite_json, now, number,
                                  read_csv, update_report, write_json)


def parse_gpu_pairs(baseline, optimized):
    pairs = [[value.strip() for value in group.split(',')] for group in (baseline, optimized)]
    flat = sum(pairs, [])
    if any(len(group) != 2 for group in pairs) or len(set(flat)) != 4 or any(not item.isdigit() for item in flat):
        raise ValueError('Supply two distinct GPUs per model and four distinct indices overall')
    return pairs


def build_long_configs(args, experiment):
    configs = build_configs(args, experiment)
    configs['optimized'] = apply_optimization_profile(configs['optimized'], 'optimized_v2')
    for arm, cfg in configs.items():
        cfg['training'].update({
            'epochs': args.epochs, 'num_minibatches': args.num_minibatches,
            'checkpoint_interval': args.checkpoint_interval,
            'latest_checkpoint_interval': 5,
            'learning_rate': args.learning_rate, 'lr_schedule': 'warmup_cosine',
            'lr_warmup_epochs': args.lr_warmup_epochs, 'lr_min': args.lr_min,
            'entropy_initial_coef': .01, 'entropy_final_coef': .002,
            'monitor_interval': args.monitor_interval, 'monitor_gradient_components': True,
            'monitor_output_dir': str(experiment / arm / 'monitoring'),
            'monitor_target_kl': .02, 'amp_init_scale': 4096.,
        })
        schedule_for_epoch(cfg, 1)
        schedule_for_epoch(cfg, args.epochs)
    return configs


def latest_monitor(path):
    if not path.exists():
        return None
    # A process may be appending its final line; retain the last complete record.
    with path.open() as handle:
        last = None
        for line in handle:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


RECORDED_TIME_SCOPE = ("sum of each previous session's last recorded elapsed time plus the current "
                       'session elapsed time; excludes downtime, discarded work, and unrecorded tails')


def recorded_session_times(rows):
    """Keep raw clocks and add a continuous clock across retained run sessions.

    Input rows must follow recording order. A restart resets run_elapsed_seconds;
    only the retained portion of each session contributes to this comparison.
    """
    offset, last_elapsed, current_session = 0., 0., None
    result = []
    for row in rows:
        elapsed = number(row, 'run_elapsed_seconds')
        session = row.get('run_session_id') or '__legacy_session__'
        output = dict(row)
        output['recorded_active_session_seconds'] = None
        if math.isfinite(elapsed) and elapsed >= 0:
            # Legacy logs may lack IDs; a backwards clock still identifies a restart.
            if current_session is not None and (session != current_session or elapsed < last_elapsed):
                offset += last_elapsed
                last_elapsed = 0.
            current_session, last_elapsed = session, elapsed
            output['recorded_active_session_seconds'] = offset + elapsed
        result.append(output)
    return result


def training_progress(manifest, status, training_rows, evaluation_rows):
    """Distinguish a configured epoch budget from completed work and termination."""
    target = int(manifest['protocol']['epochs'])
    interrupted = bool(status.get('interrupted_signal'))
    arms = {}
    for arm in manifest['arms']:
        epochs = {int(row['epoch']) for row in training_rows.get(arm, []) if row.get('epoch') not in (None, '')}
        completed = len(epochs & set(range(1, target + 1)))
        evaluations = [int(row['epoch']) for row in evaluation_rows.get(arm, [])
                       if row.get('epoch') not in (None, '') and row.get('eval_status') == 'ok']
        process = status.get('arms', {}).get(arm, {})
        code = process.get('exit_code')
        if interrupted and code is not None:
            process_state = 'interrupted'
        elif code is not None:
            process_state = 'completed' if code == 0 and completed == target else 'failed'
        elif status.get('finished_at_utc'):
            process_state = 'interrupted'
        else:
            process_state = 'running' if process.get('pid') else 'pending'
        arms[arm] = {
            'completed_training_epochs': completed,
            'latest_training_epoch': max(epochs, default=0),
            'latest_validation_epoch': max(evaluations, default=None),
            'remaining_epochs': max(0, target - completed),
            'process_state': process_state,
            'exit_code': code,
        }
    if interrupted:
        state = 'interrupted'
    elif status.get('exit_code') not in (None, 0) or any(item['process_state'] == 'failed' for item in arms.values()):
        state = 'failed'
    elif (set(arms) >= {'baseline', 'optimized'} and status.get('exit_code') == 0
          and all(item['process_state'] == 'completed' for item in arms.values())):
        state = 'completed'
    elif status.get('finished_at_utc'):
        state = 'interrupted'
    else:
        state = 'running'
    return {'target_epochs': target, 'run_state': state, 'arms': arms}


def update_long_report(experiment, manifest, status):
    update_report(experiment, manifest, status)
    path = experiment / 'comparison.json'
    report = json.loads(path.read_text())
    report['protocol'] = manifest['protocol']
    report['source_commit'] = manifest['git_commit']
    report['diagnostics'] = {}
    report['validation_vs_wall_time'] = {}
    report['validation_wall_time_scope'] = RECORDED_TIME_SCOPE
    training = {}
    train_rows, evaluation_rows = {}, {}
    for arm, spec in manifest['arms'].items():
        rows = read_csv(Path(spec['log_dir']) / 'train_log.csv')
        train_rows[arm] = rows
        evaluation_rows[arm] = read_csv(Path(spec['log_dir']) / 'eval_log.csv')
        report['arms'][arm]['lineage'] = {key: spec[key] for key in ('resume_checkpoint', 'source_experiment') if key in spec}
        training[arm] = {int(row['epoch']): row for row in rows}
        steady_times = [number(row, 'distributed_train_wall_time_s') for row in rows
                        if int(row['epoch']) > max(1, manifest['protocol']['lr_warmup_epochs'])
                        and number(row, 'eval_wall_time_s') == 0
                        and math.isfinite(number(row, 'distributed_train_wall_time_s'))]
        report['arms'][arm]['median_epoch_seconds_excluding_warmup_and_eval'] = statistics.median(steady_times) if steady_times else None
        monitors = {str(rank): latest_monitor(experiment / arm / 'monitoring' / f'monitor_rank_{rank}.jsonl')
                    for rank in range(2)}
        report['diagnostics'][arm] = {'latest_rank_monitors': monitors, 'latest_train_row': rows[-1] if rows else None}
        cumulative, curve = 0., []
        for row in recorded_session_times(rows):
            cumulative += number(row, 'epoch_wall_time_s')
            if row.get('eval_status') == 'ok':
                curve.append({'epoch': int(row['epoch']), 'epoch_wall_seconds_cumulative': cumulative,
                              'run_elapsed_seconds': number(row, 'run_elapsed_seconds'),
                              'run_session_id': row.get('run_session_id'),
                              'recorded_active_session_seconds': row['recorded_active_session_seconds'],
                              'feasible_rate': number(row, 'eval_feasible_rate'),
                              'mean_distance_km': number(row, 'eval_avg_min_objective_distance_km')})
        report['validation_vs_wall_time'][arm] = curve
    progress = training_progress(manifest, status, train_rows, evaluation_rows)
    report = {'progress': progress, **report}
    for arm, fields in progress['arms'].items():
        report['arms'][arm].update(fields)
    warmup = manifest['protocol']['lr_warmup_epochs']
    epochs = sorted(training['baseline'].keys() & training['optimized'].keys())
    matched = [epoch for epoch in epochs if epoch > max(1, warmup) and all(
        number(training[arm][epoch], 'eval_wall_time_s') == 0 and
        math.isfinite(number(training[arm][epoch], 'distributed_train_wall_time_s')) for arm in training)]
    report.pop('matched_epoch_timing', None)
    if matched:
        timing = {arm: statistics.median(number(training[arm][epoch], 'distributed_train_wall_time_s') for epoch in matched)
                  for arm in training}
        report['matched_epoch_timing'] = {
            'epochs': matched, 'baseline_median_seconds': timing['baseline'],
            'optimized_median_seconds': timing['optimized'],
            'time_reduction_pct': 100 * (1 - timing['optimized'] / timing['baseline']),
            'scope': 'slowest rank per epoch; excludes LR warmup and validation; includes synchronization/replay/diagnostics',
        }
    write_json(path, report)


def append_hardware(experiment):
    fields = ['index', 'utilization.gpu', 'memory.used', 'memory.total', 'temperature.gpu', 'power.draw']
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        return
    records = [dict(zip(fields, [value.strip() for value in row])) for row in csv.reader(result.stdout.splitlines())]
    with (experiment / 'hardware.jsonl').open('a') as handle:
        handle.write(json.dumps({'time_utc': now(), 'gpus': records}) + '\n')


def terminate_group(process):
    if process.poll() is None:
        try:
            if getattr(process, '_termination_started', None) is None:
                process._termination_started = time.monotonic()
                os.killpg(process.pid, signal.SIGTERM)
            elif time.monotonic() - process._termination_started >= 60:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def supervise(experiment):
    manifest = json.loads((experiment / 'manifest.json').read_text())
    status = {'started_at_utc': now(), 'supervisor_pid': os.getpid(), 'arms': {}}
    processes, streams = {}, []
    plot_signature = None
    def interrupted(signum, frame):
        status['interrupted_signal'] = signum
        for process in processes.values():
            terminate_group(process)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for arm, spec in manifest['arms'].items():
            stream = (experiment / arm / 'console.log').open('a', buffering=1)
            streams.append(stream)
            process = subprocess.Popen(spec['command'], cwd=manifest['code_root'],
                                       env=dict(os.environ, **spec['environment']), stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes[arm] = process
            status['arms'][arm] = {'pid': process.pid, 'gpus': spec['gpus'], 'world_size': 2, 'started_at_utc': now()}
        while True:
            for arm, process in processes.items():
                code = process.poll()
                if code is not None and 'exit_code' not in status['arms'][arm]:
                    status['arms'][arm].update(exit_code=code, finished_at_utc=now())
            write_json(experiment / 'status.json', status)
            for operation in (lambda: append_hardware(experiment), lambda: update_long_report(experiment, manifest, status)):
                try:
                    operation()
                except Exception as error:
                    print(f'Monitor update failed (training status remains available): {type(error).__name__}: {error}', flush=True)
            signature = tuple(len(read_csv(Path(spec['log_dir']) / 'eval_log.csv')) for spec in manifest['arms'].values())
            if any(signature) and signature != plot_signature:
                try:
                    from plot_plugin_comparison import render
                    render(experiment)
                    plot_signature = signature
                except Exception as error:
                    print(f'Plot update failed: {type(error).__name__}: {error}', flush=True)
            failed = [arm for arm, process in processes.items() if process.poll() not in (None, 0)]
            if failed or status.get('interrupted_signal'):
                status['fail_fast_trigger'] = failed
                for process in processes.values():
                    terminate_group(process)
            if all(process.poll() is not None for process in processes.values()):
                break
            time.sleep(30)
        status.update(finished_at_utc=now(), exit_code=int(any(p.returncode != 0 for p in processes.values())))
        write_json(experiment / 'status.json', status)
        update_long_report(experiment, manifest, status)
        return status['exit_code']
    finally:
        for process in processes.values():
            terminate_group(process)
        # Cleanup is bounded even if a CUDA worker cannot handle SIGTERM.
        deadline = time.monotonic() + 65
        while any(process.poll() is None for process in processes.values()) and time.monotonic() < deadline:
            for process in processes.values():
                terminate_group(process)
            time.sleep(.25)
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset' / 'dataset')
    parser.add_argument('--problem', choices=['cvrp', 'vrptw'], default='cvrp')
    parser.add_argument('--customers', type=int, choices=[15, 50, 100], default=50)
    parser.add_argument('--baseline-gpus', default='0,2')
    parser.add_argument('--optimized-gpus', default='1,3')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num-envs-per-gpu', dest='num_envs', type=int, default=64)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--lr-warmup-epochs', type=int, default=20)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--eval-interval', type=int, default=20)
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--monitor-interval', type=int, default=20)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--eval-limit', type=int, default=None)
    parser.add_argument('--expert-limit', type=int, default=None)
    parser.add_argument('--output-root', type=Path, default=CODE_ROOT / 'results' / 'optimization')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        raise SystemExit(supervise(args.supervise.resolve()))
    try:
        pairs = parse_gpu_pairs(args.baseline_gpus, args.optimized_gpus)
    except ValueError as error:
        parser.error(str(error))
    for key in ('epochs', 'num_envs', 'n_traj', 'num_minibatches', 'eval_interval', 'checkpoint_interval', 'monitor_interval', 'eval_batch_size'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.num_envs % args.num_minibatches:
        parser.error('num-envs-per-gpu must be divisible by num-minibatches')
    if args.init_checkpoint is None or not args.init_checkpoint.is_file():
        parser.error('--init-checkpoint must point to a completed, trusted local PPO checkpoint')
    if args.learning_rate is None:
        # A conservative experiment setting, not a universal PPO scaling law.
        args.learning_rate = 5e-5 * math.sqrt(2 * args.num_envs / 64)
    name = args.run_id or f'PLUGIN_DUAL_{args.problem.upper()}{args.customers}_S{args.seed}_E{args.epochs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    if Path(name).name != name or name in ('.', '..'):
        parser.error('--run-id must be a single directory name')
    experiment = args.output_root.resolve() / name
    try:
        configs = build_long_configs(args, experiment)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(yaml.safe_dump(configs, sort_keys=False))
        return
    if experiment.exists():
        parser.error(f'Output exists; choose a fresh run ID: {experiment}')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True).strip():
        parser.error('Commit code before launching; run from a detached snapshot for immutability')
    files = [args.init_checkpoint.resolve()]
    for split in ('train', 'val'):
        folder = args.data_root.resolve() / args.problem / split / f'Cus{args.customers}'
        files.extend(folder / name for name in ('instances.pkl', 'metadata.json', 'gurobi_summary.csv'))
        if split == 'train':
            files.append(folder / 'expert_solutions.csv')
    for path in files:
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    import torch
    init = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    old = init.get('config', {})
    for section, keys in {'data': ('problem_type', 'num_customers'),
                          'model': ('embedding_dim', 'n_encode_layers', 'use_dynamic_decision_encoder',
                                    'dynamic_decision_heads', 'dynamic_decision_delta_k', 'dynamic_decision_delta_v',
                                    'dynamic_decision_delta_action_key', 'dynamic_decision_action_bias')}.items():
        for key in keys:
            if old.get(section, {}).get(key) != configs['baseline'][section][key]:
                parser.error(f'Initialization model mismatch: {section}.{key}')
    del init
    experiment.mkdir(parents=True)
    manifest = {
        'created_at_utc': now(), 'code_root': str(CODE_ROOT),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip(),
        'comparison': 'correctness-fixed original RDI/AGDA/SLPPO versus portable v2; two synchronous GPUs each; train/val only',
        'seed': args.seed, 'init_checkpoint_sha256': digest(args.init_checkpoint),
        'input_sha256': {str(path): digest(path) for path in files},
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda},
        'gpu_topology': subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        'protocol': {'epochs': args.epochs, 'world_size_per_model': 2, 'local_instances_per_rollout': args.num_envs,
                     'global_instances_per_rollout': 2 * args.num_envs,
                     'global_trajectories_per_rollout': 2 * args.num_envs * args.n_traj,
                     'global_instances_per_optimizer_step': 2 * args.num_envs // args.num_minibatches,
                     'gradient_objective': 'mean of rank-local masked objectives; not global valid-token weighting',
                     'learning_rate_peak': args.learning_rate, 'lr_warmup_epochs': args.lr_warmup_epochs,
                     'learning_rate_final': args.lr_min, 'lr_scaling': 'sqrt(global instance batch / previous 64), a tuning hypothesis',
                     'entropy_initial': .01, 'entropy_final': .002, 'amp_init_scale': 4096., 'ppo_update_epochs': 4, 'clip_coef': .2,
                     'evaluation': 'fixed validation IDs and seed, 50 samples by default; never trains on frozen test',
                     'monitor_interval': args.monitor_interval, 'eval_n_traj': args.n_traj,
                     'latest_checkpoint_interval': 5,
                     'quality_claim': 'single-seed screening; improvements require the completed paired validation results'},
        'arms': {},
    }
    for (arm, cfg), gpu_pair in zip(configs.items(), pairs):
        folder = experiment / arm
        folder.mkdir()
        config_path = folder / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest['arms'][arm] = {
            'gpus': list(map(int, gpu_pair)), 'config': str(config_path), 'config_sha256': digest(config_path),
            'command': [sys.executable, '-B', '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                        '--nproc-per-node=2', '--max-restarts=0', '--module', 'offline2online.train',
                        '--config', str(config_path), '--seed', str(args.seed), '--device', 'cuda'],
            'environment': {'CUDA_VISIBLE_DEVICES': ','.join(gpu_pair), 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
                            'NUMBA_CACHE_DIR': str(folder / 'numba_cache'), 'PYTHONDONTWRITEBYTECODE': '1',
                            'PYTHONUNBUFFERED': '1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1'},
            'log_dir': str(CODE_ROOT / 'results' / 'logs' / f'Cus_{args.customers}_CS_0' / cfg['run_name'] / f'seed_{args.seed}'),
        }
    write_json(experiment / 'manifest.json', manifest)
    with (experiment / 'supervisor.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--supervise', str(experiment)],
                                   cwd=CODE_ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(experiment / 'launcher.json', {'supervisor_pid': process.pid, 'created_at_utc': now()})
    print(json.dumps({'experiment': str(experiment), 'supervisor_pid': process.pid,
                      'report': str(experiment / 'comparison.json')}, indent=2))


if __name__ == '__main__':
    main()
