#!/usr/bin/env python3
"""Two GPUs per task: PPO init, validation-only parameter screening, long training."""
from __future__ import annotations

import argparse
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
import traceback

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_cus100_config import build_config
from run_plugin_comparison import append_hardware, latest_monitor, parse_gpu_pairs, terminate_group
from run_slppo_comparison import digest, now, number, read_csv, write_json

PHASES = ('ppo_init', 'control', 'candidate', 'long_control', 'long_candidate')


def select_parameters(control, candidate):
    """Select on the final screening validation, never test or training losses."""
    indexed = []
    for rows in (control, candidate):
        mapping = {str(row['instance_id']): row for row in rows}
        if len(rows) != 1000 or len(mapping) != 1000:
            raise ValueError('Screening requires all 1000 unique validation instances')
        for row in rows:
            if row.get('feasible') and not math.isfinite(float(row['objective_distance_km'])):
                raise ValueError('Nonfinite feasible route distance')
            if row.get('feasible') and not all(row.get('route_validation', {}).get(k) for k in ('checked', 'valid')):
                raise ValueError('Feasible screening route did not pass independent validation')
        indexed.append(mapping)
    left, right = indexed
    if left.keys() != right.keys():
        raise ValueError('Screening validation instance IDs differ')
    counts = [sum(bool(row['feasible']) for row in rows) for rows in (control, candidate)]
    shared = [i for i in left if left[i]['feasible'] and right[i]['feasible']]
    means = [statistics.mean(indexed[k][i]['objective_distance_km'] for i in shared) if shared else None for k in (0, 1)]
    if counts[0] != counts[1]:
        choice = 'candidate' if counts[1] > counts[0] else 'control'
        reason = 'higher validation feasible coverage'
    elif not shared:
        raise ValueError('Neither screening arm has a jointly feasible validation route')
    else:
        # A 0.1-metre tie tolerance prevents float32 summation noise deciding the winner.
        choice = 'candidate' if means[1] <= means[0] + 1e-4 else 'control'
        reason = 'shorter mean distance on jointly feasible validation IDs; numerical ties favor fewer updates'
    return {'selected': choice, 'reason': reason, 'selection_epoch': 40,
            'feasible_counts': dict(zip(('control', 'candidate'), counts)),
            'jointly_feasible_count': len(shared), 'distance_tie_tolerance_km': 1e-4,
            'joint_mean_distance_km': dict(zip(('control', 'candidate'), means)),
            'scope': 'single-seed local validation screen, constant LR; not a causal module ablation or proof of long-horizon superiority'}


def phase_progress(spec):
    rows = read_csv(Path(spec['log_dir']) / 'train_log.csv')
    evaluations = read_csv(Path(spec['log_dir']) / 'eval_log.csv')
    epochs = {int(row['epoch']) for row in rows}
    times = [number(row, 'distributed_train_wall_time_s') for row in rows
             if int(row['epoch']) > 1 and number(row, 'eval_wall_time_s') == 0]
    times = [v for v in times if math.isfinite(v)]
    best_path = Path(spec['checkpoint_dir']) / 'best_checkpoint.json'
    return {'target_epochs': spec['epochs'], 'completed_training_epochs': len(epochs),
            'latest_training_epoch': max(epochs, default=0),
            'latest_validation': evaluations[-1] if evaluations else None,
            'latest_train_row': rows[-1] if rows else None,
            'median_training_seconds_excluding_eval_and_first_epoch': statistics.median(times) if times else None,
            'best_checkpoint': json.loads(best_path.read_text()) if best_path.exists() else None,
            'rank_monitors': {str(rank): latest_monitor(Path(spec['output_dir']) / 'monitoring' / f'monitor_rank_{rank}.jsonl') for rank in range(2)}}


def refresh_progress(spec, phase_status, *, finished=False):
    """Transient monitor reads must never kill a live training process."""
    try:
        phase_status.update(phase_progress(spec))
        phase_status.pop('progress_read_warning', None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if finished:
            raise
        phase_status['progress_read_warning'] = f'{type(error).__name__}: {error}'


def load_manifest(experiment):
    return json.loads((experiment / 'manifest.json').read_text())


def worker(experiment, problem):
    manifest = load_manifest(experiment)
    task = manifest['tasks'][problem]
    actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=manifest['code_root'], text=True).strip()
    if actual_commit != manifest['git_commit']:
        raise RuntimeError('Frozen source commit changed since launch')
    root = experiment / problem
    status = {'problem': problem, 'state': 'running', 'started_at_utc': now(), 'worker_pid': os.getpid(),
              'gpus': task['gpus'], 'phases': {}, 'long_training_target_epochs': 1000}
    process = None
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        status['interrupted_signal'] = signum
        if process is not None:
            terminate_group(process)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for phase in ('ppo_init', 'control', 'candidate', 'selected_long'):
            if stopped:
                raise RuntimeError('Worker interrupted')
            if phase == 'selected_long':
                exports = [root / arm / 'evaluations' / 'epoch_0040.jsonl' for arm in ('control', 'candidate')]
                data = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for path in exports]
                selection = select_parameters(*data)
                selection.update(created_at_utc=now(), validation_route_sha256={str(p): digest(p) for p in exports})
                write_json(root / 'selection.json', selection)
                status['selection'] = selection
                phase = 'long_' + selection['selected']
            spec = task['phases'][phase]
            if digest(Path(spec['config'])) != spec['config_sha256']:
                raise RuntimeError(f'Phase config changed after launch: {phase}')
            status['current_phase'] = phase
            phase_status = {'state': 'starting', 'started_at_utc': now(), 'config': spec['config']}
            status['phases'][phase] = phase_status
            if phase != 'ppo_init':
                init = Path(task['init_checkpoint'])
                if not init.is_file():
                    raise RuntimeError(f'Missing completed PPO best checkpoint: {init}')
                phase_status['init_checkpoint_sha256'] = digest(init)
            write_json(root / 'status.json', status)
            with (Path(spec['output_dir']) / 'console.log').open('a', buffering=1) as log:
                process = subprocess.Popen(spec['command'], cwd=manifest['code_root'],
                    env=dict(os.environ, **spec['environment']), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase_status.update(state='running', pid=process.pid)
                print(f'{now()} {problem} {phase} started pid={process.pid}', flush=True)
                while True:
                    code = process.poll()
                    refresh_progress(spec, phase_status, finished=code is not None)
                    write_json(root / 'status.json', status)
                    if code is not None:
                        break
                    if stopped:
                        terminate_group(process)
                    time.sleep(10)
                phase_status.update(exit_code=code, finished_at_utc=now())
            if code != 0 or phase_status['completed_training_epochs'] != spec['epochs']:
                phase_status['state'] = 'failed'
                raise RuntimeError(f'{problem}/{phase} exit={code}, epochs={phase_status["completed_training_epochs"]}/{spec["epochs"]}')
            if not (Path(spec['checkpoint_dir']) / 'checkpoint_final.pt').is_file():
                raise RuntimeError('Missing final checkpoint after successful training')
            final_eval = phase_status['latest_validation'] or {}
            if final_eval.get('eval_status') != 'ok' or int(float(final_eval.get('eval_num_instances', 0))) != 1000:
                raise RuntimeError('Final validation must contain all 1000 instances')
            phase_status['state'] = 'completed'
            write_json(root / 'status.json', status)
        # Selection is finished before the frozen test is touched. Evaluate
        # only the validation-best checkpoint of the selected long run.
        best = Path(spec['checkpoint_dir']) / 'checkpoint_best.pt'
        test_output = root / 'test_best'
        test_output.mkdir()
        command = [sys.executable, '-B', '-u', str(CODE_ROOT / 'scripts' / 'evaluate_cus100_best.py'),
                   '--checkpoint', str(best), '--output-dir', str(test_output),
                   '--test-root', manifest['test_root'], '--gurobi-root', manifest['gurobi_root'],
                   '--problem', problem, '--seed', str(manifest['seed'])]
        status['current_phase'] = 'test_best'
        status['test_best'] = {'state': 'running', 'started_at_utc': now(), 'command': command,
                               'checkpoint_sha256': digest(best), 'gpu': task['gpus'][0]}
        write_json(root / 'status.json', status)
        with (test_output / 'console.log').open('a', buffering=1) as log:
            environment = dict(os.environ, **spec['environment'])
            environment['CUDA_VISIBLE_DEVICES'] = str(task['gpus'][0])
            process = subprocess.Popen(command, cwd=manifest['code_root'], env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status['test_best']['pid'] = process.pid
            while process.poll() is None:
                write_json(root / 'status.json', status)
                if stopped:
                    terminate_group(process)
                time.sleep(10)
        status['test_best'].update(exit_code=process.returncode, finished_at_utc=now())
        if process.returncode != 0:
            status['test_best']['state'] = 'failed'
            raise RuntimeError(f'Final frozen test evaluation failed: exit={process.returncode}')
        summary = json.loads((test_output / 'summary.json').read_text())
        status['test_best'].update(state='completed', summary=summary)
        status.update(state='completed', exit_code=0)
    except Exception as error:
        status.update(state='interrupted' if stopped else 'failed', exit_code=1,
                      error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        if process is not None and process.poll() is None:
            deadline = time.monotonic() + 65
            while process.poll() is None and time.monotonic() < deadline:
                terminate_group(process)
                time.sleep(.25)
        status['finished_at_utc'] = now()
        write_json(root / 'status.json', status)
    return status['exit_code']


def supervise(experiment):
    manifest = load_manifest(experiment)
    status = {'state': 'running', 'started_at_utc': now(), 'supervisor_pid': os.getpid(), 'tasks': {}}
    processes, streams = {}, []
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        status['interrupted_signal'] = signum
        for process in processes.values():
            terminate_group(process)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for problem in manifest['tasks']:
            stream = (experiment / problem / 'worker.log').open('a', buffering=1)
            streams.append(stream)
            processes[problem] = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                '--worker', str(experiment), '--problem', problem], cwd=manifest['code_root'],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            for problem, process in processes.items():
                path = experiment / problem / 'status.json'
                detail = json.loads(path.read_text()) if path.exists() else {'state': 'starting'}
                detail['worker_exit_code'] = process.poll()
                if detail['worker_exit_code'] not in (None, 0) and detail.get('state') not in ('failed', 'interrupted'):
                    detail.update(state='failed', error='Task worker failed; inspect worker.log')
                status['tasks'][problem] = detail
            status['updated_at_utc'] = now()
            write_json(experiment / 'status.json', status)
            try:
                append_hardware(experiment)
            except Exception as error:
                print(f'Hardware monitor error: {error}', flush=True)
            if all(p.poll() is not None for p in processes.values()):
                break
            if interrupted:
                for process in processes.values():
                    terminate_group(process)
            time.sleep(15)
        success = all(p.returncode == 0 for p in processes.values()) and all(t.get('state') == 'completed' for t in status['tasks'].values())
        status.update(state='completed' if success else 'interrupted' if interrupted else 'failed',
                      exit_code=0 if success else 1, finished_at_utc=now())
        write_json(experiment / 'status.json', status)
        return status['exit_code']
    finally:
        deadline = time.monotonic() + 65
        while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
            for process in processes.values():
                terminate_group(process)
            time.sleep(.25)
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset' / 'dataset')
    parser.add_argument('--output-root', type=Path, default=CODE_ROOT / 'results' / 'optimization')
    parser.add_argument('--test-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset' / 'test_release')
    parser.add_argument('--gurobi-root', type=Path, default=CODE_ROOT.parent / 'results' / 'gurobi')
    parser.add_argument('--cvrp-gpus', default='0,2')
    parser.add_argument('--vrptw-gpus', default='1,3')
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--problem', choices=['cvrp', 'vrptw'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        raise SystemExit(supervise(args.supervise.resolve()))
    if args.worker:
        if not args.problem:
            parser.error('--worker requires --problem')
        raise SystemExit(worker(args.worker.resolve(), args.problem))
    pairs = parse_gpu_pairs(args.cvrp_gpus, args.vrptw_gpus)
    name = args.run_id or f'CUS100_FINETUNE_S{args.seed}_E1000_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    if Path(name).name != name or name in ('.', '..'):
        parser.error('--run-id must be a directory name')
    experiment = args.output_root.resolve() / name
    if experiment.exists() and not args.dry_run:
        parser.error('Output exists; choose a fresh run ID')
    tasks, configs, inputs = {}, {}, []
    for problem, gpus in zip(('cvrp', 'vrptw'), pairs):
        prefix = f'{name}_{problem.upper()}'
        init = CODE_ROOT / 'results' / 'checkpoints' / 'Cus_100_CS_0' / f'{prefix}_PPO_INIT' / f'seed_{args.seed}' / 'checkpoint_best.pt'
        phases = {}
        for phase in PHASES:
            output = experiment / problem / phase
            cfg = build_config(problem=problem, phase=phase, run_name=f'{prefix}_{phase.upper()}',
                               output_dir=output, data_root=args.data_root.resolve(),
                               init_checkpoint=None if phase == 'ppo_init' else init, seed=args.seed)
            configs[(problem, phase)] = cfg
            phases[phase] = {'output_dir': str(output), 'config': str(output / 'config.yaml'),
                'epochs': cfg['training']['epochs'],
                'log_dir': str(CODE_ROOT / 'results' / 'logs' / 'Cus_100_CS_0' / cfg['run_name'] / f'seed_{args.seed}'),
                'checkpoint_dir': str(CODE_ROOT / 'results' / 'checkpoints' / 'Cus_100_CS_0' / cfg['run_name'] / f'seed_{args.seed}'),
                'command': [sys.executable, '-B', '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                            '--nproc-per-node=2', '--max-restarts=0', '--module', 'offline2online.train',
                            '--config', str(output / 'config.yaml'), '--seed', str(args.seed), '--device', 'cuda'],
                'environment': {'CUDA_VISIBLE_DEVICES': ','.join(gpus), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'NUMBA_CACHE_DIR': str(experiment / problem / 'numba_cache'),
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1'}}
        tasks[problem] = {'gpus': list(map(int, gpus)), 'init_checkpoint': str(init), 'phases': phases}
        for split in ('train', 'val'):
            folder = args.data_root.resolve() / problem / split / 'Cus100'
            for filename in ('metadata.json', 'instances.pkl', 'gurobi_summary.csv', 'expert_solutions.csv'):
                path = folder / filename
                if not path.is_file():
                    parser.error(f'Missing input: {path}')
                inputs.append(path)
            meta = json.loads((folder / 'metadata.json').read_text())
            if meta.get('split') != split or int(meta.get('num_customers', 0)) != 100 or int(meta.get('num_instances', 0)) != (5000 if split == 'train' else 1000):
                parser.error(f'Unexpected metadata: {folder}')
    # Only existence is checked now; test objectives are read after all
    # training and checkpoint selection have completed for the task.
    for problem in tasks:
        for path in (args.test_root / problem / 'test' / 'Cus100' / 'instances.pkl',
                     args.test_root / problem / 'test' / 'Cus100' / 'metadata.json',
                     args.gurobi_root / problem / 'test' / 'Cus100' / 'gurobi_summary.csv'):
            if not path.is_file():
                parser.error(f'Missing final-test input: {path}')
    if args.dry_run:
        print(yaml.safe_dump({f'{a}/{b}': v for (a, b), v in configs.items()}, sort_keys=False))
        return
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True).strip():
        parser.error('Commit source before launching, then use a detached snapshot')
    experiment.mkdir(parents=True)
    for (problem, phase), cfg in configs.items():
        spec = tasks[problem]['phases'][phase]
        Path(spec['output_dir']).mkdir(parents=True)
        path = Path(spec['config'])
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        spec['config_sha256'] = digest(path)
    import torch
    manifest = {'created_at_utc': now(), 'code_root': str(CODE_ROOT),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip(),
        'input_sha256': {str(p): digest(p) for p in inputs}, 'tasks': tasks, 'seed': args.seed,
        'test_root': str(args.test_root.resolve()), 'gurobi_root': str(args.gurobi_root.resolve()),
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda},
        'protocol': {'problems': ['cvrp', 'vrptw'], 'customers': 100, 'world_size_per_task': 2,
            'phases': ['100 epochs target-task random-init PPO', '40 epochs control: 4 PPO passes, SL=.5',
                       '40 epochs candidate: 3 PPO passes, SL=.35', '1000 epochs selected setting from common PPO best'],
            'screen_selection': 'final epoch40 full validation; feasibility first, joint mean distance second; ties favor fewer updates',
            'global_instances_per_rollout': 64, 'global_trajectories_per_rollout': 3200,
            'global_instances_per_optimizer_step': 16, 'optimizer_steps_per_epoch': {'control': 16, 'candidate': 12, 'ppo_init': 12},
            'lr_peak': 5e-5, 'lr_min': 1e-5, 'lr_scaling': 'global instance batch128->64; previous peak7.071e-5->5e-5',
            'screen_schedule': 'constant LR5e-5, entropy.01 local probe; long stages restart warmup/cosine from the same PPO best',
            'rollout_max_steps': 201, 'training_chunk_size': 8, 'eval_batch_size': 32, 'eval_n_traj': 50,
            'validation_instances': 1000, 'checkpoint_selection': 'validation only', 'test_used_for_training_or_selection': False,
            'final_test': 'after selected long training, evaluate its validation-best checkpoint once on all 1000 frozen test instances',
            'model': 'optimized_v2 RDI/AGDA/SLPPO; hyperparameter screening, not original-vs-v2 comparison',
            'tuning_basis': 'CVRP50 update-time dominance and elevated route clipping motivate fewer passes and weaker SL; unproven transfer hypothesis',
            'latest_checkpoint_interval': 5, 'monitor_interval': 10}}
    write_json(experiment / 'manifest.json', manifest)
    with (experiment / 'supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--supervise', str(experiment)],
            cwd=CODE_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(experiment / 'launcher.json', {'supervisor_pid': process.pid, 'created_at_utc': now()})
    print(json.dumps({'experiment': str(experiment), 'supervisor_pid': process.pid, 'status': str(experiment / 'status.json')}, indent=2))


if __name__ == '__main__':
    main()
