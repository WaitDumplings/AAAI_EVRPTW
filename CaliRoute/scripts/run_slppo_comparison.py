#!/usr/bin/env python3
"""Launch a two-GPU controlled SL-PPO comparison and maintain paired reports.

The baseline shares correctness fixes with the optimized arm; all optional
model, replay, and speed changes are disabled. This is not an untouched legacy
run. Test data are never used by this experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from caliroute.methods import method_preset
from caliroute.optimization import apply_optimization_profile


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(finite_json(data), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def paired_metrics(baseline: list[dict], optimized: list[dict]) -> dict:
    def indexed(rows):
        result = {str(row['instance_id']): row for row in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate instance IDs in evaluation export')
        return result
    left, right = indexed(baseline), indexed(optimized)
    if left.keys() != right.keys():
        raise ValueError('evaluation instance IDs differ between arms')
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if not a.get('feasible') or not b.get('feasible'):
            continue
        da, db = number(a, 'objective_distance_km'), number(b, 'objective_distance_km')
        if math.isfinite(da) and math.isfinite(db) and da > 0:
            pairs.append((da, db))
    improvements = [(a - b) / a * 100 for a, b in pairs]
    out = {
        'num_instances': len(left), 'jointly_feasible_instances': len(pairs),
        'baseline_feasible_rate': sum(bool(r.get('feasible')) for r in left.values()) / max(len(left), 1),
        'optimized_feasible_rate': sum(bool(r.get('feasible')) for r in right.values()) / max(len(right), 1),
        'distance_comparison_scope': 'same jointly feasible instance IDs; improvement positive means optimized shorter',
        'optimized_wins': sum(b < a - 1e-6 for a, b in pairs),
        'ties': sum(abs(a - b) <= 1e-6 for a, b in pairs),
        'baseline_wins': sum(b > a + 1e-6 for a, b in pairs),
    }
    if pairs:
        out.update(baseline_mean_distance_km=statistics.mean(a for a, _ in pairs),
                   optimized_mean_distance_km=statistics.mean(b for _, b in pairs),
                   mean_paired_improvement_pct=statistics.mean(improvements),
                   median_paired_improvement_pct=statistics.median(improvements))
    return out


def build_configs(args, experiment: Path) -> dict[str, dict]:
    if args.problem == 'evrptw':
        raise ValueError('This comparison currently validates CVRP/VRPTW only; EVRPTW design remains experimental.')
    dataset = Path(args.data_root).resolve() / args.problem
    train, val = dataset / 'train' / f'Cus{args.customers}', dataset / 'val' / f'Cus{args.customers}'
    preset = method_preset('slppo')
    offline = preset.offline_config()
    offline.update(init_checkpoint_path=str(Path(args.init_checkpoint).resolve()),
                   expert_solution_path=str(train / 'expert_solutions.csv'), expert_dataset_path=str(train))
    rollout_steps = (110 if args.problem == 'cvrp' else 120) if args.customers == 100 else (90 if args.problem == 'cvrp' else 120)
    cfg = {
        'dataset_name': f'Geo-{args.problem.upper()}-v1',
        'data': {'problem_type': args.problem, 'num_customers': args.customers, 'num_charging_stations': 0,
                 'train_dataset_path': str(train), 'train_sample_mode': 'shuffle_cycle', 'async_instance_prefetch': False},
        'env': {'use_fast_env': True, 'use_jit_mask': True, 'normalize_reward': True,
                'reward_distance_scale_mode': 'dataset_single_customer_repair_median',
                'charging_mode': 'fixed_full', 'info_level': 'light'},
        'model': {'embedding_dim': 256, 'tanh_clipping': 15.0, 'n_encode_layers': 2, 'use_graph_token': True,
                  'use_dynamic_decision_encoder': True, 'dynamic_decision_heads': 4,
                  'dynamic_decision_delta_k': False, 'dynamic_decision_delta_v': False,
                  'dynamic_decision_delta_action_key': True, 'dynamic_decision_action_bias': True,
                  'distance_injection': 'encoder', 'use_encoder_distance_bias': True},
        'critic': {'use_decomposed_critic': False, 'advantage_mode': 'total'},
        'training': {'online_training': False, 'epochs': args.epochs, 'num_envs_per_gpu': args.num_envs,
                     'n_traj': args.n_traj, 'rollout_steps': rollout_steps, 'ppo_step_chunk_size': 16,
                     'ppo_update_epochs': 4, 'num_minibatches': 4, 'gamma': .99, 'gae_lambda': .95,
                     'clip_coef': .2, 'vf_coef': .5, 'ent_coef': .01, 'learning_rate': args.learning_rate,
                     'weight_decay': 0., 'max_grad_norm': 1., 'checkpoint_interval': args.eval_interval,
                     'debug': True, 'debug_log_every': 1, 'mixed_precision': True, 'profile_timing': True,
                     'post_init_seed': args.seed},
        'pbrs': {'use_customer_pbrs': False, 'use_repair_distance_pbrs': False,
                 'use_feasible_ratio_pbrs': False, 'use_terminal_heuristic': False},
        'evaluation': {'eval_interval': args.eval_interval, 'eval_path': str(val), 'eval_n_traj': args.n_traj,
                       'eval_decode_mode': 'sample', 'eval_max_steps': rollout_steps,
                       'eval_batch_size': args.eval_batch_size, 'eval_info_level': 'light', 'eval_save_routes': True,
                       'eval_seed': args.seed + 17000000, 'eval_before_training': True, 'gurobi_summary_path': str(val / 'gurobi_summary.csv')},
        'offline': offline, 'advantage': preset.advantage_config(),
    }
    if args.eval_limit is not None:
        cfg['evaluation']['eval_limit'] = args.eval_limit
    if args.expert_limit is not None:
        cfg['offline']['expert_limit'] = args.expert_limit
    configs = {}
    for arm in ('baseline', 'optimized'):
        config = apply_optimization_profile(cfg, arm)
        config['run_name'] = f'{experiment.name}_{arm.upper()}'
        config['evaluation']['eval_output_dir'] = str(experiment / arm / 'evaluations')
        configs[arm] = config
    return configs


def update_report(experiment: Path, manifest: dict, status: dict) -> None:
    report = {'updated_at_utc': now(), 'comparison': manifest['comparison'], 'status': status, 'arms': {}}
    evaluations, train_rows = {}, {}
    for arm, spec in manifest['arms'].items():
        rows = read_csv(Path(spec['log_dir']) / 'train_log.csv')
        train_rows[arm] = {int(row['epoch']): row for row in rows}
        eval_rows = read_csv(Path(spec['log_dir']) / 'eval_log.csv')
        evaluations[arm] = {int(row['epoch']): row for row in eval_rows if row.get('eval_status') == 'ok'}
        times = [number(row, 'epoch_wall_time_s') for row in rows
                 if number(row, 'epoch') > 1 and number(row, 'eval_wall_time_s') == 0]
        times = [t for t in times if math.isfinite(t)]
        report['arms'][arm] = {
            'completed_epochs': int(rows[-1]['epoch']) if rows else 0,
            'latest_evaluation': eval_rows[-1] if eval_rows else None,
            'median_epoch_seconds_excluding_warmup_and_eval': statistics.median(times) if times else None,
            'console_log': str(experiment / arm / 'console.log'),
        }
    common = sorted(evaluations.get('baseline', {}).keys() & evaluations.get('optimized', {}).keys())
    timing_epochs = sorted(train_rows.get('baseline', {}).keys() & train_rows.get('optimized', {}).keys())
    timing_epochs = [epoch for epoch in timing_epochs if epoch > 1 and all(
        number(train_rows[arm][epoch], 'eval_wall_time_s') == 0 and
        math.isfinite(number(train_rows[arm][epoch], 'epoch_wall_time_s'))
        for arm in ('baseline', 'optimized'))]
    if timing_epochs:
        times = {arm: statistics.median(number(train_rows[arm][epoch], 'epoch_wall_time_s') for epoch in timing_epochs)
                 for arm in ('baseline', 'optimized')}
        report['matched_epoch_timing'] = {
            'epochs': timing_epochs, 'baseline_median_seconds': times['baseline'],
            'optimized_median_seconds': times['optimized'],
            'time_reduction_pct': 100 * (1 - times['optimized'] / times['baseline']),
            'scope': 'same completed epochs, excluding first epoch and evaluation; includes design and replay costs',
        }
    report['matched_epochs'] = common
    report['paired_evaluations'] = {}
    for epoch in common:
        exports = [experiment / arm / 'evaluations' / f'epoch_{epoch:04d}.jsonl' for arm in ('baseline', 'optimized')]
        if all(path.exists() for path in exports):
            data = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for path in exports]
            report['paired_evaluations'][str(epoch)] = paired_metrics(*data)
    write_json(experiment / 'comparison.json', report)


def supervise(experiment: Path) -> int:
    manifest = json.loads((experiment / 'manifest.json').read_text())
    status = {'started_at_utc': now(), 'supervisor_pid': os.getpid(), 'arms': {}}
    processes, handles = {}, []
    def stop(signum, frame):
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for arm, spec in manifest['arms'].items():
            handle = (experiment / arm / 'console.log').open('a', buffering=1)
            handles.append(handle)
            process = subprocess.Popen(spec['command'], cwd=manifest['code_root'],
                                       env=dict(os.environ, **spec['environment']), stdout=handle, stderr=subprocess.STDOUT)
            processes[arm] = process
            status['arms'][arm] = {'pid': process.pid, 'gpu': spec['gpu'], 'started_at_utc': now()}
        while True:
            for arm, process in processes.items():
                code = process.poll()
                if code is not None and 'exit_code' not in status['arms'][arm]:
                    status['arms'][arm].update(exit_code=code, finished_at_utc=now())
            write_json(experiment / 'status.json', status)
            try:
                update_report(experiment, manifest, status)
            except Exception as error:
                print(f'Report update failed (training continues): {error}', flush=True)
            if all(process.poll() is not None for process in processes.values()):
                break
            time.sleep(15)
        status['finished_at_utc'] = now()
        status['exit_code'] = int(any(process.returncode != 0 for process in processes.values()))
        write_json(experiment / 'status.json', status)
        update_report(experiment, manifest, status)
        return status['exit_code']
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--problem', choices=['cvrp', 'vrptw'], default='cvrp')
    parser.add_argument('--customers', type=int, choices=[15, 50, 100], default=50)
    parser.add_argument('--data-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset' / 'dataset')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--eval-interval', type=int, default=20)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--eval-limit', type=int, default=None)
    parser.add_argument('--expert-limit', type=int, default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--output-root', type=Path, default=CODE_ROOT / 'results' / 'optimization')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        raise SystemExit(supervise(args.supervise.resolve()))
    if args.init_checkpoint is None or not args.init_checkpoint.is_file():
        parser.error('--init-checkpoint must name a completed PPO checkpoint')
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not g.isdigit() for g in gpus):
        parser.error('--gpus must contain two distinct GPU indices, e.g. 0,1')
    for key in ('epochs', 'num_envs', 'n_traj', 'eval_interval', 'eval_batch_size'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive')
    if any(value is not None and value <= 0 for value in (args.eval_limit, args.expert_limit)):
        parser.error('--eval-limit and --expert-limit must be positive when supplied')
    name = args.run_id or f'SLPPO_COMPARE_{args.problem.upper()}{args.customers}_S{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    if Path(name).name != name or name in {'.', '..'}:
        parser.error('--run-id must be a single directory name')
    experiment = args.output_root.resolve() / name
    configs = build_configs(args, experiment)
    if args.dry_run:
        print(yaml.safe_dump(configs, sort_keys=False))
        return
    files = [args.init_checkpoint.resolve()]
    for split in ('train', 'val'):
        directory = args.data_root.resolve() / args.problem / split / f'Cus{args.customers}'
        files.extend(directory / name for name in ('instances.pkl', 'metadata.json', 'gurobi_summary.csv'))
        if split == 'train':
            files.append(directory / 'expert_solutions.csv')
    for path in files:
        if not path.is_file():
            parser.error(f'required input missing: {path}')
    if experiment.exists():
        parser.error(f'experiment exists; choose a fresh --run-id: {experiment}')
    git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True).strip()
    if git_status:
        parser.error('commit code before launching so both arms have identifiable source')
    import torch
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
    init_cfg = checkpoint.get('config', {})
    requested = configs['baseline']
    for section, keys in {'data': ('problem_type', 'num_customers'),
                          'model': ('embedding_dim', 'n_encode_layers', 'use_dynamic_decision_encoder',
                                    'dynamic_decision_heads', 'dynamic_decision_delta_k', 'dynamic_decision_delta_v',
                                    'dynamic_decision_delta_action_key', 'dynamic_decision_action_bias')}.items():
        for key in keys:
            if init_cfg.get(section, {}).get(key) != requested[section][key]:
                parser.error(f'PPO checkpoint {section}.{key} does not match comparison config')
    del checkpoint
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip()
    experiment.mkdir(parents=True)
    manifest = {
        'created_at_utc': now(), 'code_root': str(CODE_ROOT), 'git_commit': revision,
        'runtime': {'python': sys.version, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'python_executable': sys.executable},
        'gpu_inventory': subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total', '--format=csv'], text=True),
        'init_checkpoint_sha256': digest(args.init_checkpoint),
        'input_sha256': {str(path): digest(path) for path in files},
        'comparison': 'correctness-fixed baseline vs all optimizations; same PPO initialization, train/val only; one-seed screening',
        'baseline_reference_tag': 'baseline/slppo-preopt-20261002',
        'seed': args.seed, 'arms': {},
    }
    for (arm, cfg), gpu in zip(configs.items(), gpus):
        directory = experiment / arm
        directory.mkdir()
        config_path = directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest['arms'][arm] = {
            'gpu': int(gpu), 'config': str(config_path), 'config_sha256': digest(config_path),
            'command': [sys.executable, '-B', '-u', '-m', 'offline2online.train', '--config', str(config_path),
                        '--seed', str(args.seed), '--device', 'cuda:0'],
            'environment': {'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
                            'NUMBA_CACHE_DIR': str(experiment / arm / 'numba_cache'),
                            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'},
            'log_dir': str(CODE_ROOT / 'results' / 'logs' / f'Cus_{args.customers}_CS_0' / cfg['run_name'] / f'seed_{args.seed}'),
        }
    write_json(experiment / 'manifest.json', manifest)
    with (experiment / 'supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--supervise', str(experiment)],
                                   cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    write_json(experiment / 'launcher.json', {'supervisor_pid': process.pid, 'created_at_utc': now()})
    print(json.dumps({'experiment': str(experiment), 'supervisor_pid': process.pid,
                      'report': str(experiment / 'comparison.json')}, indent=2))


if __name__ == '__main__':
    main()
