#!/usr/bin/env python3
"""Single-GPU VRPTW100 sweep: shared PPO initialization, update passes 3/4/5/6."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
from run_cus100_finetune import refresh_progress
from run_plugin_comparison import append_hardware, terminate_group
from run_slppo_comparison import digest, now, read_csv, write_json


def build_arm(base, *, updates, output, run_name, init_checkpoint, chunk_size):
    if updates not in (3, 4, 5, 6):
        raise ValueError('Expected 3, 4, 5, or 6 PPO update passes')
    if base['data']['problem_type'] != 'vrptw' or base['data']['num_customers'] != 100:
        raise ValueError('Expected a VRPTW100 source configuration')
    cfg = copy.deepcopy(base)
    cfg['run_name'] = run_name
    cfg['training'].update(epochs=500, num_envs_per_gpu=64, num_minibatches=4,
        gradient_accumulation_steps=1, ppo_update_epochs=updates,
        ppo_step_chunk_size=chunk_size, learning_rate=3e-5, amp_init_scale=1024.,
        monitor_output_dir=str(output / 'monitoring'))
    cfg['offline'].update(init_checkpoint_path=str(init_checkpoint), sl_coef=.35)
    # Always weights-only initialization: never import an arm's optimizer/replay state.
    for section in ('training', 'offline'):
        for key in list(cfg[section]):
            if key.startswith('resume_'):
                del cfg[section][key]
    cfg['evaluation'].update(eval_interval=20, eval_before_training=True,
        eval_output_dir=str(output / 'evaluations'))
    cfg['experiment_protocol'] = dict(phase='ppo_update_sweep', world_size=1,
        seed=3009, global_instances_per_rollout=64, global_trajectories_per_rollout=3200,
        global_instances_per_optimizer_step=16, ppo_update_epochs=updates,
        attempted_optimizer_steps_per_epoch=updates * 4, ppo_step_chunk_size=chunk_size,
        eval_batch_size=32, initialization='shared_PPO_init_weights_only',
        selection_split='val', schedule_scope='500_epoch_warmup_cosine',
        comparison_scope='single_seed_single_GPU_update_pass_ablation; not a paired reproduction of prior two-rank batches')
    return cfg


def successful_training(spec, detail):
    rows = read_csv(Path(spec['log_dir']) / 'train_log.csv')
    epochs = [int(row['epoch']) for row in rows]
    evaluation = detail.get('latest_validation') or {}
    return (epochs == list(range(1, 501))
        and int(evaluation.get('epoch', -1)) == 500
        and evaluation.get('eval_status') == 'ok'
        and int(float(evaluation.get('eval_num_instances', 0))) == 1000
        and (Path(spec['checkpoint_dir']) / 'checkpoint_final.pt').is_file())


def comparison_report(manifest, status):
    keys = ('state', 'stage', 'gpu', 'ppo_update_epochs', 'completed_training_epochs',
            'target_epochs', 'latest_validation', 'best_checkpoint',
            'median_training_seconds_excluding_eval_and_first_epoch', 'test_summary', 'error')
    metrics = ('epoch', 'optimizer_steps', 'optimizer_steps_epoch', 'amp_skipped_steps',
               'global_approx_kl', 'clip_fraction', 'global_entropy', 'grad_norm',
               'grad_clipped_fraction', 'learning_rate', 'global_train_feasible_rate',
               'run_elapsed_seconds', 'distributed_train_wall_time_s')
    arms = {}
    for arm, detail in status['arms'].items():
        arms[arm] = {k: detail[k] for k in keys if k in detail}
        row = detail.get('latest_train_row') or {}
        arms[arm]['latest_training_metrics'] = {k: row[k] for k in metrics if k in row}
    return dict(updated_at_utc=now(), state=status['state'], protocol=manifest['protocol'],
        arms=arms, selection_scope='Use validation to tune; test summaries are final reporting only')


def supervise(experiment):
    manifest = json.loads((experiment / 'manifest.json').read_text())
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip() != manifest['git_commit']:
        raise RuntimeError('Source commit changed')
    if digest(Path(manifest['init_checkpoint'])) != manifest['init_checkpoint_sha256']:
        raise RuntimeError('Shared PPO initialization changed')
    for path, checksum in manifest['input_sha256'].items():
        if digest(Path(path)) != checksum:
            raise RuntimeError(f'Dataset/reference changed: {path}')
    status = dict(state='running', started_at_utc=now(), supervisor_pid=os.getpid(), arms={})
    processes, streams = {}, []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        for proc in processes.values():
            terminate_group(proc)

    def launch(arm, spec, *, test=False):
        out = Path(spec['output_dir']) / 'test_best' if test else Path(spec['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        stream = (out / 'console.log').open('a', buffering=1)
        streams.append(stream)
        command = spec['test_command'] if test else spec['command']
        proc = subprocess.Popen(command, cwd=CODE_ROOT, env=dict(os.environ, **spec['environment']),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes[arm] = proc
        status['arms'][arm].update(pid=proc.pid, stage='test_best' if test else 'training', stage_started_at_utc=now())
        print(f'{now()} {arm} {status["arms"][arm]["stage"]} pid={proc.pid}', flush=True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for arm, spec in manifest['arms'].items():
            if digest(Path(spec['config'])) != spec['config_sha256']:
                raise RuntimeError(f'Configuration changed: {arm}')
            status['arms'][arm] = dict(state='running', gpu=spec['gpu'], ppo_update_epochs=spec['ppo_update_epochs'])
            launch(arm, spec)
        while True:
            for arm, spec in manifest['arms'].items():
                detail = status['arms'][arm]
                if detail['state'] != 'running':
                    continue
                proc = processes[arm]
                code = proc.poll()
                refresh_progress(spec, detail)
                detail['rank_monitors'] = {'0': detail.get('rank_monitors', {}).get('0')}
                if code is None:
                    continue
                detail['exit_code'] = code
                if interrupted:
                    detail['state'] = 'interrupted'
                elif code != 0:
                    detail.update(state='failed', error=f'{detail["stage"]} exited {code}; inspect console.log')
                elif detail['stage'] == 'training':
                    if not successful_training(spec, detail):
                        detail.update(state='failed', error='Missing complete 500 epochs, final validation, or checkpoint')
                    else:
                        detail['training_finished_at_utc'] = now()
                        launch(arm, spec, test=True)
                else:
                    summary = json.loads((Path(spec['output_dir']) / 'test_best' / 'summary.json').read_text())
                    detail.update(state='completed', test_summary=summary, finished_at_utc=now())
            status['updated_at_utc'] = now()
            done = all(d['state'] != 'running' for d in status['arms'].values())
            if done:
                status['state'] = ('interrupted' if interrupted else 'completed'
                    if all(d['state'] == 'completed' for d in status['arms'].values()) else 'failed')
                status['finished_at_utc'] = now()
            write_json(experiment / 'status.json', status)
            write_json(experiment / 'comparison.json', comparison_report(manifest, status))
            try:
                append_hardware(experiment)
            except Exception as error:
                print(f'Hardware monitor: {error}', flush=True)
            if done:
                break
            if interrupted:
                for proc in processes.values():
                    terminate_group(proc)
            time.sleep(10)
    except Exception as error:
        traceback.print_exc()
        status.update(state='failed', error=f'{type(error).__name__}: {error}', finished_at_utc=now())
        write_json(experiment / 'status.json', status)
        raise
    finally:
        deadline = time.monotonic() + 65
        while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
            for proc in processes.values():
                terminate_group(proc)
            time.sleep(.25)
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--chunk-size', type=int, default=18)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervise', type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise.resolve())
    if not args.source_experiment or not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error('Provide --source-experiment and a fresh --run-id directory name')
    gpus = [int(v) for v in args.gpus.split(',')]
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0 or not 1 <= args.chunk_size <= 201:
        parser.error('Four distinct GPUs and a positive chunk size <= 201 are required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True).strip():
        parser.error('Commit code and use a frozen checkout before launching')
    source = args.source_experiment.resolve()
    previous = json.loads((source / 'manifest.json').read_text())
    if previous['seed'] != 3009:
        parser.error('This controlled sweep uses seed 3009')
    base = yaml.safe_load(Path(previous['tasks']['vrptw']['phases']['long_candidate']['config']).read_text())
    init = Path(previous['tasks']['vrptw']['init_checkpoint'])
    experiment = CODE_ROOT / 'results' / 'optimization' / args.run_id
    experiment.mkdir(parents=True, exist_ok=False)
    shared = experiment / 'inputs' / 'ppo_init_best.pt'
    shared.parent.mkdir()
    shutil.copy2(init, shared)
    arms = {}
    for updates, gpu in zip((3, 4, 5, 6), gpus):
        arm = f'update_{updates}'
        output = experiment / arm
        output.mkdir()
        name = args.run_id + '_' + arm.upper()
        cfg = build_arm(base, updates=updates, output=output, run_name=name,
            init_checkpoint=shared, chunk_size=args.chunk_size)
        config = output / 'config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        ckpt = CODE_ROOT / 'results/checkpoints/Cus_100_CS_0' / name / 'seed_3009'
        arms[arm] = dict(gpu=gpu, ppo_update_epochs=updates, epochs=500,
            output_dir=str(output), config=str(config), config_sha256=digest(config),
            log_dir=str(CODE_ROOT / 'results/logs/Cus_100_CS_0' / name / 'seed_3009'), checkpoint_dir=str(ckpt),
            command=[sys.executable, '-B', '-u', '-m', 'offline2online.train', '--config', str(config), '--seed', '3009', '--device', 'cuda:0'],
            test_command=[sys.executable, '-B', '-u', str(CODE_ROOT / 'scripts/evaluate_cus100_best.py'),
                '--checkpoint', str(ckpt / 'checkpoint_best.pt'), '--output-dir', str(output / 'test_best'),
                '--test-root', previous['test_root'], '--gurobi-root', previous['gurobi_root'], '--problem', 'vrptw', '--seed', '3009'],
            environment=dict(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=str(output / 'numba_cache'),
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
    inputs = {path: checksum for path, checksum in previous['input_sha256'].items() if '/vrptw/' in path}
    manifest = dict(created_at_utc=now(), code_root=str(CODE_ROOT), source_experiment=str(source),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip(),
        init_checkpoint=str(shared), init_checkpoint_sha256=digest(shared), source_init_checkpoint=str(init),
        input_sha256=inputs, arms=arms,
        protocol=dict(task='vrptw100', epochs=500, seed=3009, updates=[3, 4, 5, 6],
            single_gpu_per_arm=True, global_batch=64, n_traj=50, num_minibatches=4,
            sl_coef=.35, learning_rate=3e-5, lr_min=1e-5, warmup_epochs=20, amp_init_scale=1024.,
            lr_schedule='cosine over 500 epochs', eval_interval=20, validation_instances=1000,
            eval_n_traj=50, init='identical PPO init weights, fresh optimizer and replay',
            scope='Only PPO update passes vary across arms; use elapsed time as well as epochs',
            tuning_rationale='Prior VRPTW epochs 101-300 KL mean 0.0301 and p95 0.0742; lower shared peak LR from 5e-5 to 3e-5. This is a hypothesis to test, not a demonstrated accuracy gain. Keep architecture, SL=.35, entropy .01->.002 and max_grad_norm=1. AMP scale 4096->1024 only affects numerical scaling.'))
    write_json(experiment / 'manifest.json', manifest)
    write_json(experiment / 'status.json', dict(state='prepared', arms={}))
    if not args.prepare_only:
        with (experiment / 'supervisor.log').open('a', buffering=1) as log:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--supervise', str(experiment)],
                cwd=CODE_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (experiment / 'supervisor.pid').write_text(str(proc.pid) + '\n')
    print(experiment, flush=True)


if __name__ == '__main__':
    main()
