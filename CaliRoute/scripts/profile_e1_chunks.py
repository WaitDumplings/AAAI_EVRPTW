#!/usr/bin/env python3
"""Isolated, sequential single-GPU chunk timing from a frozen E1 configuration.

This never resumes or modifies formal runs. It preserves batch, trajectories,
optimizer geometry and precision. Results are hardware probes, not E1 results.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--method', choices=['ppo_base', 'ppo_rdi_agda', 'awbc', 'dapg', 'slppo'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--chunks', type=int, nargs='+', default=[8, 32, 64, 128])
    p.add_argument('--epochs', type=int, default=6, help='Online probe loops; first excluded from timing')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.epochs < 3 or not a.chunks or min(a.chunks) < 1 or len(set(a.chunks)) != len(a.chunks):
        p.error('Need >=3 epochs and unique positive chunks')
    manifest = json.loads((a.campaign / 'manifest.json').read_text())
    spec = manifest['methods'][a.method]
    original_bytes = Path(spec['config']).read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != spec['config_sha256']:
        raise ValueError('Frozen configuration hash mismatch')
    cfg = yaml.safe_load(original_bytes)
    if cfg['experiment_protocol']['world_size'] != 1:
        raise ValueError('This probe supports single-GPU configurations only; do not silently change global batch')
    cards = subprocess.check_output(['nvidia-smi', f'--id={a.gpu}', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    name, uuid, memory = [x.strip() for x in cards.split(',')]
    def require_idle():
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if uuid in apps:
            raise RuntimeError(f'GPU {a.gpu} has an existing compute process; left untouched')
    require_idle()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source' / 'CaliRoute'
    # Copy exactly the audited source, excluding all original result artifacts.
    for relative, expected in manifest['source']['files'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] == 'results':
            raise ValueError(f'Unsafe source entry: {relative}')
        data = (Path(manifest['code_root']) / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {relative}')
        dst = source / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    (root / 'artifacts').mkdir()
    (source / 'results').symlink_to(root / 'artifacts', target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=str(a.gpu),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    report = dict(purpose='hardware_probe_only_not_E1_training', method=a.method,
                  campaign=str(a.campaign.resolve()), config_sha256=spec['config_sha256'],
                  source_content_sha256=manifest['source']['content_sha256'],
                  gpu=dict(index=a.gpu, name=name, uuid=uuid, total_mib=float(memory)),
                  same_scratch_seed=True, evaluation_disabled=True, first_online_loop_excluded=True,
                  protocol={k: cfg['training'][k] for k in ['num_envs_per_gpu', 'n_traj', 'ppo_update_epochs', 'num_minibatches', 'gradient_accumulation_steps', 'mixed_precision', 'learning_rate']},
                  candidates=[])
    dump(root / 'summary.json', report)
    for chunk in a.chunks:
        require_idle()
        run = root / f'chunk_{chunk}'
        run.mkdir()
        probe = copy.deepcopy(cfg)
        probe['run_name'] = f'PROBE_{a.method}_chunk{chunk}'
        warmup = int(probe['offline'].get('bc_warmup_epochs', 0))
        probe['training'].update(epochs=warmup+a.epochs, ppo_step_chunk_size=chunk,
            monitor_interval=1, monitor_output_dir=str(run / 'monitoring'),
            checkpoint_interval=0, latest_checkpoint_interval=0, post_update_kl_interval=0)
        probe['training'].pop('checkpoint_backup_dir', None)
        probe['evaluation'].update(eval_interval=0, eval_before_training=False, eval_output_dir=str(run / 'validation'))
        probe['experiment_protocol'].update(purpose='hardware_probe_only', total_online_loops=a.epochs,
            total_outer_loops=warmup+a.epochs, total_instance_exposures=a.epochs*cfg['experiment_protocol']['global_batch'],
            total_online_trajectory_attempts=a.epochs*cfg['experiment_protocol']['global_batch']*cfg['training']['n_traj'],
            checkpoint_backup_required=False)
        path = run / 'config.yaml'
        path.write_text(yaml.safe_dump(probe, sort_keys=False))
        cmd = [sys.executable, '-m', 'e1.worker', '--config', str(path), '--method', a.method, '--device', 'cuda']
        started = time.time()
        with (run / 'train.log').open('w') as log:
            child = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
        item = dict(chunk=chunk, exit_code=code, process_wall_s=time.time()-started, log=str(run / 'train.log'))
        monitor = run / 'monitoring' / 'monitor_rank_0.jsonl'
        rows = [json.loads(x) for x in monitor.read_text().splitlines()] if monitor.exists() else []
        online = [r for r in rows if int(r.get('online_epoch', 0)) > 1]
        if code == 0 and len(online) == a.epochs-1:
            item.update(state='passed', median_train_s=statistics.median(r['distributed_train_wall_time_s'] for r in online),
                        median_ppo_s=statistics.median(r['ppo_update_time_s'] for r in online),
                        peak_reserved_mib=max(r['gpu_peak_reserved_bytes'] for r in rows)/2**20,
                        peak_allocated_mib=max(r['gpu_peak_allocated_bytes'] for r in rows)/2**20,
                        optimizer_steps_per_loop=sorted(set(r['optimizer_steps_epoch'] for r in online)))
            item['reserved_fraction'] = item['peak_reserved_mib']/float(memory)
        else:
            failure_log = (run / 'train.log').read_text(errors='replace')
            item['state'] = 'oom' if 'CUDA out of memory' in failure_log else 'failed_or_incomplete'
        report['candidates'].append(item)
        eligible = [r for r in report['candidates'] if r['state']=='passed' and r['reserved_fraction'] < .90]
        report['fastest_measured_chunk'] = min(eligible, key=lambda r:r['median_train_s'])['chunk'] if eligible else None
        best_time = min((r['median_train_s'] for r in eligible), default=None)
        near_best = [r for r in eligible if r['median_train_s'] <= best_time * 1.03] if eligible else []
        report['smallest_chunk_within_3pct_of_fastest'] = min((r['chunk'] for r in near_best), default=None)
        report['selection_note'] = 'Short scratch timing only; confirm mature-checkpoint timing and loss/gradient equivalence before formal adoption. No 80% occupancy requirement.'
        dump(root / 'summary.json', report)
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
