#!/usr/bin/env python3
"""Supervise two independent native RADAR arms through best-checkpoint test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1.native import atomic_json
from e2.radar_native import prepare_variant, verify_run, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--gpus', nargs=2, default=['0', '1'])
    p.add_argument('--agda-chunk', type=int, default=50)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if len(set(a.gpus)) != 2:
        raise ValueError('Two distinct GPUs required')
    if not a.execute:
        print(json.dumps(dict(gpus=dict(zip(['rdi', 'agda'], a.gpus)),
            campaign=str(a.campaign), baseline=str(a.baseline_run),
            batch=64, n_traj=50, seed=3009, exposures=96000), indent=2))
        return
    a.campaign.mkdir(parents=True, exist_ok=True)
    lock = (a.campaign/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = dict((k.strip(), int(v)) for k,v in (line.split(',') for line in info.splitlines()))
    if any(memory[g] > 1024 for g in a.gpus):
        raise RuntimeError(f'GPU occupied; refusing to start: {memory}')
    git_head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    state = dict(state='preparing', pid=os.getpid(), started_at=now(), git_head=git_head,
        checkout=str(ROOT), gpu_assignment=dict(zip(['rdi', 'agda'], a.gpus)), workers={})
    status_path = a.campaign/'scheduler_a6000/status.json'
    def save():
        state['updated_at'] = now()
        atomic_json(status_path, state)
    save()
    handles = []
    try:
        for mode in ['preflight', 'train']:
            jobs = {}
            for variant,gpu in zip(['rdi', 'agda'], a.gpus):
                run = a.campaign/('preflight' if mode=='preflight' else 'runs')/variant
                if not run.exists():
                    prepare_variant(ROOT, a.baseline_run, a.campaign, variant,
                        chunk=a.agda_chunk, preflight=mode=='preflight')
                cfg,manifest = verify_run(run)
                if mode=='preflight' and (run/'preflight.json').exists():
                    result = json.loads((run/'preflight.json').read_text())
                    if not result['passed'] or result['harness_sha256']!=manifest['harness']['sha256']:
                        raise ValueError('Invalid saved preflight')
                    continue
                if (run/'native_train.jsonl').exists():
                    raise FileExistsError(f'Refusing implicit restart: {run}')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                log = run/('preflight.log' if mode=='preflight' else 'console.log')
                handle = log.open('a'); handles.append(handle)
                worker = run/'native_harness/scripts/run_e2_radar_native.py'
                command = [sys.executable, '-u', str(worker), mode, '--run-dir', str(run)]
                proc = subprocess.Popen(command, cwd=run, env=env, stdout=handle,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                jobs[variant] = (proc,run)
                state['workers'][variant] = dict(pid=proc.pid, gpu=gpu, mode=mode,
                    run=str(run), log=str(log), command=command, state='running')
            state['state'] = 'preflight' if mode=='preflight' else 'training_and_test'; save()
            while jobs:
                for variant,(proc,run) in list(jobs.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    accepted = code==0
                    if accepted and mode=='train':
                        accepted = json.loads((run/'test/manifest.json').read_text()).get('accepted', False)
                    state['workers'][variant].update(exit_code=code,
                        state='completed' if accepted else 'failed', finished_at=now())
                    del jobs[variant]; save()
                if jobs:
                    time.sleep(10)
            if any(v['state']=='failed' for v in state['workers'].values()):
                raise RuntimeError(f'{mode} worker failed; see individual logs')
        state['state']='completed'; save()
    except Exception as exc:
        state.update(state='failed', error=repr(exc)); save()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__=='__main__':
    main()
