#!/usr/bin/env python3
"""Archive a failed B04, restart it from scratch, and adopt the live B01.

This stops only the old supervisor PID, never its process group or PPO worker.
The fresh SL-PPO must pass a full-protocol preflight and all-expert audit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1.native import atomic_json, file_hash
from e2.rrnco_ppo_launch import prepare, now


def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--ppo-pid', type=int, required=True)
    parser.add_argument('--old-supervisor-pid', type=int, required=True)
    args = parser.parse_args()
    camp = args.campaign.resolve()
    ppo_command = command(args.ppo_pid)
    if str(camp/'runs/ppo') not in ppo_command or 'run_e2b_rrnco_worker.py train' not in ppo_command:
        raise RuntimeError('PPO PID does not match the active campaign')
    old_command = command(args.old_supervisor_pid)
    if 'run_e2b_rrnco_a6000.py' not in old_command or str(camp) not in old_command:
        raise RuntimeError('Old supervisor PID does not match the campaign')
    old_status = json.loads((camp/'scheduler_a6000/status.json').read_text())
    failed = old_status['workers']['slppo']
    if failed['state'] != 'failed' or command(failed['pid']):
        raise RuntimeError('SL-PPO must be stopped before archival')
    memory = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(memory.strip()) > 1024:raise RuntimeError('GPU1 is occupied')
    archive = camp/'failed_attempts'/('slppo_missing_final_depot_'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    archive.mkdir(parents=True)
    atomic_json(archive/'supervisor_status.json', old_status)
    atomic_json(archive/'protocol.json', json.loads((camp/'protocol.json').read_text()))
    os.kill(args.old_supervisor_pid, signal.SIGTERM)
    lock = (camp/'launcher.lock').open('a')
    for attempt in range(100):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('Old supervisor did not release campaign lock')
    if command(args.ppo_pid) != ppo_command:raise RuntimeError('PPO process identity changed')
    moves = {}
    for relative in ['runs/slppo', 'preflight/slppo', 'checkpoint_backups/formal/slppo', 'checkpoint_backups/preflight/slppo']:
        src = camp/relative
        if src.exists():
            dst = archive/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            moves[str(src)] = str(dst)
    recovery = dict(reason='Retain explicit final depot in native expert replay',
                    initialization='scratch; no failed-attempt weights or optimizer state reused',
                    archived_paths=moves, original_supervisor=args.old_supervisor_pid,
                    adopted_ppo_pid=args.ppo_pid, started_at=now(),
                    git_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    atomic_json(camp/'slppo_recovery.json', recovery)
    state = dict(state='slppo_preflight', pid=os.getpid(), started_at=now(), recovery=str(camp/'slppo_recovery.json'),
                 workers={'ppo':dict(old_status['workers']['ppo'], adopted=True)})

    def save():
        state['updated_at'] = now()
        atomic_json(camp/'scheduler_a6000/status.json', state)

    save()
    try:
        for mode in ['preflight', 'train']:
            run = prepare(ROOT, camp, 'slppo', preflight=mode=='preflight')
            log = run/('preflight.log' if mode=='preflight' else 'console.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with log.open('a') as output:
                proc = subprocess.Popen([sys.executable, '-u', str(run/'native_harness/scripts/run_e2b_rrnco_worker.py'), mode, '--run-dir', str(run)],
                                        env=env, cwd=run, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state.update(state='slppo_'+mode)
            state['workers']['slppo'] = dict(state='running', mode=mode, gpu=1, pid=proc.pid, run=str(run), log=str(log))
            save()
            while proc.poll() is None:
                accepted = camp/'runs/ppo/test/manifest.json'
                if accepted.exists() and json.loads(accepted.read_text()).get('accepted'):
                    state['workers']['ppo']['state'] = 'completed'
                elif command(args.ppo_pid) != ppo_command:
                    state['workers']['ppo']['state'] = 'failed'
                save()
                time.sleep(10)
            evidence_path = run/('preflight.json' if mode=='preflight' else 'test/manifest.json')
            evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
            if proc.returncode != 0 or not evidence.get('passed' if mode=='preflight' else 'accepted'):
                state['workers']['slppo'].update(state='failed', exit_code=proc.returncode)
                raise RuntimeError(f'SL-PPO {mode} failed; see {log}')
            if mode=='preflight':
                if not evidence['expert_completion_audit']['passed']:raise RuntimeError('Expert audit not passed')
                recovery.update(preflight_sha256=file_hash(evidence_path), expert_audit=evidence['expert_completion_audit'])
                atomic_json(camp/'slppo_recovery.json', recovery)
                protocol = json.loads((camp/'protocol.json').read_text())
                protocol['slppo_recovery'] = recovery
                atomic_json(camp/'protocol.json', protocol)
                reports = json.loads((camp/'smoke_results.json').read_text())
                reports['slppo'] = evidence
                atomic_json(camp/'smoke_results.json', reports)
            else:state['workers']['slppo']['state'] = 'completed'
        state['state'] = 'waiting_for_ppo'
        while state['workers']['ppo']['state'] == 'running':
            accepted = camp/'runs/ppo/test/manifest.json'
            if accepted.exists() and json.loads(accepted.read_text()).get('accepted'):
                state['workers']['ppo']['state'] = 'completed'
            elif command(args.ppo_pid) != ppo_command:state['workers']['ppo']['state'] = 'failed'
            save()
            if state['workers']['ppo']['state'] == 'running':time.sleep(10)
        state['state'] = 'completed' if all(w['state']=='completed' for w in state['workers'].values()) else 'failed'
        save()
    except Exception as exc:
        state.update(state='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':main()
