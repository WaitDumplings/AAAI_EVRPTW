#!/usr/bin/env python3
"""Wait for completed controlled E1 runs, test selected best, and merge reports.

A JSON plan lists jobs (campaign, method, scheduler, GPUs, report_dir) and shared
report paths. --execute detaches the watcher; --once only checks readiness.
Never retries partial tests or changes frozen training configurations.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1.evaluation import atomic_json, summarize_rows
from e1.configs import digest


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def csv_read(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def csv_write(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(stream.getvalue())
    temporary.replace(path)


def job_info(job):
    import yaml
    manifest = read(Path(job['campaign']) / 'manifest.json')
    spec = manifest['methods'][job['method']]
    cfg = yaml.safe_load(Path(spec['config']).read_text())
    from e1.lifecycle import _checkpoint_dir
    return manifest, spec, cfg, _checkpoint_dir(manifest, cfg) / 'checkpoint_best.pt'


def readiness(job, spec):
    status_path = Path(job['scheduler']) / 'status.json'
    if not status_path.exists():
        return 'waiting_for_scheduler'
    scheduler = read(status_path)
    state = scheduler.get('methods', {}).get(job['method'], {}).get('state', 'unknown')
    if state in ('failed', 'blocked', 'interrupted'):
        return 'blocked_training_' + state
    if state != 'completed':
        return 'waiting_for_training'
    result = Path(spec['run_dir']) / 'training_result.json'
    if not result.exists() or read(result).get('state') != 'completed':
        return 'waiting_for_completion_marker'
    return 'ready'


def merge_report(plan, job, summary, evidence):
    """Merge only this problem/scale/method; retain remote results and missing fields."""
    overview = Path(plan['overview'])
    with (overview.parent / '.test_report.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = Path(job['report_dir'])
        main = report / 'main_results_partial.csv'
        fields, rows = csv_read(main)
        matches = [r for r in rows if r['method'] == job['label']]
        if len(matches) != 1:
            raise ValueError('Expected exactly one report method row')
        row = matches[0]
        fmt = lambda value: 'TBD' if value is None else str(value)
        row.update(status='Training completed; independent test accepted; val-selected epoch' + str(evidence['checkpoint_epoch']),
                   feasible_count=str(summary['feasible_count']), test_count=str(summary['instances']),
                   all_feasible_mean_km=fmt(summary['mean_distance_km']),
                   runtime_s_per_instance=fmt(summary['mean_runtime_s']))
        ofields, orows = csv_read(overview)
        targets = [r for r in orows if r['problem'] == job['problem']
                   and int(r['customers']) == job['customers'] and r['method'] == job['label']]
        if len(targets) != 1:
            raise ValueError('Expected exactly one overview row')
        targets[0].update(row)
        csv_write(main, fields, rows)
        csv_write(overview, ofields, orows)
        partial = report / 'test_results_partial.csv'
        pf = ['method','status','feasible_count','test_count','mean_km','runtime_s_per_instance','common_mean_km','common_gap_pct']
        pf, prows = csv_read(partial) if partial.exists() else (pf, [])
        pr = next((r for r in prows if r['method'] == job['label']), None)
        if pr is None:
            pr = dict.fromkeys(pf, 'TBD'); prows.append(pr)
        pr.update({k: row[k] for k in ('method','status','feasible_count','test_count','runtime_s_per_instance','common_mean_km')})
        pr.update(mean_km=row['all_feasible_mean_km'], common_gap_pct=row['common_gap_to_gurobi_pct'])
        csv_write(partial, pf, prows)
        ep = report / 'evidence.json'
        doc = read(ep) if ep.exists() else {}
        doc.update(updated_at=utc(), rows=rows)
        doc.setdefault('automated_tests', {})[job['label']] = evidence
        atomic_json(ep, doc)
        if plan.get('renderer'):
            subprocess.run([sys.executable, plan['renderer'], '--problem', job['problem']], check=True)


def accept_and_publish(plan, job, manifest, spec, cfg, checkpoint, execution):
    from e1.runner import verify_campaign
    from e1.lifecycle import assert_accepted_test_result, assert_test_ready
    verify_campaign(job['campaign'])
    assert_test_ready(manifest, job['method'], checkpoint)
    if manifest['schema'] == 'aaai_e1_vrptw100_v1' and 'splits' not in read(manifest['audit_path']):
        from run_e1_vrptw_test import assert_accepted
        accepted = assert_accepted(manifest, job['method'])
    else:
        accepted = assert_accepted_test_result(manifest, job['method'])
    folder = Path(spec['run_dir']) / 'test'
    rows = [json.loads(line) for line in (folder / 'instances.jsonl').read_text().splitlines() if line.strip()]
    summary = summarize_rows(rows)
    import torch
    epoch = torch.load(checkpoint, map_location='cpu', weights_only=False)['epoch']
    evidence = dict(campaign=job['campaign'], config=spec['config'], checkpoint=str(checkpoint),
                    checkpoint_epoch=epoch, checkpoint_sha256=digest(checkpoint),
                    acceptance=str(folder / 'manifest.json'), instances=str(folder / 'instances.jsonl'),
                    test_data_sha256=accepted['test_data_sha256'], result=summary,
                    training_seed=manifest['training_seed'], K=50, inference_batch_size=plan['batch_size'],
                    timing='Batch-amortized candidate generation, validation and best feasible selection',
                    execution=execution, verified_at=utc(), training_time_status='TBD; cumulative wall time not inferred from epoch timing')
    merge_report(plan, job, summary, evidence)
    return summary


def poll(plan, status, folder, execute):
    from e1.runner import _runtime
    cards = _runtime().gpu_snapshot() if execute else {}
    for job in plan['jobs']:
        key = job['id']; item = status['jobs'].setdefault(key, {})
        if item.get('state') in ('completed', 'blocked'):
            continue
        try:
            manifest, spec, cfg, checkpoint = job_info(job)
            test = Path(spec['run_dir']) / 'test'
            state = readiness(job, spec)
            item.update(state=state, checked_at=utc(), checkpoint=str(checkpoint))
            if state.startswith('blocked_'):
                item.update(state='blocked', error=state); continue
            if state != 'ready':
                continue
            if test.exists():
                if not (test / 'manifest.json').exists():
                    item.update(state='blocked', error='Partial test directory exists; inspect before retry'); continue
                if execute:
                    item['result'] = accept_and_publish(plan, job, manifest, spec, cfg, checkpoint, item.get('execution', {}))
                    item.update(state='completed', finished_at=utc())
                else:
                    item['state'] = 'existing_test_needs_acceptance_check'
                continue
            if not execute:
                continue
            gpu = next((g for g in job['gpus'] if g in cards and _runtime().idle_gpu(cards[g])), None)
            if gpu is None:
                item['state'] = 'waiting_for_idle_gpu'; continue
            entry = ([str(ROOT / 'scripts/run_e1_vrptw_test.py')]
                     if manifest['schema'] == 'aaai_e1_vrptw100_v1' and 'splits' not in read(manifest['audit_path']) else ['-m', 'e1.runner', 'evaluate'])
            cmd = [sys.executable, *entry, '--campaign', job['campaign'],
                   '--method', job['method'], '--checkpoint', str(checkpoint), '--device', f'cuda:{gpu}',
                   '--batch-size', str(plan['batch_size'])]
            log_path = folder / (key + '.test.log')
            item.update(state='testing', execution=dict(command=cmd, gpu=gpu, hardware=cards[gpu],
                        started_at=utc(), log=str(log_path)))
            atomic_json(folder / 'status.json', status)
            env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            env.pop('CUDA_VISIBLE_DEVICES', None)
            with log_path.open('a') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            item['execution'].update(returncode=result.returncode, ended_at=utc())
            if result.returncode:
                if not test.exists():
                    # Only resource contention is retried automatically, never invalid configs.
                    tail = log_path.read_text()[-12000:]
                    if any(message in tail for message in ('Requested evaluation GPU is busy', 'Evaluation GPU became reserved/busy')):
                        item.update(state='waiting_to_retry', error='GPU became busy; retry after it is idle')
                        continue
                raise RuntimeError('Test process failed; partial artifacts preserved; see ' + str(log_path))
            item['result'] = accept_and_publish(plan, job, manifest, spec, cfg, checkpoint, item['execution'])
            item.update(state='completed', finished_at=utc())
            cards = _runtime().gpu_snapshot()
        except Exception as error:
            item.update(state='blocked', error=f'{type(error).__name__}: {error}', checked_at=utc())
        finally:
            status['updated_at'] = utc()
            atomic_json(folder / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--once', action='store_true')
    args = parser.parse_args(); path = args.plan.resolve(); plan = read(path)
    folder = path.parent
    if args.execute:
        with (folder / 'watcher.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--plan', str(path), '--worker'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(folder / 'pid.json', dict(pid=process.pid, started_at=utc(), plan=str(path)))
        print(f'Watcher PID {process.pid}; {folder / "status.json"}')
        return
    # One watcher per plan, persistent across terminal disconnects; never mutate frozen source.
    with (folder / 'watcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read(folder / 'status.json') if (folder / 'status.json').exists() else dict(jobs={})
        status.update(pid=os.getpid(), started_at=utc(), state='watching')
        while True:
            poll(plan, status, folder, execute=args.worker)
            states = [item['state'] for item in status['jobs'].values()]
            done = all(s in ('completed', 'blocked') for s in states)
            status.update(state=('finished_with_errors' if 'blocked' in states else 'completed') if done else 'watching', updated_at=utc())
            atomic_json(folder / 'status.json', status)
            if args.once or done:
                print(json.dumps(status, indent=2)); break
            time.sleep(plan.get('poll_seconds', 60))


if __name__ == '__main__':
    main()
