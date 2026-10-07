#!/usr/bin/env python3
"""Queue full-state continuation of an existing reward/norm comparison.

The source finishes its current budget unchanged; its final checkpoints then
resume at the next epoch up to the requested TOTAL budget. No source job is
stopped, and no preflight or weights-only initialization is repeated.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
from run_reward_norm_comparison import verify_manifest, supervise
from run_slppo_comparison import digest, now, write_json
from reward_norm_extension import (validate_extension_checkpoint, import_single_gpu_history,
    positive_integer, validation_epochs, required_validation_epochs)


def select_source(experiment, seed):
    if experiment is not None:
        return experiment.resolve()
    if seed is None:
        raise ValueError('Specify --experiment /path/to/run or --seed for the unique active run')
    candidates = []
    for path in (CODE_ROOT / 'results/optimization').glob('REWARD_NORM_*'):
        try:
            manifest = json.loads((path / 'manifest.json').read_text())
            status = json.loads((path / 'status.json').read_text())
        except (OSError, ValueError):
            continue
        if manifest.get('protocol', {}).get('seed') == seed and status.get('state') in {'running', 'waiting_gpu', 'prepared'}:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f'Expected one active reward/norm experiment for seed {seed}; found {candidates}. Pass --experiment explicitly.')
    return candidates[0].resolve()


def validate_plan(manifest, target):
    old_target = positive_integer(manifest['protocol']['epochs'], 'source epochs')
    if positive_integer(target, 'total epochs') <= old_target:
        raise ValueError('Total epochs must exceed the source budget')
    if int(manifest['protocol']['world_size_per_arm']) != 1:
        raise ValueError('This extension supports the single-GPU-per-arm reward/norm protocol')
    for spec in manifest['arms'].values():
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        train = cfg['training']
        interval = positive_integer(cfg['evaluation']['eval_interval'], 'eval_interval')
        if manifest['protocol'].get('eval_interval', interval) != interval:
            raise ValueError('Source arm evaluation intervals differ from the protocol')
        if int(spec['epochs']) != old_target or int(train['epochs']) != old_target:
            raise ValueError('Source arm budgets differ')
        if train.get('lr_schedule') != 'constant' or train.get('lr_warmup_epochs', 0) != 0:
            raise ValueError('Budget extension requires constant LR without warmup')
        if train.get('entropy_initial_coef') != train.get('entropy_final_coef'):
            raise ValueError('Budget extension requires a constant entropy schedule')
    verify_manifest(manifest)


def copy_source(source_manifest, destination):
    """Keep original training code; update only continuation orchestration."""
    hashes = {}
    original = Path(source_manifest['code_root'])
    for name, checksum in source_manifest['source']['files'].items():
        src, dst = original / name, destination / name
        if digest(src) != checksum:
            raise ValueError(f'Source changed: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = digest(dst)
        if hashes[name] != checksum:
            raise ValueError(f'Source changed while copying: {src}')
    for filename in ('extend_reward_norm_comparison.py', 'reward_norm_extension.py', 'run_reward_norm_comparison.py'):
        src, dst = CODE_ROOT / 'scripts' / filename, destination / 'scripts' / filename
        shutil.copy2(src, dst)
        hashes['scripts/' + filename] = digest(dst)
    (destination / 'results').symlink_to(CODE_ROOT / 'results', target_is_directory=True)
    return dict(files=hashes, content_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True),
        training_source_git_commit=source_manifest['source']['git_commit'],
        training_source_content_sha256=source_manifest['source']['content_sha256'],
        orchestration_overrides=['scripts/' + n for n in ('extend_reward_norm_comparison.py', 'reward_norm_extension.py', 'run_reward_norm_comparison.py')])


def prepare_extension(args):
    source = select_source(args.experiment, args.seed)
    with (source / 'extension_prepare.lock').open('a+') as reservation:
        fcntl.flock(reservation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare_extension(args, source)


def _prepare_extension(args, source):
    old_manifest = json.loads((source / 'manifest.json').read_text())
    validate_plan(old_manifest, args.epochs)
    if (source / 'extension.json').exists():
        previous = json.loads((source / 'extension.json').read_text())
        raise ValueError(f'An extension is already recorded: {previous["experiment"]}; inspect its status before creating another')
    seed = old_manifest['protocol']['seed']
    run_id = args.run_id or f'REWARD_NORM_VRPTW100_S{seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be a fresh directory name')
    experiment = CODE_ROOT / 'results/optimization' / run_id
    experiment.mkdir(parents=True, exist_ok=False)
    source_metadata = copy_source(old_manifest, experiment / 'source/CaliRoute')
    plan = dict(created_at_utc=now(), source_experiment=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'),
        source_target_epochs=old_manifest['protocol']['epochs'], target_epochs=args.epochs,
        seed=seed, source=source_metadata, code_root=str(experiment / 'source/CaliRoute'),
        output_root=str(CODE_ROOT), run_id=run_id, poll_seconds=10,
        strategy='finish source budget, then full-state resume to total target; preserve original training code')
    write_json(experiment / 'extension_plan.json', plan)
    write_json(experiment / 'status.json', dict(state='waiting_for_source', target_epochs=args.epochs,
        source_experiment=str(source), arms={a:dict(state='waiting_for_resume', target_epochs=args.epochs) for a in old_manifest['arms']}))
    write_json(source / 'extension.json', dict(experiment=str(experiment), target_epochs=args.epochs, created_at_utc=now()))
    if not args.prepare_only:
        with (experiment / 'supervisor.log').open('a', buffering=1) as log:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(experiment / 'source/CaliRoute/scripts/extend_reward_norm_comparison.py'),
                '--supervise', str(experiment)], cwd=experiment / 'source/CaliRoute',
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (experiment / 'supervisor.pid').write_text(str(process.pid)+'\n')
    print(experiment, flush=True)
    return experiment


def prepare_continuation(experiment, plan, old):
    import torch
    verify_manifest(old)
    manifest = copy.deepcopy(old)
    manifest.update(created_at_utc=now(), code_root=plan['code_root'], source=plan['source'], arms={},
        wait_for_experiments=[], resume_source=plan['source_experiment'], resume_history={})
    manifest['protocol'].update(epochs=plan['target_epochs'],
        stage='extended fine-tuning comparison; total budget includes imported source epochs',
        continuation=dict(source_epochs=plan['source_target_epochs'], additional_epochs=plan['target_epochs']-plan['source_target_epochs'],
            mode='full optimizer, RNG, sampler, replay and normalization state resume'))
    for arm, old_spec in old['arms'].items():
        output = experiment / arm
        output.mkdir()
        cfg = yaml.safe_load(Path(old_spec['config']).read_text())
        source_checkpoint = Path(old_spec['checkpoint_dir']) / 'checkpoint_final.pt'
        before = digest(source_checkpoint)
        checkpoint = output / 'resume_inputs/checkpoint_source_final.pt'
        checkpoint.parent.mkdir()
        shutil.copy2(source_checkpoint, checkpoint)
        if digest(checkpoint) != before or digest(source_checkpoint) != before:
            raise ValueError(f'Final checkpoint changed during copy: {arm}')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        epoch = validate_extension_checkpoint(payload, cfg, plan['target_epochs'], plan['seed'])
        del payload
        run_name = plan['run_id'] + '_' + arm.upper()
        cfg['run_name'] = run_name
        cfg['training'].update(epochs=plan['target_epochs'], resume_append_logs=True, resume_truncate_logs=True,
            monitor_output_dir=str(output / 'monitoring'))
        cfg['offline'].update(resume_checkpoint_path=str(checkpoint), resume_checkpoint_strict=True)
        cfg['evaluation']['eval_output_dir'] = str(output / 'evaluations')
        inherited_evals = required_validation_epochs(old_spec, epoch)
        cfg['experiment_protocol'].update(epochs=plan['target_epochs'], continuation_from_epoch=epoch,
            inherited_validation_epochs=inherited_evals,
            initialization='full-state continuation of the original shared-initialization comparison')
        config_path = output / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        root = Path(plan['output_root'])
        spec = dict(config=str(config_path), config_sha256=digest(config_path), output_dir=str(output),
            epochs=plan['target_epochs'], restored_through_epoch=epoch, validation_instances=1000,
            required_validation_epochs=validation_epochs(plan['target_epochs'], cfg['evaluation']['eval_interval'], inherited_evals),
            log_dir=str(root / 'results/logs/Cus_100_CS_0' / run_name / f'seed_{plan["seed"]}'),
            checkpoint_dir=str(root / 'results/checkpoints/Cus_100_CS_0' / run_name / f'seed_{plan["seed"]}'),
            resume_checkpoint=str(checkpoint), resume_checkpoint_sha256=before,
            command=[sys.executable, '-B', '-u', '-m', 'offline2online.train', '--config', str(config_path), '--seed', str(plan['seed']), '--device', 'cuda:0'])
        manifest['resume_history'][arm] = import_single_gpu_history(old_spec, spec, epoch)
        manifest['arms'][arm] = spec
    write_json(experiment / 'manifest.json', manifest)
    write_json(experiment / 'status.json', dict(state='prepared', arms={a:dict(state='prepared') for a in manifest['arms']}))
    verify_manifest(manifest)
    return manifest


def supervise_extension(experiment):
    lock = (experiment / 'extension.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((experiment / 'extension_plan.json').read_text())
    source = Path(plan['source_experiment'])
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            if digest(source / 'manifest.json') != plan['source_manifest_sha256']:
                raise ValueError('Source experiment manifest changed after extension was requested')
            original = json.loads((source / 'status.json').read_text())
            status = dict(state='waiting_for_source', target_epochs=plan['target_epochs'], source_target_epochs=plan['source_target_epochs'], supervisor_pid=os.getpid(),
                updated_at_utc=now(), source_experiment=str(source), source_state=original['state'], arms={})
            for arm, detail in original['arms'].items():
                status['arms'][arm] = dict(detail, state='waiting_for_resume', source_state=detail['state'],
                    stage='source_training', target_epochs=plan['target_epochs'], source_target_epochs=plan['source_target_epochs'])
            if stopped:
                status.update(state='interrupted', finished_at_utc=now())
                write_json(experiment / 'status.json', status)
                return
            if original['state'] in {'failed', 'interrupted'} or any(d['state'] in {'failed', 'interrupted'} for d in original['arms'].values()):
                raise ValueError(f'Source experiment did not complete: {original["state"]}; original jobs were not stopped by extension')
            if original['state'] == 'completed':
                break
            write_json(experiment / 'status.json', status)
            old_comparison = source / 'comparison.json'
            if old_comparison.is_file():
                comparison = json.loads(old_comparison.read_text())
                comparison.update(state=status['state'], updated_at_utc=now(), arms=status['arms'],
                    source_experiment=str(source))
                comparison['protocol'] = dict(comparison.get('protocol', {}), epochs=plan['target_epochs'],
                    source_target_epochs=plan['source_target_epochs'])
                write_json(experiment / 'comparison.json', comparison)
            time.sleep(plan['poll_seconds'])
        old = json.loads((source / 'manifest.json').read_text())
        print(f'{now()} Source completed; importing final checkpoints and history', flush=True)
        prepare_continuation(experiment, plan, old)
        if stopped:
            write_json(experiment / 'status.json', dict(state='interrupted', target_epochs=plan['target_epochs']))
            return
        print(f'{now()} Resuming from epoch {plan["source_target_epochs"]+1} through {plan["target_epochs"]}', flush=True)
        supervise(experiment, stop_requested=lambda: stopped)
    except BaseException as exc:
        write_json(experiment / 'status.json', dict(state='failed', target_epochs=plan['target_epochs'],
            source_experiment=str(source), error=f'{type(exc).__name__}: {exc}', finished_at_utc=now()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--seed', type=int, help='Select the unique active source run on this host')
    parser.add_argument('--epochs', type=int, default=300, help='TOTAL target, including completed source epochs')
    parser.add_argument('--run-id')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervise', type=Path)
    args = parser.parse_args()
    if args.supervise:
        supervise_extension(args.supervise.resolve())
    else:
        prepare_extension(args)


if __name__ == '__main__':
    main()
