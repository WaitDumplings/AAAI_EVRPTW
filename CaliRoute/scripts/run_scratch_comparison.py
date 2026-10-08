#!/usr/bin/env python3
"""From-scratch VRPTW100 comparison against the exact first repository commit.

No initialization checkpoint is read, copied or used. The original training
implementation runs in its own frozen checkout, with an external measurement
adapter for fixed evaluation seeds, epoch zero and independent route checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
import run_reward_norm_comparison as shared
import run_physics_exploration_comparison as improved
from original_scratch_config import build_original_config

ORIGINAL_COMMIT = 'f388343dbb1d54bbd3f76dd29ca95208070d31a8'
ARMS = ('legacy', 'physics', 'archive', 'explore')


def assert_scratch(cfg):
    for section in ('training', 'offline'):
        for key, value in cfg.get(section, {}).items():
            if value and ('checkpoint' in key and any(x in key for x in ('init', 'initial', 'resume', 'reference', 'pretrained')) and not key.endswith('_strict')):
                raise ValueError(f'Scratch training forbids {section}.{key}={value}')


def build_arm(base, *, arm, output, run_name, data_root, seed, epochs=300,
              chunk_size=15, legacy_chunk_size=8, eval_interval=50, learning_rate=1e-4):
    if arm not in ARMS:
        raise ValueError('Unknown scratch arm: ' + str(arm))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if arm == 'legacy':
        cfg = build_original_config(output=output, run_name=run_name, data_root=data_root,
            seed=seed, epochs=epochs, chunk_size=legacy_chunk_size,
            eval_interval=eval_interval, learning_rate=learning_rate)
    else:
        cfg = improved.build_arm(base, arm=arm, output=output, run_name=run_name,
            init_checkpoint=None, data_root=data_root, seed=seed,
            units=dict(reward_distance_scale_km=improved.DISTANCE_UNIT_KM,
                       observation_distance_scale_km=improved.DISTANCE_UNIT_KM),
            epochs=epochs, chunk_size=chunk_size, eval_interval=eval_interval,
            learning_rate=learning_rate, target_kl=None)
        # The shared configuration constructor is reused, never its checkpoint
        # preparation. Remove every warm-start/resume/reference loading path.
        for section in ('training', 'offline'):
            for key in list(cfg[section]):
                if 'checkpoint' in key and any(x in key for x in ('init', 'initial', 'resume', 'reference', 'pretrained')):
                    del cfg[section][key]
        cfg['training'].update(ppo_update_epochs=5, learning_rate=learning_rate,
            lr_min=learning_rate, ent_coef=.01, entropy_initial_coef=.01, entropy_final_coef=.01)
        cfg['offline'].update(sl_coef=.5, use_priority_sampler=False)
        cfg['experiment_protocol'].update(attempted_optimizer_steps_per_epoch=20,
            ppo_update_epochs=5, source_initialization_experiment=None,
            initial_evaluation_equivalence_group='physics_archive_explore_random_initialization')
    cfg['training']['ppo_update_epochs'] = 5
    if arm == 'legacy':
        cfg['experiment_protocol']['evaluation_adapter'] = dict(
            epoch_zero=True, fixed_isolated_rng=True, export_selected_routes=True,
            independent_selected_route_validation=True, native_caveats='evaluation_caveats below describe the original runtime before this external adapter',
            checkpoint_selection='original rule; epoch zero not eligible',
            trajectory_feasibility='native environment metric; selected-route feasibility uses independent checks')
        cfg['experiment_protocol']['protocol_overrides'].append(dict(
            parameter='training.ppo_update_epochs', original=4, used=5,
            reason='User explicitly requested five PPO passes for every scratch arm'))
    cfg.setdefault('experiment_protocol', {}).update(ppo_update_epochs=5, attempted_optimizer_steps_per_epoch=20)
    cfg['evaluation'].update(eval_before_training=True, eval_seed=17000000+seed,
                             eval_output_dir=str(output / 'evaluations'), eval_save_routes=True)
    cfg.setdefault('experiment_protocol', {}).update(phase='original_vs_physics_from_scratch',
        arm=arm, initialization_mode='scratch', initialization='random model initialization from the run seed; empty optimizer, replay and running statistics; no PPO init or learned checkpoint',
        source_init_epoch=None, source_init_checkpoint=None,
        comparison_scope='original implementation versus complete improvements under a declared common training budget; not a single-factor architecture ablation',
        batch_controls=dict(instances=64, trajectories=50, minibatches=4, ppo_passes=5, rollout_steps=201),
        ppo_chunk_size=cfg['training']['ppo_step_chunk_size'],
        chunk_scope='memory-only time chunking; same instances, trajectories and optimizer passes; legacy retains its original loss reduction')
    assert_scratch(cfg)
    return cfg


def build_preflight(cfg, output):
    result = copy.deepcopy(cfg)
    result['run_name'] += '_PREFLIGHT'
    result['training'].update(epochs=2, monitor_interval=1, post_update_kl_interval=1,
                             monitor_output_dir=str(output / 'monitoring'))
    result['evaluation'].update(eval_before_training=False, eval_interval=2,
        eval_n_traj=4, eval_batch_size=4, eval_limit=4, eval_output_dir=str(output / 'evaluations'))
    if result['offline'].get('branch_exploration_enabled'):
        result['offline']['exploration_interval'] = 1
    result['experiment_protocol'].update(phase='scratch_gpu_preflight', epochs=2,
        validation_instances=4, comparison_scope='same formal training allocation; two epochs with reduced validation; all resulting state discarded')
    assert_scratch(result)
    return result


def original_snapshot(destination):
    """Extract exact historical blobs, without touching the user's checkout."""
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=CODE_ROOT, text=True).strip())
    revision = subprocess.check_output(['git', 'rev-parse', ORIGINAL_COMMIT + '^{commit}'], cwd=repo, text=True).strip()
    if revision != ORIGINAL_COMMIT:
        raise ValueError('Original baseline must resolve to the pinned full commit')
    archive = subprocess.check_output(['git', 'archive', '--format=tar', ORIGINAL_COMMIT, 'CaliRoute'], cwd=repo)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as data:
        for member in data.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'CaliRoute':
                raise ValueError('Unsafe historical archive path')
            relative = Path(*path.parts[1:])
            if member.isdir():
                continue
            if not member.isfile() or not relative.parts or relative.parts[0] == 'results':
                raise ValueError('Unsupported historical archive entry: ' + member.name)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = data.extractfile(member).read()
            target.write_bytes(blob)
            target.chmod(member.mode & 0o777)
            files[str(relative)] = hashlib.sha256(blob).hexdigest()
    (destination / 'results').symlink_to(CODE_ROOT / 'results', target_is_directory=True)
    return dict(git_commit=revision, files=files,
        content_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        source_modifications='none; external adapter only controls measurement')


def prepare(args):
    arms = args.arms.split(',')
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('arms must be distinct names from: ' + ','.join(ARMS))
    for value, label in ((args.epochs, 'epochs'), (args.eval_interval, 'eval_interval'),
                         (args.chunk_size, 'chunk_size'), (args.legacy_chunk_size, 'legacy_chunk_size')):
        shared.positive_integer(value, label)
    if max(args.chunk_size, args.legacy_chunk_size) > 201:
        raise ValueError('Time chunks cannot exceed the rollout horizon 201')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not 1 <= args.poll_seconds <= 60 or not 2 <= args.idle_checks <= 10:
        raise ValueError('poll_seconds must be 1..60 and idle_checks 2..10')
    gpus = shared.parse_gpus(args.gpus)
    hardware = shared.probe_requested_gpus(gpus)
    run_id = args.run_id or (f'SCRATCH_ORIGINAL_VRPTW100_S{args.seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be a fresh directory name')
    data_root = args.data_root.resolve()
    inputs = {}
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')), ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        for name in names:
            path = data_root / 'dataset/vrptw' / split / 'Cus100' / name
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; put AAAI_Dataset beside CaliRoute or pass --data-root')
            inputs[f'vrptw/{split}/Cus100/{name}'] = dict(path=str(path), sha256=shared.digest(path))
        path = data_root / 'dataset/vrptw' / split / 'Cus100/metadata.json'
        if path.is_file():
            inputs[f'vrptw/{split}/Cus100/metadata.json'] = dict(path=str(path), sha256=shared.digest(path))
    base = yaml.safe_load(args.base_config.resolve().read_text())
    experiment = CODE_ROOT / 'results/optimization' / run_id
    experiment.mkdir(parents=True, exist_ok=False)
    frozen = experiment / 'source/CaliRoute'
    source = shared.source_snapshot(frozen, include_initialization_assets=False)
    extra = {}
    original = experiment / 'original_source/CaliRoute'
    if 'legacy' in arms:
        extra['legacy'] = dict(code_root=str(original), source=original_snapshot(original))
    specs = {}
    for arm in arms:
        output = experiment / arm
        output.mkdir()
        name = run_id + '_' + arm.upper()
        cfg = build_arm(base, arm=arm, output=output, run_name=name,
            data_root=data_root, seed=args.seed, epochs=args.epochs, chunk_size=args.chunk_size,
            legacy_chunk_size=args.legacy_chunk_size, eval_interval=args.eval_interval, learning_rate=args.learning_rate)
        stages = [(False, cfg, output), (True, build_preflight(cfg, output/'preflight'), output/'preflight')]
        spec = {}
        for preflight, configuration, destination in stages:
            destination.mkdir(exist_ok=True)
            config = destination / 'config.yaml'
            config.write_text(yaml.safe_dump(configuration, sort_keys=False))
            name_stage = configuration['run_name']
            if arm == 'legacy':
                command = [sys.executable, '-B', '-u', str(frozen/'scripts/run_original_scratch.py'), '--source-root', str(original)]
                (destination/'original_provenance.json').write_text(json.dumps(configuration['experiment_protocol'], indent=2)+'\n')
            else:
                command = [sys.executable, '-B', '-u', '-m', 'offline2online.train']
            command += ['--config', str(config), '--seed', str(args.seed), '--device', 'cuda:0']
            value = dict(config=str(config), config_sha256=shared.digest(config), output_dir=str(destination),
                code_root=str(original if arm == 'legacy' else frozen),
                epochs=2 if preflight else args.epochs, validation_instances=4 if preflight else 1000,
                required_validation_epochs=[2] if preflight else shared.validation_epochs(args.epochs, args.eval_interval),
                log_dir=str(CODE_ROOT/'results/logs/Cus_100_CS_0'/name_stage/f'seed_{args.seed}'),
                checkpoint_dir=str(CODE_ROOT/'results/checkpoints/Cus_100_CS_0'/name_stage/f'seed_{args.seed}'), command=command)
            if preflight:
                spec['preflight'] = value
            else:
                spec.update(value)
        specs[arm] = spec
    manifest = dict(created_at_utc=shared.now(), initialization_mode='scratch',
        init_checkpoint=None, init_checkpoint_sha256=None, source_init_checkpoint=None, source_init_epoch=None,
        code_root=str(frozen), source=source, additional_sources=extra, inputs=inputs, arms=specs, gpus=gpus,
        hardware_at_prepare=list(hardware.values()) if hardware is not None else None,
        wait_for_experiments=[str(p.resolve()) for p in args.wait_for_experiment],
        idle_checks=args.idle_checks, poll_seconds=args.poll_seconds,
        protocol=dict(phase='original_vs_physics_from_scratch', task='vrptw100', epochs=args.epochs, seed=args.seed,
            initialization='all models random from seed; no learned checkpoint or PPO init; preflight state discarded',
            original_commit=ORIGINAL_COMMIT, legacy_implementation='exact frozen original source, external evaluation adapter',
            world_size_per_arm=1, global_batch=64, n_traj=50, num_minibatches=4, ppo_update_epochs=5,
            learning_rate=args.learning_rate, lr_schedule='constant', entropy_coef=.01, sl_coef=.5,
            ppo_step_chunk_size=args.chunk_size, legacy_ppo_step_chunk_size=args.legacy_chunk_size,
            validation_instances=1000, eval_interval=args.eval_interval, eval_n_traj=50, eval_batch_size=32,
            eval_seed=17000000+args.seed, test_enabled=False,
            initial_evaluation_pairs=[[a,b] for a,b in [('physics','archive'), ('physics','explore')] if a in arms and b in arms],
            initial_evaluation_consistency_scope='same_architecture_random_initialization_pairs_only',
            initialization_caveat='Legacy uses the original architecture; identical seeds do not imply identical models across different architectures. Physics/archive/explore share an architecture and should match at epoch zero.',
            objective='mean raw feasible route km; always report feasibility and coverage',
            best_checkpoint_caveat='Original native best minimizes feasible-subset distance once feasibility is positive; modern native best maximizes feasibility first. Compare matching full validation epochs; if feasibility differs, select stored checkpoints by one common feasibility-first rule before test, not by native best filenames.',
            sampling='uniform shuffle_cycle in every arm; explicit common-budget override of original priority sampling',
            evaluation='same validation split, fixed evaluation RNG, best-of-50, epoch zero and independent VRPTW route checks; original training algorithm unchanged',
            original_default_overrides=dict(epochs=args.epochs, instances_per_rollout=64, rollout_steps=201,
                ppo_update_epochs=dict(original=4, used=5),
                sampling='uniform instead of priority', evaluation_interval=args.eval_interval, evaluation_batch_size=32),
            variants={'legacy':'f388343 original input/reward/model/SLPPO', 'physics':'complete physical/model/PPO improvements',
                      'archive':'physics plus structural elite/exploration archives', 'explore':'archive plus independent branch search'},
            extra_search_budget=dict(arm='explore', interval=5, max_instances=8, trajectories_per_instance=8),
            compute_caveat='Matched on-policy instances and optimizer passes, not matched wall time; explore adds search. Legacy time chunking may differ for memory without changing optimizer budget.',
            comparison_scope='original versus full improvement bundles; multi-seed scratch screen, not paper-protocol replication or single-factor proof',
            gpu_preflight=dict(epochs=2, instances_per_rollout=64, n_traj=50, ppo_update_epochs=5,
                num_minibatches=4, chunk_size=args.chunk_size, legacy_chunk_size=args.legacy_chunk_size,
                validation_instances=4, validation_n_traj=4)))
    shared.write_json(experiment/'manifest.json', manifest)
    shared.write_json(experiment/'status.json', dict(state='prepared', arms={a:dict(state='prepared') for a in arms}))
    shared.verify_manifest(manifest)
    if args.launch:
        with (experiment/'supervisor.log').open('a', buffering=1) as log:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(frozen/'scripts/run_reward_norm_comparison.py'), '--supervise', str(experiment)],
                cwd=frozen, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (experiment/'supervisor.pid').write_text(str(process.pid)+'\n')
    print(experiment, flush=True)
    return experiment


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', type=Path, default=CODE_ROOT/'configs/experiments/physics_exploration_vrptw100.yaml')
    p.add_argument('--data-root', type=Path, default=CODE_ROOT.parent/'AAAI_Dataset')
    p.add_argument('--seed', type=int, default=3010)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--eval-interval', type=int, default=50)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--chunk-size', type=int, default=15)
    p.add_argument('--legacy-chunk-size', type=int, default=8)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--run-id')
    p.add_argument('--wait-for-experiment', type=Path, action='append', default=[])
    p.add_argument('--poll-seconds', type=int, default=10)
    p.add_argument('--idle-checks', type=int, default=3)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--launch', action='store_true')
    return p


if __name__ == '__main__':
    prepare(make_parser().parse_args())
