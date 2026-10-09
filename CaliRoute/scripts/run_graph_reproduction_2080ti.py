#!/usr/bin/env python3
"""Replay the recorded Graph/current experiment on two 11GB GPUs per arm.

Only execution memory limits and per-rank budgets differ from the recorded
single-GPU experiment. No training implementation is patched by this launcher.
"""
from __future__ import annotations
import hashlib
import math
import json
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
import run_evrptw_dual_scratch as shared

REFERENCE = CODE_ROOT / 'docs/experiments/graph_rdi100_20261008_provenance.json'
TRAINING_COMMIT = 'd4364926999d8b55be6ec3aa374c6cb6fc238cea'
# These are execution/recording changes, never changes to model or loss code.
SOURCE_EXCEPTIONS = {'scripts/run_evrptw_dual_scratch.py'}
PATH_FIELDS = {
    'run_name', 'data.train_dataset_path', 'evaluation.eval_output_dir',
    'evaluation.eval_path', 'evaluation.gurobi_summary_path',
    'offline.expert_dataset_path', 'offline.expert_solution_path',
    'training.monitor_output_dir',
}
MEMORY_FIELDS = {
    'training.ppo_step_chunk_size', 'offline.sl_expert_logprob_chunk_size',
    'advantage.sl_expert_logprob_chunk_size',
}


def reference_run(task, encoder_variant):
    batch = 40 if task == 'vrptw' else 32
    return f'{encoder_variant.upper()}_{task.upper()}100_S3011_B{batch}_E1500_20261008_r2'


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def verify_reference_config(config, *, task, encoder_variant, world_size):
    """Fail closed on an accidental algorithm/default change; return all diffs."""
    if world_size != 2:
        raise ValueError('The 2080Ti reproduction requires two GPUs per arm')
    reference = json.loads(REFERENCE.read_text())['runs'][reference_run(task, encoder_variant)]['effective_config']
    old, new = flatten(reference), flatten(config)
    expected = {
        'training.num_envs_per_gpu': (40 if task == 'vrptw' else 32) // world_size,
        'offline.exploration_instances': reference['offline']['exploration_instances'] // world_size,
        'offline.policy_replay_max_new_routes': reference['offline']['policy_replay_max_new_routes'] // world_size,
    }
    for key, value in expected.items():
        if new.get(key) != value:
            raise ValueError(f'{key} must be {value} to preserve the global reference budget')
    diffs = []
    for key in sorted(old.keys() | new.keys()):
        if key.startswith('experiment_protocol.') or old.get(key) == new.get(key):
            continue
        if key not in PATH_FIELDS | MEMORY_FIELDS | set(expected) | {'training.epochs'}:
            raise ValueError(f'Unexpected reference change {key}: {old.get(key)!r} -> {new.get(key)!r}')
        diffs.append(dict(parameter=key, reference=old.get(key), used=new.get(key)))
    return diffs


def verify_reference_source():
    """Check every recorded source file except the declared launcher hook."""
    provenance = json.loads(REFERENCE.read_text())
    hashes = provenance['source']['file_sha256']
    checked = {}
    overrides = {}
    for relative, expected in hashes.items():
        path = CODE_ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative in SOURCE_EXCEPTIONS:
            overrides[relative] = dict(reference=expected, used=actual)
        elif actual != expected:
            raise ValueError(f'Source differs from the successful remote run: {relative}')
        else:
            checked[relative] = actual
    return dict(reference_training_commit=TRAINING_COMMIT,
                matching_recorded_files=len(checked), launcher_overrides=overrides,
                matched_file_sha256=checked)


def build_config(base, **kwargs):
    if kwargs.get('variant') != 'optimized':
        raise ValueError('Both arms use the remote optimized trainer, with graph/current encoders')
    task = kwargs.get('task', 'vrptw')
    world_size = kwargs.get('world_size', 2)
    cfg = shared.build_config(base, **kwargs)
    # This is an ingest cap per rank, not capacity per instance. Keep global32.
    cfg['offline']['policy_replay_max_new_routes'] = 32 // world_size
    differences = verify_reference_config(cfg, task=task,
        encoder_variant=kwargs.get('encoder_variant', 'graph'), world_size=world_size)
    cfg['experiment_protocol'].update(
        phase=f'{task}100_graph_reproduction_2080ti',
        arm=kwargs.get('encoder_variant', 'graph'), require_preflight_health=True,
        reference_run=reference_run(task, kwargs.get('encoder_variant', 'graph')),
        reference_training_commit=TRAINING_COMMIT,
        reference_config_differences=differences,
        global_policy_replay_max_new_routes=32,
        ppo_warmup_epochs=0,
        comparison_scope='Same remote training/model source; graph versus current static encoder and matching decoder edge interface. Same local distributed rules, global batch, reward, SL, exploration and validation budgets.',
        remote_reproduction_caveat='Dual GPU uses independent rank samplers and archives and averages rank-local masked losses; it is not a bitwise or global-valid-action-loss equivalent of the remote single-GPU run. Hardware, software and chunk arithmetic also differ.',
    )
    integration = cfg['experiment_protocol'].get('model_integration', {})
    for key in list(integration):
        if key in cfg['model']:
            integration[key] = cfg['model'][key]
    integration.update(use_joint_graph_encoder=cfg['model']['use_joint_graph_encoder'],
        joint_graph_edge_dim=cfg['model'].get('joint_graph_edge_dim'),
        active_edge_state_dim=(cfg['model']['joint_graph_edge_dim'] if cfg['model']['use_joint_graph_encoder']
                               else cfg['model']['edge_relation_dim']))
    cfg['experiment_protocol']['model_integration'] = integration
    return cfg


def validate_preflight(spec, world_size):
    """Do not start a long run after an incomplete or numerically bad smoke run."""
    output = Path(spec['output_dir']) / 'monitoring'
    report = {}
    for rank in range(world_size):
        rows = [json.loads(line) for line in (output / f'monitor_rank_{rank}.jsonl').read_text().splitlines() if line.strip()]
        if [row['epoch'] for row in rows] != [1, 2]:
            raise ValueError(f'Preflight rank {rank}: require exactly epochs 1 and 2')
        for row in rows:
            expected = {'optimizer_steps_epoch': 20, 'amp_skipped_steps_epoch': 0,
                        'grad_norm_nonfinite_count': 0}
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ValueError(f"Preflight rank {rank} epoch {row['epoch']}: {key}={row.get(key)}, expected {value}")
            sync = row.get('parameter_sync', {})
            for key in ('parameter_sync_checksum_max_diff', 'optimizer_steps_rank_max_diff',
                        'amp_skipped_steps_rank_max_diff', 'amp_scale_rank_max_diff'):
                if sync.get(key) != 0:
                    raise ValueError(f'Preflight rank {rank}: missing or failed synchronization: {key}={sync.get(key)}')
            if not math.isfinite(row.get('grad_norm', float('nan'))):
                raise ValueError(f'Preflight rank {rank}: nonfinite gradient norm')
        report[str(rank)] = dict(epochs=2, successful_updates=sum(r['optimizer_steps_epoch'] for r in rows),
            amp_skips=0, nonfinite_gradient_steps=0, parameter_checksum_max_diff=0,
            peak_allocated_bytes=max(r.get('gpu_peak_allocated_bytes', 0) for r in rows),
            peak_reserved_bytes=max(r.get('gpu_peak_reserved_bytes', 0) for r in rows))
    return report


def main():
    parser = shared.make_parser()
    parser.description = __doc__
    parser.set_defaults(task='vrptw', variant='optimized', encoder_variant='graph',
                        seed=3011, batch_per_gpu=None, chunk_size=48, expert_chunk_size=64)
    args = parser.parse_args()
    if args.supervise:
        shared.supervise(args.supervise.resolve())
        return
    if args.single_gpu:
        parser.error('Use two GPUs per arm for this reproduction')
    if args.batch_per_gpu is None:
        args.batch_per_gpu = 20 if args.task == 'vrptw' else 16
    source = verify_reference_source()
    inputs, _ = shared.dataset_inputs(args.data_root.resolve(), args.task)
    reference = json.loads(REFERENCE.read_text())['runs'][reference_run(args.task, args.encoder_variant)]
    for name, value in inputs.items():
        if value['sha256'] != reference['inputs'][name]['sha256']:
            raise ValueError(f'Dataset differs from the recorded remote run: {name}')
    if not args.run_id:
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        args.run_id = (f'{args.task.upper()}100_REMOTE_{args.encoder_variant.upper()}_2080TI_DUAL'
                       f'_B{args.batch_per_gpu * 2}_C{args.chunk_size}_U5_S{args.seed}_E{args.epochs}_{stamp}')
    def builder(base, **kwargs):
        cfg = build_config(base, **kwargs)
        cfg['experiment_protocol']['remote_source_verification'] = source
        return cfg
    shared.prepare(args, config_builder=builder, arm_label=args.encoder_variant)


if __name__ == '__main__':
    main()
