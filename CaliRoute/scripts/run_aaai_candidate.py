#!/usr/bin/env python3
"""Prepare a versioned AAAI candidate run; start GPUs only with --launch.

The recipe owns research parameters. The hardware profile owns memory chunks
and rank allocation. Existing historical launchers and running snapshots remain
independent. E1 is not defined or started by this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
from caliroute.recipes import build_recipe_config
import run_evrptw_dual_scratch as runtime

REFERENCE_PATH = CODE_ROOT / 'docs/experiments/graph_rdi100_20261008_provenance.json'
RUNTIME_ROOTS = {'offline2online', 'caliroute', 'EVRPTW_Benchmark', 'evrptw_core', 'evrptw_hierarchy'}


def verify_training_engine():
    """Keep the candidate's known training semantics pinned to the reviewed run."""
    reference = json.loads(REFERENCE_PATH.read_text())
    verified = {}
    for name, expected in reference['source']['file_sha256'].items():
        path = Path(name)
        if path.parts[0] not in RUNTIME_ROOTS or path.suffix != '.py':
            continue
        actual = hashlib.sha256((CODE_ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Training engine changed from the reviewed reference: {name}; version and validate a new recipe before launching')
        verified[name] = actual
    return dict(reference_training_commit=reference['exact_training_code_commit'],
                matching_runtime_python_files=len(verified), file_sha256=verified)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=('vrptw', 'evrptw'), required=True)
    parser.add_argument('--customers', type=int, default=100)
    parser.add_argument('--encoder', choices=('graph', 'current'), default='graph')
    parser.add_argument('--hardware', choices=('rtx48_single', '2080ti_dual'), required=True)
    parser.add_argument('--gpus', help='Physical indices; defaults to 0 or 0,1 according to hardware profile')
    parser.add_argument('--global-batch', type=int)
    parser.add_argument('--chunk-size', type=int, help='Execution memory override; does not change rollout batch')
    parser.add_argument('--expert-chunk-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--seed', type=int, default=3011)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--data-root', type=Path, default=runtime.default_data_root(), help='AAAI_Dataset directory containing dataset/')
    parser.add_argument('--run-id')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--launch', action='store_true', help='Launch a detached supervisor with an independent full-shape preflight')
    modes.add_argument('--prepare-only', action='store_true', help='Default: freeze source and record configs without starting any training')
    modes.add_argument('--print-config', action='store_true', help='Print resolved YAML only; no output directory or GPU process')
    return parser


def recipe_config(args, output, run_name, world_size):
    return build_recipe_config(problem=args.problem, customers=args.customers,
        encoder=args.encoder, seed=args.seed, epochs=args.epochs,
        data_root=args.data_root.resolve(), output_dir=output, run_name=run_name,
        world_size=world_size, global_batch=args.global_batch,
        ppo_chunk_size=args.chunk_size, expert_chunk_size=args.expert_chunk_size,
        eval_batch_size=args.eval_batch_size, hardware=args.hardware)


def prepare(args):
    gpus = runtime.shared.parse_gpus(args.gpus or ('0' if args.hardware == 'rtx48_single' else '0,1'))
    world_size = len(gpus)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    run_id = args.run_id or f'AAAI_CANDIDATE_V1_{args.problem.upper()}{args.customers}_{args.encoder.upper()}_{args.hardware.upper()}_S{args.seed}_E{args.epochs}_{stamp}'
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run-id must be a fresh directory name')
    output = CODE_ROOT / 'results/optimization' / run_id / 'optimized'
    cfg = recipe_config(args, output, run_id + '_OPTIMIZED', world_size)
    engine = verify_training_engine()
    cfg['experiment_protocol']['training_engine_verification'] = engine
    cfg['experiment_protocol']['require_preflight_health'] = True
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False), end='')
        return None
    # Reuse the tested source-freezing/process/evaluation lifecycle. The recipe
    # supplies the complete config; no historical configuration builder runs.
    legacy_args = runtime.make_parser().parse_args([
        '--task', args.problem, '--variant', 'optimized', '--encoder-variant', args.encoder,
        '--gpus', ','.join(map(str, gpus)), '--seed', str(args.seed), '--epochs', str(args.epochs),
        '--batch-per-gpu', str(cfg['training']['num_envs_per_gpu']),
        '--chunk-size', str(cfg['training']['ppo_step_chunk_size']),
        '--expert-chunk-size', str(cfg['offline']['sl_expert_logprob_chunk_size']),
        '--learning-rate', str(cfg['training']['learning_rate']),
        '--eval-interval', str(cfg['evaluation']['eval_interval']),
        '--data-root', str(args.data_root.resolve()), '--run-id', run_id,
        '--base-config', str(CODE_ROOT / 'configs/recipes/aaai_graph_v1.yaml'),
        '--launch' if args.launch else '--prepare-only',
    ])
    return runtime.prepare(legacy_args, config_builder=lambda base, **kwargs: cfg, arm_label=args.encoder)


def main():
    prepare(make_parser().parse_args())


if __name__ == '__main__':
    main()
