#!/usr/bin/env python3
"""Prepare or launch a frozen VRPTW100 physical-input normalization comparison.

Four single-GPU arms cross depot-centered fixed-scale coordinates with an
optional physical node/global context adapter. Reward, PopArt, actor RMS and
SL-PPO are shared. Python prepares by default; the shell launches in background.
All validation is full-split best-of-50, including each arm's own epoch zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))

import run_reward_norm_comparison as shared
from caliroute.input_normalization import input_normalization_signature

DEFAULT_CHECKPOINT = Path('assets/input_norm/vrptw100_norm_epoch0300.pt')
SOURCE_RUN = 'REWARD_NORM_VRPTW100_S3009_E300_20261007T070618Z'
DISTANCE_UNIT_KM = 43.638668060302734
ARMS = {
    'legacy': ('legacy_minmax', False),
    'depot': ('depot_fixed', False),
    'context': ('legacy_minmax', True),
    'combined': ('depot_fixed', True),
}
INITIAL_PAIRS = [('legacy', 'context'), ('depot', 'combined')]


def build_arm(base, *, arm, output, run_name, init_checkpoint, data_root,
              seed, units, epochs=300, chunk_size=18, eval_interval=50):
    """Change input factors only; inherit the current Norm training protocol."""
    if arm not in ARMS:
        raise ValueError(f'Unknown input normalization arm: {arm}')
    for key in ('reward_distance_scale_km', 'observation_distance_scale_km'):
        if units.get(key) != DISTANCE_UNIT_KM:
            raise ValueError(f'{key} must remain {DISTANCE_UNIT_KM} for this controlled screen')
    cfg = shared.build_arm(base, arm='normalization', output=output, run_name=run_name,
        init_checkpoint=init_checkpoint, data_root=data_root, seed=seed,
        units=units, epochs=epochs, chunk_size=chunk_size, eval_interval=eval_interval)
    coordinates, context = ARMS[arm]
    cfg['env'].update(observation_coordinate_mode=coordinates,
        observation_input_context=context)
    cfg['model'].update(use_physical_input_context=context,
        physical_input_context_hidden_dim=32)
    # The loader allows only new adapter parameters to be missing. It must not
    # silently accept absent old backbone parameters when strict=False is used.
    cfg['offline']['init_checkpoint_strict'] = not context
    signature = input_normalization_signature(cfg)
    cfg['experiment_protocol'].update(phase='physical_input_normalization_2x2',
        arm=arm, input_normalization_signature=signature,
        source_initialization_experiment=SOURCE_RUN,
        input_units='fixed physical distance unit; only coordinate mode and optional input context vary',
        initialization='shared mature Norm epoch-300 raw-unit inference weights; fresh optimizer/replay/actor RMS/PopArt training state',
        initial_evaluation_equivalence_group=coordinates,
        comparison_scope='single-scale checkpoint-migration screen; coordinate modes have separate epoch-zero quality, not from-scratch evidence')
    return cfg


def prepare(args):
    selected = args.arms.split(',')
    # Validate names before constructing protocol descriptors or reading files.
    if not selected or len(set(selected)) != len(selected) or any(a not in ARMS for a in selected):
        raise ValueError('arms must be distinct names from: '+','.join(ARMS))
    protocol = dict(phase='physical_input_normalization_2x2',
        source_initialization_experiment=SOURCE_RUN,
        variants={a: dict(observation_coordinate_mode=ARMS[a][0],
            observation_input_context=ARMS[a][1],
            gamma=.99, reward_norm_mode='physical_shared_popart') for a in selected},
        initial_evaluation_pairs=[list(pair) for pair in INITIAL_PAIRS if all(a in selected for a in pair)],
        input_distance_scale_km=DISTANCE_UNIT_KM,
        initialization='same fixed mature Norm epoch-300 raw-unit weights; fresh optimizer/replay/actor RMS/PopArt; preflight weights discarded',
        initialization_caveat='Only same-coordinate zero-output adapter pairs should match at epoch zero; log coordinate-migration cost separately.',
        input_context_hidden_dim=32,
        changed_factors=['coordinate_mode', 'physical_node_and_global_context'],
        unchanged_components=['physical_reward', 'gamma', 'PopArt', 'actor_RMS', 'SLPPO', 'RDI', 'AGDA', 'hidden_normalization'],
        stage='300-epoch input-representation migration screen unless explicitly overridden; no test selection or from-scratch superiority claim')
    return shared.prepare(args,
        arm_definitions={a: (.99, 'physical_shared_popart') for a in ARMS},
        arm_builder=build_arm, default_checkpoint=DEFAULT_CHECKPOINT,
        protocol_overrides=protocol, prerequisite_source_run=None)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path,
        default=CODE_ROOT / 'configs/experiments/input_norm_vrptw100.yaml')
    parser.add_argument('--init-checkpoint', type=Path, default=CODE_ROOT / DEFAULT_CHECKPOINT,
        help='Shared mature Norm epoch-300 compact weights distributed with this branch')
    parser.add_argument('--expected-init-epoch', type=int, default=300)
    parser.add_argument('--data-root', type=Path, default=CODE_ROOT.parent / 'AAAI_Dataset')
    parser.add_argument('--run-id')
    parser.add_argument('--seed', type=int, default=3009)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval-interval', type=int, default=50,
        help='Full validation every N epochs, plus separate epoch 0 and final epoch for each arm')
    parser.add_argument('--chunk-size', type=int, default=18)
    parser.add_argument('--arms', default=','.join(ARMS),
        help='Selected factorial arms (default: legacy,depot,context,combined)')
    parser.add_argument('--gpus', default='0,1,2,3',
        help='One to four identical-model GPUs; fewer GPUs queue the independent arms')
    parser.add_argument('--wait-for-experiment', action='append', type=Path, default=[],
        help='Previous status.json whose assigned GPU jobs must finish')
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--idle-checks', type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervise', type=Path)
    return parser


def main():
    args = make_parser().parse_args()
    if args.supervise:
        shared.supervise(args.supervise.resolve())
    else:
        if args.run_id is None:
            args.run_id = f'INPUT_NORM_VRPTW100_S{args.seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        prepare(args)


if __name__ == '__main__':
    main()
