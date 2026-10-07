#!/usr/bin/env python3
"""Frozen four-arm VRPTW100 embedding/encoder/decoder comparison.

All arms use the combined depot/fixed-unit/physical-context input. Static and
resource-decoder integrations form a 2x2 screen; reward and SL-PPO stay fixed.
Python prepares by default. The shell detaches a supervisor with per-arm GPU
preflights and queues arms if fewer than four idle, matching GPUs are supplied.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))

import run_input_norm_comparison as inputs
import run_reward_norm_comparison as shared

DEFAULT_CHECKPOINT = inputs.DEFAULT_CHECKPOINT
SOURCE_RUN = inputs.SOURCE_RUN
DISTANCE_UNIT_KM = inputs.DISTANCE_UNIT_KM
ARMS = {
    'baseline': (False, False),
    'static': (True, False),
    'dynamic': (False, True),
    'combined': (True, True),
}
INITIAL_PAIRS = [('baseline', arm) for arm in ARMS if arm != 'baseline']


def model_options(arm, *, edge_messages=False, edge_updates=False):
    if arm not in ARMS:
        raise ValueError(f'Unknown model integration arm: {arm}')
    static, dynamic = ARMS[arm]
    return dict(use_typed_static_fusion=static, use_edge_relation_encoder=static,
        edge_relation_dim=16, use_resource_decoder=dynamic,
        decoder_observation_mode='dual' if dynamic else 'feasible',
        use_edge_value_messages=static and bool(edge_messages),
        use_edge_state_updates=static and bool(edge_updates))


def build_arm(base, *, arm, output, run_name, init_checkpoint, data_root,
              seed, units, epochs=300, chunk_size=12, eval_interval=50,
              edge_messages=False, edge_updates=False):
    options = model_options(arm, edge_messages=edge_messages, edge_updates=edge_updates)
    cfg = inputs.build_arm(base, arm='combined', output=output, run_name=run_name,
        init_checkpoint=init_checkpoint, data_root=data_root, seed=seed,
        units=units, epochs=epochs, chunk_size=chunk_size, eval_interval=eval_interval)
    cfg['model'].update(options)
    # The bundled source has no physical-input adapter or stage-2 modules. The
    # trainer permits only explicitly whitelisted added parameters to be absent.
    cfg['offline']['init_checkpoint_strict'] = False
    cfg['experiment_protocol'].update(phase='physical_model_integration_2x2', arm=arm,
        model_integration=options,
        input_units='same depot-fixed coordinates and node/global physical context in every arm',
        initial_evaluation_equivalence_group='all_arms',
        initialization='shared mature Norm epoch-300 raw-unit inference weights; added output heads initialize to zero; fresh optimizer/replay/actor RMS/PopArt',
        comparison_scope='combined input is fixed by design, not a selected winner; checkpoint-migration screen, not from-scratch evidence')
    return cfg


def build_preflight(cfg, output):
    """Exercise the formal training allocation before starting the 300-epoch run.

    Only duration, monitoring location/cadence and validation size are reduced.
    In particular, the instance batch, trajectory count and PPO chunk retain the
    formal values so an undersized toy rollout cannot hide a training OOM.
    """
    result = shared.build_preflight(cfg, output)
    for key in ('num_envs_per_gpu', 'n_traj', 'ppo_step_chunk_size'):
        result['training'][key] = cfg['training'][key]
    for key in ('global_instances_per_rollout', 'global_trajectories_per_rollout',
                'global_instances_per_optimizer_step'):
        result['experiment_protocol'][key] = cfg['experiment_protocol'][key]
    result['experiment_protocol'].update(
        comparison_scope='Formal training batch/trajectories/chunk memory and pipeline check; two epochs only, reduced validation; excluded from formal comparison')
    return result


def prepare(args):
    selected = args.arms.split(',')
    if not selected or len(set(selected)) != len(selected) or any(a not in ARMS for a in selected):
        raise ValueError('arms must be distinct names from: '+','.join(ARMS))
    if (args.edge_messages or args.edge_updates) and not any(ARMS[a][0] for a in selected):
        raise ValueError('edge options require a static-enabled arm: static or combined')
    variants = {a: dict(model_options(a, edge_messages=args.edge_messages,
        edge_updates=args.edge_updates), gamma=.99,
        reward_norm_mode='physical_shared_popart') for a in selected}
    protocol = dict(phase='physical_model_integration_2x2',
        source_initialization_experiment=SOURCE_RUN,
        gpu_preflight=dict(epochs=2, instances_per_rollout=64, n_traj=50,
            chunk_size=args.chunk_size, num_minibatches=4, ppo_update_epochs=5,
            validation_instances=4, validation_n_traj=4,
            scope='full formal training shape; reduced duration and validation only'),
        variants=variants,
        initial_evaluation_pairs=[list(pair) for pair in INITIAL_PAIRS if all(a in selected for a in pair)],
        initial_evaluation_consistency_scope='all_four_arms_zero_output_new_modules',
        input_distance_scale_km=DISTANCE_UNIT_KM,
        input_coordinate_mode='depot_fixed', input_physical_context=True,
        initialization='same bundled mature Norm epoch-300 raw-unit weights; fresh optimizer/replay/actor RMS/PopArt; preflight weights discarded',
        initialization_caveat='Source is the prior Norm epoch-300 model, not a trained combined-input winner. The coordinate migration is common to all arms; new output heads start at zero, so all arms should match at epoch zero.',
        input_context_hidden_dim=32, edge_relation_dim=16,
        optional_edge_messages=bool(args.edge_messages),
        optional_edge_updates=bool(args.edge_updates),
        changed_factors=['typed_static_fusion_and_directed_edge_encoder', 'resource_conditioned_decoder_and_dual_observation'],
        unchanged_components=['input_normalization', 'physical_reward', 'gamma', 'PopArt', 'actor_RMS', 'SLPPO', 'legacy_RDI_AGDA_parameters', 'hidden_normalization', 'action_feasibility_mask'],
        stage='300-epoch model-integration migration screen unless explicitly overridden; no test selection or from-scratch superiority claim')
    return shared.prepare(args,
        arm_definitions={a: (.99, 'physical_shared_popart') for a in ARMS},
        arm_builder=partial(build_arm, edge_messages=args.edge_messages, edge_updates=args.edge_updates),
        default_checkpoint=DEFAULT_CHECKPOINT, protocol_overrides=protocol,
        preflight_builder=build_preflight,
        prerequisite_source_run=None)


def make_parser():
    parser = inputs.make_parser()
    parser.description = __doc__
    parser.set_defaults(base_config=CODE_ROOT / 'configs/experiments/model_integration_vrptw100.yaml',
        seed=3010, chunk_size=12, arms=','.join(ARMS))
    # Reusing argument validation and portability does not reuse the first-stage
    # arm semantics. Keep --help explicit about the different experiment.
    for action in parser._actions:
        if action.dest == 'arms':
            action.help = 'Selected model arms (default: baseline,static,dynamic,combined)'
        elif action.dest == 'init_checkpoint':
            action.help = 'Bundled prior Norm epoch-300 weights; no dependency on another server\'s results'
    parser.add_argument('--edge-messages', action='store_true',
        help='Optional heavier edge-value messages in static/combined only; default off')
    parser.add_argument('--edge-updates', action='store_true',
        help='Optional edge-state updates in static/combined only; default off')
    return parser


def main():
    args = make_parser().parse_args()
    if args.supervise:
        shared.supervise(args.supervise.resolve())
    else:
        if args.run_id is None:
            args.run_id = f'MODEL_INTEGRATION_VRPTW100_S{args.seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        prepare(args)


if __name__ == '__main__':
    main()
