#!/usr/bin/env python3
"""VRPTW100 physical-contract, archive-diversity and branch-exploration screen.

Four independent single-GPU arms share combined inputs, full stage-two model,
initial weights and nominal PPO rollout budget. The exploration arm adds search
compute; compare wall time and completed search trajectories as well as epochs.
Python prepares by default; the shell launches a detached supervisor by default.
"""
from __future__ import annotations

from functools import partial
import math
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))

import run_input_norm_comparison as inputs
import run_model_integration_comparison as integration
import run_reward_norm_comparison as shared
from offline2online.input_normalization import signature as input_signature
from offline2online.model_integration import signature as model_signature

DEFAULT_CHECKPOINT = inputs.DEFAULT_CHECKPOINT
DISTANCE_UNIT_KM = inputs.DISTANCE_UNIT_KM
ARMS = ('legacy', 'physics', 'archive', 'explore')
INITIAL_PAIRS = [('physics', 'archive'), ('physics', 'explore')]


def _check_optimization(learning_rate, target_kl):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0):
        raise ValueError('target_kl must be positive or omitted to keep five fixed passes')


def build_arm(base, *, arm, output, run_name, init_checkpoint, data_root,
              seed, units, epochs=300, chunk_size=15, eval_interval=50,
              learning_rate=1e-5, target_kl=None):
    if arm not in ARMS:
        raise ValueError('Unknown physics/exploration arm: '+str(arm))
    _check_optimization(learning_rate, target_kl)
    cfg = integration.build_arm(base, arm='combined', output=output,
        run_name=run_name, init_checkpoint=init_checkpoint, data_root=data_root,
        seed=seed, units=units, epochs=epochs, chunk_size=chunk_size,
        eval_interval=eval_interval, edge_messages=False, edge_updates=False)
    physics = arm != 'legacy'
    archive = arm in ('archive', 'explore')
    explore = arm == 'explore'
    cfg['data']['strict_road_metric'] = physics
    cfg['env'].update(prefer_explicit_edge_matrices=physics,
        reward_contract='strict_distance' if physics else 'legacy',
        failure_penalty_km=1000., reward_mode='distance')
    cfg['model'].update(agda_physical_candidate_features=physics,
                        agda_smooth_distance_features=physics)
    cfg['training'].update(gamma=1. if physics else .99,
        ppo_loss_reduction='valid_actions' if physics else 'legacy_step_mean',
        bootstrap_truncation=physics,
        post_update_kl_interval=10, target_kl=target_kl,
        learning_rate=float(learning_rate), lr_min=float(learning_rate))
    cfg['offline'].update(
        policy_replay_selection='structural' if archive else 'legacy',
        policy_replay_partition_weight=.5,
        policy_replay_min_structure_distance=.1,
        policy_replay_exploration_capacity=4 if archive else 0,
        policy_replay_exploration_max_relative_gap=.25,
        policy_replay_exploration_stagnation_epochs=10,
        branch_exploration_enabled=explore, exploration_enabled=False,
        exploration_interval=5, exploration_instances=8,
        exploration_trajectories=8, exploration_temperature=1.2,
        exploration_prefix_fractions=[0., .1, .25, .5],
        exploration_max_prefix_steps=32, exploration_exclude_anchor_action=True,
        exploration_prefer_stagnant_archive=True)
    # These experiments isolate the declared bundles, regardless of edits to
    # a supplied base YAML; no shaping bonus/penalty is inherited accidentally.
    cfg['pbrs'].update(use_customer_pbrs=False, use_repair_distance_pbrs=False,
                       use_feasible_ratio_pbrs=False, use_terminal_heuristic=False)
    cfg['experiment_protocol'].update(
        phase='physics_exploration_incremental', arm=arm,
        input_normalization_signature=input_signature(cfg),
        model_integration=model_signature(cfg),
        initial_evaluation_equivalence_group='physics_archive_explore' if physics else 'legacy',
        initialization='same bundled Norm epoch-300 raw-unit weights; fresh optimizer/replay/actor RMS/PopArt; preflight discarded',
        comparison_scope='incremental bundles on a fixed scale; not an individual-factor ablation or a from-scratch comparison',
        extra_search_enabled=explore,
        search_budget=dict(interval=5, max_instances=8, trajectories_per_instance=8,
            max_trajectories_per_event=64, temperature=1.2) if explore else None,
        compute_caveat='explore has extra independent rollouts; equal nominal PPO epochs are not equal compute',
        reward_failure_guard='1000 km on failed termination is an explicit guard, not a proven universal feasibility bound',
        kl_caveat='fresh replay-action KL is an estimator, not full-distribution KL or a hard post-update bound')
    return cfg


def build_preflight(cfg, output):
    result = integration.build_preflight(cfg, output)
    result['training']['post_update_kl_interval'] = 1
    if result['offline'].get('branch_exploration_enabled'):
        result['offline']['exploration_interval'] = 1
    result['experiment_protocol']['comparison_scope'] = (
        'Two-epoch formal-batch GPU preflight, including enabled KL/search paths; reduced validation only; discarded before formal training')
    return result


def prepare(args):
    selected = args.arms.split(',')
    if not selected or len(set(selected)) != len(selected) or any(a not in ARMS for a in selected):
        raise ValueError('arms must be distinct names from: '+','.join(ARMS))
    _check_optimization(args.learning_rate, args.target_kl)
    if args.run_id is None:
        args.run_id = f'PHYSICS_EXPLORATION_VRPTW100_S{args.seed}_E{args.epochs}_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    variants = {arm: dict(gamma=.99 if arm == 'legacy' else 1.,
        reward_norm_mode='physical_shared_popart', physical_contract=arm != 'legacy',
        structural_archive=arm in ('archive', 'explore'), independent_search=arm == 'explore')
        for arm in selected}
    protocol = dict(phase='physics_exploration_incremental', variants=variants,
        learning_rate=float(args.learning_rate), target_kl=args.target_kl,
        input_coordinate_mode='depot_fixed', input_distance_scale_km=DISTANCE_UNIT_KM,
        model='combined stage-two static fusion, directed relations, resource decoder and dual readout in every arm',
        initial_evaluation_pairs=[list(pair) for pair in INITIAL_PAIRS if all(a in selected for a in pair)],
        initial_evaluation_consistency_scope='physics_archive_explore_only',
        initialization_caveat='Same checkpoint is not necessarily the same initial policy: physics updates existing AGDA feature semantics. Record epoch-zero migration cost; archive/explore share physics semantics and should match.',
        initialization='same bundled mature Norm epoch-300 weights; fresh optimizer/replay/actor RMS/PopArt; full-batch preflight weights discarded',
        gpu_preflight=dict(epochs=2, instances_per_rollout=64, n_traj=50,
            chunk_size=args.chunk_size, num_minibatches=4, ppo_update_epochs=5,
            validation_instances=4, validation_n_traj=4, exploration_interval=1,
            scope='formal training allocation, including enabled independent search; reduced validation'),
        comparison_scope='four incremental bundles; repeat across seeds before selecting or attributing gains',
        extra_search_budget=dict(arm='explore', interval=5, max_instances=8,
            trajectories_per_instance=8, max_trajectories_per_event=64),
        compute_caveat='Report search trajectories and wall time. Equal PPO rollout budgets do not make the exploration arm compute-matched.',
        normalization_bundle='frozen physical reward/input units, actor historical RMS, PopArt, shared physical-cost SL; gamma and PPO reduction change only after legacy',
        stage='300-epoch local fine-tuning screen unless overridden; validation selection only, no test tuning or generalization claim')
    return shared.prepare(args,
        arm_definitions={arm: (.99 if arm == 'legacy' else 1., 'physical_shared_popart') for arm in ARMS},
        arm_builder=partial(build_arm, learning_rate=args.learning_rate, target_kl=args.target_kl),
        default_checkpoint=DEFAULT_CHECKPOINT, protocol_overrides=protocol,
        preflight_builder=build_preflight, prerequisite_source_run=None)


def make_parser():
    parser = inputs.make_parser()
    parser.description = __doc__
    parser.set_defaults(base_config=CODE_ROOT/'configs/experiments/physics_exploration_vrptw100.yaml',
                        seed=3010, chunk_size=15, arms=','.join(ARMS))
    for action in parser._actions:
        if action.dest == 'arms':
            action.help = 'Incremental arms (default: legacy,physics,archive,explore)'
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Same constant LR for every arm; default 1e-5')
    parser.add_argument('--target-kl', type=float, default=None,
                        help='Optional fresh pass-end KL stopping threshold; omitted keeps five passes')
    return parser


def main():
    args = make_parser().parse_args()
    if args.supervise:
        shared.supervise(args.supervise.resolve())
    else:
        prepare(args)


if __name__ == '__main__':
    main()
