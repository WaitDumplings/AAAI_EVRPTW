#!/usr/bin/env python3
"""Original versus light P0/P1 VRPTW100; pure PPO warm-up then native SL-PPO.

One invocation owns two GPUs. Both arms start from scratch and save their own
PPO boundary checkpoint while continuing the same optimizer into SL-PPO.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
import run_evrptw_dual_scratch as shared
from ppo_warmup import PPOWarmupSchedule

# The old model's training-set scale, audited from its VRPTW100 saved config.
# Fixed on other customer sizes; never recomputed from validation/test data.
VRPTW_DISTANCE_UNIT_KM = 43.638668060302734


def build_config(base, *, warmup_epochs=100, **kwargs):
    if kwargs.get('task', 'vrptw') != 'vrptw':
        raise ValueError('This controlled experiment is VRPTW100')
    if kwargs.get('world_size', 2) != 2:
        raise ValueError('This protocol requires two GPUs per arm')
    if kwargs.get('encoder_variant', 'current') != 'current':
        raise ValueError('P1 uses the original Transformer, not the heavy joint graph')
    variant = kwargs['variant']
    if variant not in ('original', 'optimized'):
        raise ValueError('variant must be original or optimized')
    original_kwargs = dict(kwargs, variant='original', task='vrptw')
    cfg = shared.build_config(base, **original_kwargs)
    total = cfg['training']['epochs']
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or not 0 < warmup_epochs < total:
        raise ValueError('warmup-epochs must be positive and smaller than total epochs')
    cfg['training'].update(ppo_update_epochs=3, target_kl=None, ppo_warmup_epochs=warmup_epochs)
    if variant == 'optimized':
        # Keep original reward, GAE, SL-PPO, expert and incumbent coefficients.
        # Only opt into compact representation changes and equivalent caches.
        cfg['model'].update(use_physical_input_context=True,
            use_resource_isolation=True, use_directed_road_profile=True,
            directed_profile_hidden_dim=32, use_directed_score_mixer=True,
            directed_score_hidden=8, optimize_dynamic_projections=True,
            cache_static_observations=True, use_static_rollout_cache=True)
        cfg['env'].update(observation_input_context=True,
            observation_coordinate_mode='legacy_minmax',
            observation_distance_scale_km=VRPTW_DISTANCE_UNIT_KM)
        cfg['offline'].pop('original_share_static_expert_observations', None)
        cfg['offline']['share_expert_static_observations'] = True
    cfg['evaluation']['prefer_explicit_edge_matrices'] = True
    protocol = cfg['experiment_protocol']
    protocol.update(phase='vrptw100_original_vs_p0_p1_ppo_warmup', arm=variant,
        implementation='legacy' if variant == 'original' else 'original_p0_p1',
        ppo_update_epochs=3, attempted_optimizer_steps_per_epoch=12,
        ppo_warmup_epochs=warmup_epochs, slppo_epochs=total-warmup_epochs,
        warmup_mode='pure PPO: no expert/reference/group SL advantages, archive, search or SL loss',
        phase_transition='own checkpoint at warmup boundary; retain weights and optimizer, enable original SL-PPO next epoch',
        source_init_checkpoint=None, source_init_epoch=None,
        extra_search_enabled=False, search_budget=None,
        input_units=f'original training-set reward D0; original coordinates; P1 observation D0 fixed at {VRPTW_DISTANCE_UNIT_KM} km',
        failure_handling='native original reward in both arms',
        comparison_scope='same original reward/SL coefficients, warmup, rollout/update/eval budget; improved arm adds P0 resource isolation, P1 directed node profiles and content-edge mixing plus equivalent caches',
        original_runtime=protocol.get('original_runtime') if variant == 'original' else None,
        original_algorithm_commit=shared.scratch.ORIGINAL_COMMIT,
        architecture=dict(base='original light graph Transformer',
            use_resource_isolation=variant == 'optimized',
            use_directed_road_profile=variant == 'optimized',
            use_directed_score_mixer=variant == 'optimized',
            use_joint_graph_encoder=False, training_bundle='original_slppo'),
        best_checkpoint_caveat='Compare common validation epochs and feasibility; architectures have different random initial policies.')
    if variant == 'optimized':
        protocol['original_source']['execution_requirement'] = 'configuration and algorithm reference only; this arm runs the recorded current P0/P1 source snapshot'
        protocol['protocol_overrides'] = [x for x in protocol['protocol_overrides'] if x['parameter'] != 'offline.original_share_static_expert_observations']
        for item in protocol['protocol_overrides']:
            if item['parameter'] == 'execution.world_size':
                item['reason'] = 'current synchronous distributed runtime; same two-rank budget'
    protocol['batch_controls']['ppo_passes'] = 3
    protocol['protocol_overrides'] = [x for x in protocol.get('protocol_overrides', [])
                                    if x['parameter'] != 'training.ppo_update_epochs']
    protocol['protocol_overrides'].extend([
        dict(parameter='training.ppo_update_epochs', original=4, used=3,
             reason='requested fixed three PPO passes in both phases and both arms'),
        dict(parameter='training.ppo_warmup_epochs', original=0, used=warmup_epochs,
             reason='requested pure PPO before SL-PPO; save own boundary checkpoint')])
    PPOWarmupSchedule(cfg)
    shared.scratch.assert_scratch(cfg)
    return cfg


def main():
    parser = shared.make_parser()
    parser.description = __doc__
    parser.set_defaults(task='vrptw', seed=3011, batch_per_gpu=16,
                        chunk_size=64, expert_chunk_size=128)
    parser.add_argument('--warmup-epochs', type=int, default=100)
    args = parser.parse_args()
    if args.supervise:
        shared.supervise(args.supervise.resolve())
        return
    if not args.variant:
        parser.error('--variant is required')
    if not args.run_id:
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        args.run_id = f'VRPTW100_PPO{args.warmup_epochs}_P1_{args.variant.upper()}_B{args.batch_per_gpu}_U3_S{args.seed}_E{args.epochs}_{stamp}'
    shared.prepare(args, config_builder=lambda base, **kwargs: build_config(base, warmup_epochs=args.warmup_epochs, **kwargs))


if __name__ == '__main__':
    main()
