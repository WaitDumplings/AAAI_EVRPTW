"""Controlled original/P0+P1 protocol, including PPO-only warmup and dual budgets."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_original_p1_dual as launch
import run_evrptw_dual_scratch as common
import run_reward_norm_comparison as shared
from ppo_warmup import PPOWarmupSchedule
from test_scratch_comparison import base_config, checkpoint_paths
from test_scratch_comparison import prepared as scratch_prepared


def config(tmp_path, variant='optimized', **overrides):
    args = dict(variant=variant, output=tmp_path / variant, run_name='P1_' + variant.upper(),
                data_root=tmp_path / 'AAAI_Dataset', seed=3011, epochs=1500,
                batch_per_gpu=16, chunk_size=64, expert_chunk_size=128,
                world_size=2, task='vrptw')
    args.update(overrides)
    return launch.build_config(base_config(), **args)


def test_both_arms_share_native_original_algorithm_and_training_budget(tmp_path):
    original = config(tmp_path, 'original')
    improved = config(tmp_path, 'optimized')
    for cfg in (original, improved):
        common.scratch.assert_scratch(cfg)
        assert not list(checkpoint_paths(cfg))
        train, off, protocol = cfg['training'], cfg['offline'], cfg['experiment_protocol']
        assert train['epochs'] == 1500 and train['ppo_warmup_epochs'] == 100
        assert train['num_envs_per_gpu'] == 16 and train['n_traj'] == 50
        assert train['ppo_update_epochs'] == 3 and train['num_minibatches'] == 4
        assert train['gradient_accumulation_steps'] == 1 and train['target_kl'] is None
        assert train['learning_rate'] == 1e-4 and train['gamma'] == .99
        assert train['gae_lambda'] == .95 and train['ppo_step_chunk_size'] == 64
        assert 'reward_norm_mode' not in train and 'reward_contract' not in cfg['env']
        assert off['method'] == 'sl_ppo' and off['sl_coef'] == .5
        assert off['use_priority_sampler'] is False
        assert not off.get('branch_exploration_enabled', False)
        assert not off.get('policy_replay_enabled', False)
        assert protocol['world_size'] == 2
        assert protocol['global_instances_per_rollout'] == 32
        assert protocol['global_trajectories_per_rollout'] == 1600
        assert protocol['global_instances_per_optimizer_step'] == 8
        assert protocol['attempted_optimizer_steps_per_epoch'] == 12
        assert protocol['batch_controls']['ppo_passes'] == 3
        assert protocol['slppo_epochs'] == 1400 and protocol['ppo_warmup_epochs'] == 100
        assert protocol['search_budget'] is None and not protocol['extra_search_enabled']
        assert protocol['source_init_checkpoint'] is None and protocol['source_init_epoch'] is None
        eval_cfg = cfg['evaluation']
        assert eval_cfg['eval_n_traj'] == 50 and eval_cfg['eval_interval'] == 50
        assert eval_cfg['eval_before_training'] and 'eval_limit' not in eval_cfg
        update = [x for x in protocol['protocol_overrides'] if x['parameter'] == 'training.ppo_update_epochs']
        assert len(update) == 1 and update[0]['used'] == 3
    for section in ('pbrs', 'advantage', 'critic', 'data'):
        assert original[section] == improved[section]
    for section in ('training', 'evaluation', 'offline'):
        left, right = deepcopy(original[section]), deepcopy(improved[section])
        for key in ('monitor_output_dir', 'eval_output_dir', 'original_share_static_expert_observations', 'share_expert_static_observations'):
            left.pop(key, None)
            right.pop(key, None)
        assert left == right
    assert original['offline']['original_share_static_expert_observations']
    assert improved['offline']['share_expert_static_observations']
    allowed_model_additions = {'use_physical_input_context', 'use_resource_isolation',
        'use_directed_road_profile', 'directed_profile_hidden_dim', 'use_directed_score_mixer',
        'directed_score_hidden', 'optimize_dynamic_projections', 'cache_static_observations',
        'use_static_rollout_cache'}
    assert set(improved['model']) - set(original['model']) == allowed_model_additions
    assert all(improved['model'][key] == val for key, val in original['model'].items())
    for field in ('use_resource_isolation', 'use_directed_road_profile', 'use_directed_score_mixer'):
        assert improved['model'][field] and not original['model'].get(field, False)
    for field in ('use_joint_graph_encoder', 'use_resource_decoder', 'use_typed_static_fusion',
                  'use_edge_relation_encoder', 'use_rdi_v2', 'use_agda_v2'):
        assert not improved['model'].get(field, False)
    assert improved['env']['observation_coordinate_mode'] == 'legacy_minmax'
    assert improved['env']['observation_distance_scale_km'] == launch.VRPTW_DISTANCE_UNIT_KM


def test_builder_default_task_remains_vrptw_not_shared_evrptw_default(tmp_path):
    cfg = launch.build_config(base_config(), variant='optimized', output=tmp_path / 'a',
                              run_name='DEFAULT_TASK', data_root=tmp_path / 'dataset')
    assert cfg['data']['problem_type'] == 'vrptw'
    assert cfg['data']['num_charging_stations'] == 0
    assert cfg['training']['rollout_steps'] == 201


@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_preflight_exercises_ppo_to_sl_boundary_and_formal_restarts_from_scratch(tmp_path, variant):
    formal = config(tmp_path, variant)
    before = deepcopy(formal)
    preflight = common.preflight_config(formal, tmp_path / variant / 'preflight')
    assert formal == before
    assert preflight['training']['epochs'] == 2
    assert preflight['training']['ppo_warmup_epochs'] == 1
    for key in ('num_envs_per_gpu', 'n_traj', 'ppo_update_epochs', 'num_minibatches',
                'ppo_step_chunk_size', 'learning_rate'):
        assert preflight['training'][key] == formal['training'][key]
    assert preflight['evaluation']['eval_limit'] == 4
    assert not preflight['evaluation']['eval_before_training']
    schedule = PPOWarmupSchedule(preflight)
    assert schedule.fields(1)['effective_offline_method'] == 'ppo'
    assert schedule.fields(2)['effective_offline_method'] == 'sl_ppo'
    formal_schedule = PPOWarmupSchedule(formal)
    assert all(formal_schedule.fields(e)['effective_offline_method'] == 'ppo' for e in (1, 50, 100))
    assert formal_schedule.fields(101)['effective_offline_method'] == 'sl_ppo'
    assert formal_schedule.fields(1500)['phase_epoch'] == 1400
    metadata = formal_schedule.checkpoint_metadata(100)
    assert metadata['warmup_boundary_checkpoint']
    assert metadata['next_effective_offline_method'] == 'sl_ppo'
    assert 'no_checkpoint_reload_or_reset' in metadata['transition']
    assert not list(checkpoint_paths(formal)) and not list(checkpoint_paths(preflight))
    assert formal['run_name'] != preflight['run_name']


@pytest.mark.parametrize('kwargs,match', [
    ({'task': 'evrptw'}, 'VRPTW100'),
    ({'world_size': 1}, 'two GPUs'),
    ({'encoder_variant': 'graph'}, 'original Transformer'),
    ({'warmup_epochs': 0}, 'warmup-epochs'),
    ({'warmup_epochs': True}, 'warmup-epochs'),
    ({'warmup_epochs': 1500}, 'warmup-epochs'),
    ({'epochs': 50}, 'warmup-epochs'),
])
def test_controlled_protocol_rejects_incompatible_requests(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        config(tmp_path, **kwargs)


@pytest.mark.parametrize('variant,gpus', [('original', '0,1'), ('optimized', '2,3')])
def test_shells_launch_disjoint_gpu_pairs_with_tunable_memory_budget(tmp_path, monkeypatch, variant, gpus):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake))
    for key in ('GPUS', 'DATA_ROOT', 'RUN_ID'):
        env.pop(key, None)
    script = ROOT / 'scripts' / f'run_vrptw100_p1_{variant}_dual.sh'
    calls = []
    monkeypatch.setattr(common, 'prepare', lambda args, **kwargs: calls.append((args, kwargs)))
    for extra in ([], ['--prepare-only', '--batch-per-gpu', '8', '--chunk-size', '96', '--run-id', 'TUNED']):
        command = json.loads(subprocess.check_output(['bash', str(script), *extra], text=True, env=env))
        assert command[:2] == ['-B', str(ROOT / 'scripts/run_original_p1_dual.py')]
        monkeypatch.setattr(sys, 'argv', command[1:])
        launch.main()
    defaults, tuned = (record[0] for record in calls)
    assert defaults.variant == variant and defaults.gpus == gpus and defaults.launch
    assert defaults.task == 'vrptw' and defaults.warmup_epochs == 100
    assert defaults.batch_per_gpu == 16 and defaults.chunk_size == 64
    assert defaults.epochs == 1500 and defaults.seed == 3011
    assert tuned.prepare_only and not tuned.launch
    assert tuned.batch_per_gpu == 8 and tuned.chunk_size == 96 and tuned.run_id == 'TUNED'


@pytest.mark.parametrize('variant', ['original', 'optimized'])
def test_prepare_records_fixed_three_passes_and_own_warmup_boundary(scratch_prepared, monkeypatch, variant):
    state = scratch_prepared
    monkeypatch.setattr(common, 'CODE_ROOT', state.root)
    data = state.root.parent / 'AAAI_Dataset'
    for split, count in [('train', 5000), ('val', 1000)]:
        folder = data / 'dataset/vrptw' / split / 'Cus100'
        (folder / 'metadata.json').write_text(json.dumps(dict(num_instances=count,
            num_customers=100, num_charging_stations=0)))
    args = common.make_parser().parse_args(['--task', 'vrptw', '--variant', variant,
        '--prepare-only', '--gpus', '0,1' if variant == 'original' else '2,3',
        '--run-id', 'P1_' + variant.upper(), '--batch-per-gpu', '16', '--chunk-size', '64',
        '--data-root', str(data), '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml')])
    run = common.prepare(args, config_builder=launch.build_config)
    manifest = json.loads((run / 'manifest.json').read_text())
    protocol = manifest['protocol']
    assert manifest['initialization_mode'] == 'scratch' and manifest['init_checkpoint'] is None
    assert manifest['source_init_checkpoint'] is None and not state.launches
    assert manifest['gpus'] == ([0, 1] if variant == 'original' else [2, 3])
    assert protocol['ppo_update_epochs'] == 3 and protocol['num_minibatches'] == 4
    assert protocol['global_batch'] == 32 and protocol['global_trajectories_per_rollout'] == 1600
    assert protocol['gpu_preflight']['ppo_passes'] == 3
    assert protocol['gpu_preflight']['ppo_warmup_epochs'] == 1
    assert protocol['ppo_warmup_epochs'] == 100 and protocol['slppo_epochs'] == 1400
    assert protocol['search_budget'] is None and protocol['original_algorithm_commit'] == common.scratch.ORIGINAL_COMMIT
    spec = manifest['arms'][variant]
    assert spec['required_validation_epochs'] == list(range(0, 1501, 50))
    for phase in (spec, spec['preflight']):
        assert '--nproc-per-node=2' in phase['command']
        if variant == 'original':
            assert '--source-root' in phase['command']
            assert manifest['additional_sources']['original']['code_root'] == phase['code_root']
        else:
            assert 'offline2online.train' in phase['command']
            assert not manifest['additional_sources']
        cfg = yaml.safe_load(Path(phase['config']).read_text())
        assert cfg['training']['ppo_update_epochs'] == 3
        assert cfg['training']['ppo_warmup_epochs'] == (1 if phase is spec['preflight'] else 100)
        common.scratch.assert_scratch(cfg)
    shared.verify_manifest(manifest)
