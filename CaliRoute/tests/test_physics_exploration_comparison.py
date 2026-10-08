"""Four-arm preparation must isolate declared bundles and preserve real budgets."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_physics_exploration_comparison as launch
import run_reward_norm_comparison as shared

UNITS = dict(reward_distance_scale_km=launch.DISTANCE_UNIT_KM,
             observation_distance_scale_km=launch.DISTANCE_UNIT_KM)


def source():
    return yaml.safe_load((launch.CODE_ROOT/'configs/experiments/physics_exploration_vrptw100.yaml').read_text())


def arm(tmp_path, name, **options):
    return launch.build_arm(source(), arm=name, output=tmp_path/name, run_name=name,
        init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path/'data', seed=3010,
        units=UNITS, **options)


def test_common_architecture_and_budgets_with_explicit_incremental_changes(tmp_path):
    configs = {name: arm(tmp_path, name) for name in launch.ARMS}
    for name, cfg in configs.items():
        physics = name != 'legacy'
        archive = name in ('archive', 'explore')
        assert cfg['model']['use_typed_static_fusion']
        assert cfg['model']['use_edge_relation_encoder']
        assert cfg['model']['use_resource_decoder']
        assert cfg['model']['decoder_observation_mode'] == 'dual'
        assert not cfg['model']['use_edge_value_messages']
        assert not cfg['model']['use_edge_state_updates']
        assert cfg['data']['strict_road_metric'] is physics
        assert cfg['env']['prefer_explicit_edge_matrices'] is physics
        assert cfg['env']['reward_contract'] == ('strict_distance' if physics else 'legacy')
        assert cfg['env']['failure_penalty_km'] == 1000
        for key, value in UNITS.items():
            assert cfg['env'][key] == value
        assert cfg['model']['agda_physical_candidate_features'] is physics
        assert cfg['model']['agda_smooth_distance_features'] is physics
        assert cfg['training']['gamma'] == (1. if physics else .99)
        assert cfg['training']['ppo_loss_reduction'] == ('valid_actions' if physics else 'legacy_step_mean')
        assert cfg['training']['bootstrap_truncation'] is physics
        assert cfg['offline']['policy_replay_selection'] == ('structural' if archive else 'legacy')
        assert cfg['offline']['policy_replay_exploration_capacity'] == (4 if archive else 0)
        assert cfg['offline']['branch_exploration_enabled'] is (name == 'explore')
        assert not cfg['offline']['exploration_enabled']
        assert cfg['training']['epochs'] == 300
        assert cfg['training']['num_envs_per_gpu'] == 64
        assert cfg['training']['n_traj'] == 50
        assert cfg['training']['ppo_update_epochs'] == 5
        assert cfg['training']['num_minibatches'] == 4
        assert cfg['training']['learning_rate'] == 1e-5
        assert cfg['training']['target_kl'] is None
        assert cfg['evaluation']['eval_interval'] == 50
        assert cfg['evaluation']['eval_before_training']
        assert cfg['evaluation']['eval_n_traj'] == 50
        assert cfg['evaluation']['eval_seed'] == 17003010
        assert 'eval_limit' not in cfg['evaluation']
        assert not any(cfg['pbrs'][key] for key in ('use_customer_pbrs', 'use_repair_distance_pbrs', 'use_feasible_ratio_pbrs', 'use_terminal_heuristic'))
    # Archive/search do not change policy features or PPO optimizer settings.
    physics = configs['physics']
    for name in ('archive', 'explore'):
        assert configs[name]['model'] == physics['model']
        assert configs[name]['env'] == physics['env']
        trained = dict(configs[name]['training'])
        expected = dict(physics['training'])
        trained.pop('monitor_output_dir'); expected.pop('monitor_output_dir')
        assert trained == expected


def test_preflight_uses_formal_shape_and_exercises_search_and_kl(tmp_path):
    cfg = arm(tmp_path, 'explore', learning_rate=1.5e-5, target_kl=.02)
    before = copy.deepcopy(cfg)
    smoke = launch.build_preflight(cfg, tmp_path/'preflight')
    assert smoke['training']['epochs'] == 2
    assert smoke['training']['num_envs_per_gpu'] == 64
    assert smoke['training']['n_traj'] == 50
    assert smoke['training']['ppo_step_chunk_size'] == 15
    assert smoke['training']['learning_rate'] == 1.5e-5
    assert smoke['training']['target_kl'] == .02
    assert smoke['training']['post_update_kl_interval'] == 1
    assert smoke['offline']['exploration_interval'] == 1
    assert smoke['offline']['exploration_instances'] == 8
    assert smoke['offline']['exploration_trajectories'] == 8
    assert smoke['evaluation']['eval_limit'] == 4
    assert smoke['evaluation']['eval_n_traj'] == 4
    assert cfg == before


@pytest.mark.parametrize('options', [dict(learning_rate=0), dict(learning_rate=float('nan')), dict(target_kl=-1), dict(target_kl=float('inf'))])
def test_invalid_optimizer_controls_fail_before_preparation(tmp_path, options):
    with pytest.raises(ValueError):
        arm(tmp_path, 'physics', **options)


def test_prepare_only_freezes_configuration_and_never_starts_training(tmp_path, monkeypatch):
    root = tmp_path/'project/CaliRoute'
    root.mkdir(parents=True)
    data = root.parent/'AAAI_Dataset'
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')), ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        directory = data/'dataset/vrptw'/split/'Cus100'
        directory.mkdir(parents=True)
        for name in names:
            (directory/name).write_text('fixture\n')
    checkpoint = root/launch.DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixed-initialization')
    monkeypatch.setattr(shared, 'CODE_ROOT', root)
    monkeypatch.setattr(shared, 'probe_requested_gpus', lambda _: None)
    monkeypatch.setattr(shared, 'load_initialization', lambda path, epoch, metadata_path=None: (
        dict(config=dict(env=UNITS)), dict(sha256=shared.digest(path))))
    def snapshot(path):
        path.mkdir(parents=True)
        return dict(files={}, git_commit='fixture', content_sha256='fixture')
    monkeypatch.setattr(shared, 'source_snapshot', snapshot)
    monkeypatch.setattr(shared.subprocess, 'Popen', lambda *a, **k: pytest.fail('prepare-only started a process'))
    args = launch.make_parser().parse_args(['--prepare-only', '--run-id', 'PHYSICS_PREPARE_TEST',
        '--init-checkpoint', str(checkpoint), '--data-root', str(data),
        '--epochs', '150', '--learning-rate', '1.5e-5', '--target-kl', '.02'])
    run = launch.prepare(args)
    manifest = json.loads((run/'manifest.json').read_text())
    assert tuple(manifest['arms']) == launch.ARMS
    assert manifest['gpus'] == [0, 1, 2, 3]
    assert manifest['protocol']['phase'] == 'physics_exploration_incremental'
    assert manifest['protocol']['initial_evaluation_pairs'] == [list(pair) for pair in launch.INITIAL_PAIRS]
    assert manifest['protocol']['learning_rate'] == 1.5e-5
    assert manifest['protocol']['target_kl'] == .02
    assert manifest['protocol']['extra_search_budget']['max_trajectories_per_event'] == 64
    assert manifest['wait_for_experiments'] == []
    for name, spec in manifest['arms'].items():
        assert spec['required_validation_epochs'] == [0, 50, 100, 150]
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        smoke = yaml.safe_load(Path(spec['preflight']['config']).read_text())
        assert cfg['offline']['init_checkpoint_path'] == manifest['init_checkpoint']
        assert cfg['training']['learning_rate'] == 1.5e-5
        assert cfg['training']['epochs'] == 150
        assert smoke['training']['num_envs_per_gpu'] == 64
        assert smoke['training']['n_traj'] == 50
        assert cfg['experiment_protocol']['input_normalization_signature'] == launch.input_signature(cfg)
        assert cfg['experiment_protocol']['model_integration'] == launch.model_signature(cfg)
    shared.verify_manifest(manifest)


def test_defaults_and_shell_allow_user_overrides(tmp_path):
    args = launch.make_parser().parse_args([])
    assert args.arms == 'legacy,physics,archive,explore'
    assert args.epochs == 300 and args.eval_interval == 50
    assert args.init_checkpoint == launch.CODE_ROOT/launch.DEFAULT_CHECKPOINT
    assert not args.launch and args.target_kl is None
    fake = tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env python3\nimport sys\nprint("\\n".join(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(fake))
    for key in ('SEED','GPUS','DATA_ROOT','RUN_ID'):
        env.pop(key, None)
    script = launch.CODE_ROOT/'scripts/run_physics_exploration_comparison.sh'
    default = subprocess.check_output(['bash', str(script)], env=env, text=True).splitlines()
    assert '--launch' in default
    assert default[1].endswith('run_physics_exploration_comparison.py')
    prepared = subprocess.check_output(['bash', str(script), '--prepare-only', '--seed', '3011'], env=env, text=True).splitlines()
    assert '--launch' not in prepared
    assert prepared[-2:] == ['--seed','3011']
