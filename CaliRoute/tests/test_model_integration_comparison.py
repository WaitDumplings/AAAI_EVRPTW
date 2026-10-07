"""Second-stage factorial isolation, portable preparation and visible metadata."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_model_integration_comparison as launch
import run_input_norm_comparison as inputs
import run_reward_norm_comparison as shared

UNITS = dict(reward_distance_scale_km=launch.DISTANCE_UNIT_KM,
             observation_distance_scale_km=launch.DISTANCE_UNIT_KM)


def source_config():
    return yaml.safe_load((launch.CODE_ROOT / 'configs/experiments/model_integration_vrptw100.yaml').read_text())


def make_arm(tmp_path, arm, **kwargs):
    return launch.build_arm(source_config(), arm=arm, output=tmp_path/arm,
        run_name=arm, init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path/'data',
        seed=3010, units=UNITS, **kwargs)


def test_input_and_training_controls_are_identical_across_four_arms(tmp_path):
    shared_configs = []
    for arm, (static, dynamic) in launch.ARMS.items():
        cfg = make_arm(tmp_path, arm)
        assert cfg['env']['observation_coordinate_mode'] == 'depot_fixed'
        assert cfg['env']['observation_input_context'] is True
        assert cfg['model']['use_physical_input_context'] is True
        assert cfg['model']['use_typed_static_fusion'] is static
        assert cfg['model']['use_edge_relation_encoder'] is static
        assert cfg['model']['edge_relation_dim'] == 16
        assert cfg['model']['use_resource_decoder'] is dynamic
        assert cfg['model']['decoder_observation_mode'] == ('dual' if dynamic else 'feasible')
        assert cfg['model']['use_edge_value_messages'] is False
        assert cfg['model']['use_edge_state_updates'] is False
        assert cfg['offline']['init_checkpoint_strict'] is False
        assert cfg['training']['epochs'] == 300
        assert cfg['training']['ppo_update_epochs'] == 5
        assert cfg['training']['num_envs_per_gpu'] == 64
        assert cfg['training']['n_traj'] == 50
        assert cfg['training']['gamma'] == .99
        assert cfg['training']['reward_norm_mode'] == 'physical_shared_popart'
        assert cfg['training']['ppo_step_chunk_size'] == 12
        assert cfg['evaluation']['eval_interval'] == 50
        assert cfg['evaluation']['eval_before_training'] is True
        assert cfg['evaluation']['eval_seed'] == 17003010
        assert cfg['evaluation']['eval_n_traj'] == 50
        assert 'eval_limit' not in cfg['evaluation']
        assert 'eval_num_batches' not in cfg['evaluation']
        assert cfg['experiment_protocol']['initial_evaluation_equivalence_group'] == 'all_arms'
        for key in launch.model_options(arm):
            cfg['model'].pop(key)
        cfg['training'].pop('monitor_output_dir')
        cfg['evaluation'].pop('eval_output_dir')
        cfg.pop('run_name')
        cfg.pop('experiment_protocol')
        shared_configs.append(cfg)
    assert all(cfg == shared_configs[0] for cfg in shared_configs[1:])


def test_preserves_combined_input_norm_and_loss_bundle(tmp_path):
    base = source_config()
    original = copy.deepcopy(base)
    cfg = launch.build_arm(base, arm='baseline', output=tmp_path/'baseline',
        run_name='baseline', init_checkpoint=tmp_path/'shared.pt',
        data_root=tmp_path/'data', seed=3010, units=UNITS)
    expected = inputs.build_arm(base, arm='combined', output=tmp_path/'baseline',
        run_name='baseline', init_checkpoint=tmp_path/'shared.pt',
        data_root=tmp_path/'data', seed=3010, units=UNITS, chunk_size=12)
    for section in ('env', 'training', 'offline', 'evaluation', 'critic', 'advantage', 'pbrs'):
        assert cfg[section] == expected[section]
    assert cfg['experiment_protocol']['input_normalization_signature'] == expected['experiment_protocol']['input_normalization_signature']
    assert base == original


@pytest.mark.parametrize('messages,updates', [(False, False), (True, False), (False, True), (True, True)])
def test_optional_heavier_features_change_static_arms_only(tmp_path, messages, updates):
    for arm, (static, _) in launch.ARMS.items():
        cfg = make_arm(tmp_path, arm, edge_messages=messages, edge_updates=updates)
        assert cfg['model']['use_edge_value_messages'] is (static and messages)
        assert cfg['model']['use_edge_state_updates'] is (static and updates)
        assert cfg['experiment_protocol']['model_integration'] == launch.model_options(
            arm, edge_messages=messages, edge_updates=updates)


def test_contaminated_base_switches_are_explicitly_reset(tmp_path):
    base = source_config()
    base['model'].update(launch.model_options('combined', edge_messages=True, edge_updates=True))
    cfg = launch.build_arm(base, arm='baseline', output=tmp_path, run_name='baseline',
        init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path/'data', seed=3010, units=UNITS)
    assert all(cfg['model'][key] == value for key, value in launch.model_options('baseline').items())


@pytest.mark.parametrize('key', list(UNITS))
def test_frozen_physical_unit_is_enforced(tmp_path, key):
    with pytest.raises(ValueError, match='must remain'):
        launch.build_arm(source_config(), arm='combined', output=tmp_path, run_name='bad',
            init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path, seed=3010,
            units=dict(UNITS, **{key: 1.0}))


@pytest.mark.parametrize('args', [['--arms', 'static,static'], ['--arms', 'reward'],
    ['--arms', 'dynamic', '--edge-updates'], ['--arms', 'baseline', '--edge-messages']])
def test_invalid_factorials_fail_before_preparing(args, monkeypatch):
    monkeypatch.setattr(shared, 'prepare', lambda *a, **k: pytest.fail('Invalid arms must fail first'))
    with pytest.raises(ValueError):
        launch.prepare(launch.make_parser().parse_args(args))


def test_portable_defaults_are_second_stage_specific():
    args = launch.make_parser().parse_args([])
    assert args.seed == 3010
    assert args.epochs == 300
    assert args.eval_interval == 50
    assert args.chunk_size == 12
    assert args.gpus == '0,1,2,3'
    assert args.arms == 'baseline,static,dynamic,combined'
    assert args.init_checkpoint == launch.CODE_ROOT/launch.DEFAULT_CHECKPOINT
    assert args.base_config.name == 'model_integration_vrptw100.yaml'
    assert not args.edge_messages and not args.edge_updates and not args.launch
    help_text = launch.make_parser().format_help()
    assert 'baseline,static,dynamic,combined' in help_text
    assert 'legacy,depot,context,combined' not in help_text


def test_prepare_freezes_every_config_and_reuses_portable_checkpoint(tmp_path, monkeypatch):
    root = tmp_path/'portable'/'CaliRoute'
    root.mkdir(parents=True)
    data = root.parent/'AAAI_Dataset'
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')),
                         ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        directory = data/'dataset/vrptw'/split/'Cus100'
        directory.mkdir(parents=True)
        for name in names:
            (directory/name).write_text('fixture\n')
    checkpoint = root/launch.DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixed-test-checkpoint')
    calls = []
    def load(path, epoch, metadata_path=None):
        calls.append((path, epoch, metadata_path))
        return dict(config=dict(env=UNITS)), dict(sha256=shared.digest(path))
    monkeypatch.setattr(shared, 'CODE_ROOT', root)
    monkeypatch.setattr(shared, 'probe_requested_gpus', lambda _: None)
    monkeypatch.setattr(shared, 'load_initialization', load)
    def snapshot(path):
        path.mkdir(parents=True)
        return dict(files={}, git_commit='fixture', content_sha256='fixture')
    monkeypatch.setattr(shared, 'source_snapshot', snapshot)
    monkeypatch.setattr(shared.subprocess, 'Popen', lambda *a, **k: pytest.fail('prepare-only must not launch'))
    args = launch.make_parser().parse_args(['--prepare-only', '--run-id', 'MODEL_PREPARE_TEST',
        '--init-checkpoint', str(checkpoint), '--data-root', str(data), '--gpus', '0,1', '--edge-updates'])
    experiment = launch.prepare(args)
    manifest = json.loads((experiment/'manifest.json').read_text())
    assert calls == [(checkpoint.resolve(), 300, checkpoint.with_suffix('.json'))]
    assert list(manifest['arms']) == list(launch.ARMS)
    protocol = manifest['protocol']
    assert protocol['epochs'] == 300
    assert protocol['seed'] == 3010
    assert protocol['phase'] == 'physical_model_integration_2x2'
    assert protocol['initial_evaluation_pairs'] == [list(x) for x in launch.INITIAL_PAIRS]
    assert protocol['input_coordinate_mode'] == 'depot_fixed'
    assert protocol['optional_edge_updates'] is True
    assert protocol['optional_edge_messages'] is False
    assert protocol['gpu_preflight']['instances_per_rollout'] == 64
    assert protocol['gpu_preflight']['n_traj'] == 50
    assert protocol['gpu_preflight']['chunk_size'] == 12
    assert protocol['gpu_preflight']['validation_instances'] == 4
    assert protocol['test_enabled'] is False
    assert protocol['validation_instances'] == 1000
    assert protocol['eval_interval'] == 50
    assert manifest['gpus'] == [0, 1]
    assert manifest['wait_for_experiments'] == []
    frozen = Path(manifest['init_checkpoint'])
    assert frozen != checkpoint and frozen.read_bytes() == checkpoint.read_bytes()
    for arm, spec in manifest['arms'].items():
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        smoke = yaml.safe_load(Path(spec['preflight']['config']).read_text())
        assert spec['required_validation_epochs'] == [0, 50, 100, 150, 200, 250, 300]
        assert cfg['offline']['init_checkpoint_path'] == str(frozen)
        assert smoke['offline']['init_checkpoint_path'] == str(frozen)
        assert smoke['training']['epochs'] == 2
        assert smoke['training']['n_traj'] == 50
        assert smoke['training']['num_envs_per_gpu'] == 64
        assert smoke['training']['ppo_step_chunk_size'] == 12
        assert smoke['training']['ppo_update_epochs'] == cfg['training']['ppo_update_epochs'] == 5
        assert smoke['training']['num_minibatches'] == cfg['training']['num_minibatches'] == 4
        assert smoke['evaluation']['eval_n_traj'] == 4
        assert smoke['evaluation']['eval_limit'] == 4
        assert smoke['env'] == cfg['env']
        assert smoke['model'] == cfg['model']
        assert cfg['model']['use_edge_state_updates'] is launch.ARMS[arm][0]
        assert cfg['offline']['init_checkpoint_strict'] is False
    shared.verify_manifest(manifest)
    checkpoint.write_bytes(b'source edited after prepare')
    shared.verify_manifest(manifest)
    frozen.write_bytes(b'tampered frozen archive')
    with pytest.raises(ValueError, match='Frozen initialization changed'):
        shared.verify_manifest(manifest)


def test_plot_labels_identify_model_stage(tmp_path):
    import plot_reward_norm_eval as plot
    arms = {}
    for arm in launch.ARMS:
        folder = tmp_path/arm
        folder.mkdir()
        with (folder/'eval_log.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'eval_status',
                'eval_avg_objective_distance_km', 'eval_feasible_rate', 'eval_num_instances', 'eval_n_traj'])
            writer.writeheader()
            writer.writerow(dict(epoch=0, eval_status='ok', eval_avg_objective_distance_km=241,
                eval_feasible_rate=1, eval_num_instances=1000, eval_n_traj=50))
        arms[arm] = dict(log_dir=str(folder))
    manifest = dict(arms=arms, protocol=dict(phase='physical_model_integration_2x2',
        epochs=300, seed=3010, eval_interval=50, validation_instances=1000, eval_n_traj=50))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    plot.render(tmp_path)
    svg = (tmp_path/'plots/validation_curves.svg').read_text()
    assert 'Physical model integration' in svg
    assert 'Static fusion + edges' in svg
    assert 'Resource decoder' in svg
    assert 'Reward + Norm' not in svg


def test_shell_wrapper_uses_second_stage_defaults(tmp_path):
    fake_python = tmp_path/'capture-python'
    fake_python.write_text('#!/usr/bin/env python3\nimport sys\nprint("\\n".join(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    import os
    env = dict(os.environ, PYTHON_BIN=str(fake_python))
    for key in ('SEED', 'GPUS', 'DATA_ROOT', 'RUN_ID'):
        env.pop(key, None)
    script = launch.CODE_ROOT/'scripts/run_model_integration_comparison.sh'
    default = subprocess.check_output(['bash', str(script)], env=env, text=True).splitlines()
    assert default[default.index('--seed')+1] == '3010'
    assert default[default.index('--gpus')+1] == '0,1,2,3'
    assert '--launch' in default
    assert default[1].endswith('/run_model_integration_comparison.py')
    prepared = subprocess.check_output(['bash', str(script), '--prepare-only'], env=env, text=True).splitlines()
    assert '--launch' not in prepared and '--prepare-only' in prepared


@pytest.mark.parametrize('chunk_size', [8, 12])
def test_full_shape_preflight_preserves_optimizer_and_training_allocation(tmp_path, chunk_size):
    cfg = make_arm(tmp_path, 'combined', chunk_size=chunk_size)
    original = copy.deepcopy(cfg)
    smoke = launch.build_preflight(cfg, tmp_path/'preflight')
    expected = copy.deepcopy(cfg['training'])
    expected.update(epochs=2, monitor_interval=1,
                    monitor_output_dir=str(tmp_path/'preflight/monitoring'))
    assert smoke['training'] == expected
    assert smoke['offline'] == cfg['offline']
    assert smoke['env'] == cfg['env']
    assert smoke['model'] == cfg['model']
    assert smoke['evaluation']['eval_n_traj'] == 4
    assert smoke['evaluation']['eval_limit'] == 4
    assert smoke['evaluation']['eval_before_training'] is False
    assert smoke['experiment_protocol']['global_instances_per_rollout'] == 64
    assert smoke['experiment_protocol']['global_trajectories_per_rollout'] == 3200
    assert smoke['experiment_protocol']['global_instances_per_optimizer_step'] == 16
    assert cfg == original
    # Stage-one/reward launchers keep their existing small preflight budget.
    previous = shared.build_preflight(cfg, tmp_path/'generic-preflight')
    assert previous['training']['num_envs_per_gpu'] == 4
    assert previous['training']['n_traj'] == 4
    assert previous['training']['ppo_step_chunk_size'] == 4
