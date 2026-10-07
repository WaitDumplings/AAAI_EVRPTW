"""Input factorial, preserved training controls, and paired initialization checks."""
import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_input_norm_comparison as launch
import run_reward_norm_comparison as shared

UNITS = dict(reward_distance_scale_km=launch.DISTANCE_UNIT_KM,
             observation_distance_scale_km=launch.DISTANCE_UNIT_KM)


def source_config():
    return yaml.safe_load((launch.CODE_ROOT / 'configs/experiments/input_norm_vrptw100.yaml').read_text())


def make_arm(tmp_path, arm, **kwargs):
    return launch.build_arm(source_config(), arm=arm, output=tmp_path / arm,
        run_name=arm, init_checkpoint=tmp_path / 'shared.pt',
        data_root=tmp_path / 'data', seed=3010, units=UNITS, **kwargs)


def test_only_coordinate_and_context_factors_change(tmp_path):
    common = []
    for arm, (coordinates, context) in launch.ARMS.items():
        cfg = make_arm(tmp_path, arm)
        assert cfg['env']['observation_coordinate_mode'] == coordinates
        assert cfg['env']['observation_input_context'] is context
        assert cfg['model']['use_physical_input_context'] is context
        assert cfg['model']['physical_input_context_hidden_dim'] == 32
        assert cfg['offline']['init_checkpoint_strict'] is not context
        assert cfg['training']['gamma'] == .99
        assert cfg['training']['reward_norm_mode'] == 'physical_shared_popart'
        assert cfg['training']['ppo_update_epochs'] == 5
        assert cfg['training']['n_traj'] == 50
        assert cfg['training']['epochs'] == 300
        assert cfg['evaluation']['eval_before_training'] is True
        assert cfg['evaluation']['eval_interval'] == 50
        assert cfg['evaluation']['eval_n_traj'] == 50
        assert cfg['evaluation']['eval_seed'] == 17003010
        assert 'eval_limit' not in cfg['evaluation']
        assert 'eval_num_batches' not in cfg['evaluation']
        signature = cfg['experiment_protocol']['input_normalization_signature']
        assert signature['observation_coordinate_mode'] == coordinates
        assert signature['observation_distance_scale_km'] == launch.DISTANCE_UNIT_KM
        assert len(signature['node_context_features']) == (12 if context else 0)
        assert len(signature['graph_context_features']) == (10 if context else 0)
        for section, key in [('env', 'observation_coordinate_mode'),
                             ('env', 'observation_input_context'),
                             ('model', 'use_physical_input_context'),
                             ('offline', 'init_checkpoint_strict'),
                             ('training', 'monitor_output_dir'),
                             ('evaluation', 'eval_output_dir')]:
            cfg[section].pop(key)
        cfg.pop('experiment_protocol')
        cfg.pop('run_name')
        common.append(cfg)
    assert all(cfg == common[0] for cfg in common[1:])


def test_shared_norm_optimization_and_loss_configuration_is_preserved(tmp_path):
    base = source_config()
    original = copy.deepcopy(base)
    cfg = make_arm(tmp_path, 'legacy')
    expected = shared.build_arm(base, arm='normalization', output=tmp_path/'legacy',
        run_name='legacy', init_checkpoint=tmp_path/'shared.pt', data_root=tmp_path/'data',
        seed=3010, units=UNITS, epochs=300, eval_interval=50)
    for section in ('training', 'offline', 'evaluation', 'critic', 'advantage', 'pbrs'):
        assert cfg[section] == expected[section]
    assert base == original


@pytest.mark.parametrize('key', list(UNITS))
def test_changed_physical_units_are_rejected(tmp_path, key):
    units = dict(UNITS, **{key: 1.0})
    with pytest.raises(ValueError, match='must remain'):
        launch.build_arm(source_config(), arm='combined', output=tmp_path,
            run_name='wrong', init_checkpoint=tmp_path/'init.pt', data_root=tmp_path,
            seed=3009, units=units)


def report_for(tmp_path, rows, *, expected_count=1000):
    specs = {}
    for arm, row in rows.items():
        folder = tmp_path/arm
        folder.mkdir(exist_ok=True)
        with (folder/'eval_log.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'eval_status',
                'eval_avg_objective_distance_km', 'eval_feasible_rate', 'eval_num_instances'])
            writer.writeheader()
            writer.writerow(dict(epoch=0, eval_status='ok', **row))
        specs[arm] = dict(log_dir=str(folder))
    manifest = dict(arms=specs, protocol=dict(initial_evaluation_pairs=launch.INITIAL_PAIRS,
        validation_instances=expected_count))
    return shared.comparison_report(manifest, dict(state='running', arms={}))


def test_epoch_zero_checks_only_matching_coordinate_pairs(tmp_path):
    def row(value):
        return dict(eval_avg_objective_distance_km=value, eval_feasible_rate=1.0, eval_num_instances=1000)
    report = report_for(tmp_path, dict(legacy=row(241), context=row(241), depot=row(290), combined=row(290)))
    assert report['initial_evaluation_consistent'] is True
    assert report['initial_evaluation_pairs'] == dict(legacy__context=True, depot__combined=True)
    assert report['initial_distance_delta_from_legacy_km']['depot'] == 49
    assert report['initial_evaluation_consistency_scope'] == 'within_coordinate_mode_pairs_only'


@pytest.mark.parametrize('changed', [dict(eval_avg_objective_distance_km=245),
    dict(eval_feasible_rate=.99), dict(eval_num_instances=32)])
def test_epoch_zero_distance_coverage_or_count_mismatch_is_visible(tmp_path, changed):
    row = dict(eval_avg_objective_distance_km=241, eval_feasible_rate=1.0, eval_num_instances=1000)
    rows = {arm: dict(row) for arm in launch.ARMS}
    rows['context'].update(changed)
    report = report_for(tmp_path, rows)
    assert report['initial_evaluation_consistent'] is False
    assert report['initial_evaluation_pairs']['legacy__context'] is False
    assert report['initial_evaluation_pairs']['depot__combined'] is True


def test_incomplete_epoch_zero_read_stays_pending(tmp_path):
    row = dict(eval_avg_objective_distance_km=241, eval_feasible_rate=1.0, eval_num_instances=1000)
    rows = {arm: dict(row) for arm in launch.ARMS}
    rows['context']['eval_avg_objective_distance_km'] = ''
    report = report_for(tmp_path, rows)
    assert report['initial_evaluation_consistent'] is None
    assert report['initial_evaluation_pairs']['legacy__context'] is None


def test_prepared_manifest_reuses_freeze_preflight_and_shared_supervision(tmp_path, monkeypatch):
    root = tmp_path/'portable'/'CaliRoute'
    root.mkdir(parents=True)
    data = root.parent/'AAAI_Dataset'
    for split, names in [('train', ('instances.pkl', 'expert_solutions.csv')),
                         ('val', ('instances.pkl', 'gurobi_summary.csv'))]:
        directory = data/'dataset/vrptw'/split/'Cus100'
        directory.mkdir(parents=True)
        for name in names:
            (directory/name).write_text('test fixture\n')
    checkpoint = root/launch.DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixed-test-checkpoint')
    metadata_calls = []
    def load(path, epoch, metadata_path=None):
        metadata_calls.append((path, epoch, metadata_path))
        return dict(config=dict(env=UNITS)), dict(sha256=shared.digest(path))
    monkeypatch.setattr(shared, 'CODE_ROOT', root)
    monkeypatch.setattr(shared, 'probe_requested_gpus', lambda _: None)
    monkeypatch.setattr(shared, 'load_initialization', load)
    snapshots = []
    def snapshot(path):
        snapshots.append(path)
        path.mkdir(parents=True)
        return dict(files={}, git_commit='test-fixture', content_sha256='test-fixture')
    monkeypatch.setattr(shared, 'source_snapshot', snapshot)
    monkeypatch.setattr(shared.subprocess, 'Popen', lambda *a, **k: pytest.fail('prepare-only must not launch'))
    args = launch.make_parser().parse_args(['--prepare-only', '--run-id', 'INPUT_PREPARE_TEST',
        '--init-checkpoint', str(checkpoint), '--data-root', str(data), '--gpus', '0,1', '--seed', '3010'])
    experiment = launch.prepare(args)
    manifest = json.loads((experiment/'manifest.json').read_text())
    assert metadata_calls == [(checkpoint.resolve(), 300, checkpoint.with_suffix('.json'))]
    assert snapshots == [experiment/'source/CaliRoute']
    assert list(manifest['arms']) == list(launch.ARMS)
    assert manifest['protocol']['epochs'] == 300
    assert manifest['protocol']['initial_evaluation_pairs'] == [list(x) for x in launch.INITIAL_PAIRS]
    assert manifest['protocol']['test_enabled'] is False
    assert manifest['protocol']['eval_interval'] == 50
    assert manifest['protocol']['eval_n_traj'] == 50
    assert manifest['gpus'] == [0, 1]
    assert len(manifest['inputs']) == 4
    assert manifest['wait_for_experiments'] == []
    frozen_input = Path(manifest['init_checkpoint'])
    assert frozen_input != checkpoint and frozen_input.read_bytes() == checkpoint.read_bytes()
    for arm, spec in manifest['arms'].items():
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        smoke = yaml.safe_load(Path(spec['preflight']['config']).read_text())
        assert spec['required_validation_epochs'] == [0, 50, 100, 150, 200, 250, 300]
        assert cfg['offline']['init_checkpoint_path'] == str(frozen_input)
        assert smoke['offline']['init_checkpoint_path'] == str(frozen_input)
        assert smoke['training']['epochs'] == 2
        assert smoke['env']['observation_coordinate_mode'] == cfg['env']['observation_coordinate_mode']
        assert smoke['model']['use_physical_input_context'] == cfg['model']['use_physical_input_context']
        assert spec['command'][4:6] == ['offline2online.train', '--config']
    shared.verify_manifest(manifest)
    checkpoint.write_bytes(b'source edited after prepare')
    shared.verify_manifest(manifest)  # Frozen copied archive is authoritative.
    frozen_input.write_bytes(b'tampered frozen archive')
    with pytest.raises(ValueError, match='Frozen initialization changed'):
        shared.verify_manifest(manifest)


def test_portable_cli_defaults_and_budget():
    args = launch.make_parser().parse_args([])
    assert args.epochs == 300
    assert args.eval_interval == 50
    assert args.arms == 'legacy,depot,context,combined'
    assert args.init_checkpoint == launch.CODE_ROOT/launch.DEFAULT_CHECKPOINT
    assert not args.launch
