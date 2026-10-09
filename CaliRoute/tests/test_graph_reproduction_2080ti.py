"""2080Ti Graph/CURRENT controls preserve the audited remote experiment.

These tests exercise configuration only: no CUDA probes, jobs or checkpoints.
They deliberately compare the entire effective configuration with the recorded
remote provenance, rather than maintain a second hand-picked model preset.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_graph_reproduction_2080ti as launch
import run_evrptw_dual_scratch as shared
from test_scratch_comparison import prepared as scratch_prepared

PROVENANCE = json.loads((ROOT / 'docs/experiments/graph_rdi100_20261008_provenance.json').read_text())


def reference(task, encoder_variant):
    batch = 40 if task == 'vrptw' else 32
    name = f'{encoder_variant.upper()}_{task.upper()}100_S3011_B{batch}_E1500_20261008_r2'
    return deepcopy(PROVENANCE['runs'][name]['effective_config'])


def base():
    return yaml.safe_load((ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml').read_text())


def build(tmp_path, task='vrptw', encoder_variant='graph', **changes):
    kwargs = dict(task=task, encoder_variant=encoder_variant, variant='optimized',
                  output=tmp_path / 'optimized', run_name='CPU_CONFIG_ONLY',
                  data_root=tmp_path / 'AAAI_Dataset', seed=3011, epochs=1500,
                  eval_interval=50, world_size=2,
                  batch_per_gpu=20 if task == 'vrptw' else 16,
                  chunk_size=24, expert_chunk_size=64, learning_rate=1e-4)
    kwargs.update(changes)
    return launch.build_config(base(), **kwargs)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {path: leaf for key, child in value.items()
                for path, leaf in flatten(child, f'{prefix}.{key}' if prefix else key).items()}
    return {prefix: value}


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
@pytest.mark.parametrize('encoder_variant', ['graph', 'current'])
def test_entire_effective_semantic_config_matches_remote_provenance(tmp_path, task, encoder_variant):
    cfg = build(tmp_path, task, encoder_variant)
    source, actual = flatten(reference(task, encoder_variant)), flatten(cfg)
    # Only execution/storage changes are allowed here. Entire protocol metadata
    # may describe local execution, but actual model/loss settings must match.
    permitted = {
        'run_name', 'data.train_dataset_path', 'offline.expert_dataset_path',
        'offline.expert_solution_path', 'evaluation.eval_path',
        'evaluation.gurobi_summary_path', 'evaluation.eval_output_dir',
        'training.monitor_output_dir', 'training.num_envs_per_gpu',
        'training.ppo_step_chunk_size', 'offline.sl_expert_logprob_chunk_size',
        'advantage.sl_expert_logprob_chunk_size', 'offline.exploration_instances',
        'offline.policy_replay_max_new_routes',
    }
    missing = object()
    drift = {key: (source.get(key, missing), actual.get(key, missing))
             for key in source.keys() | actual.keys()
             if not key.startswith('experiment_protocol.') and key not in permitted
             and source.get(key, missing) != actual.get(key, missing)}
    assert not drift, drift
    launch.verify_reference_config(cfg, task=task, encoder_variant=encoder_variant, world_size=2)
    shared.scratch.assert_scratch(cfg)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_both_encoders_have_identical_data_loss_sampling_and_eval_settings(tmp_path, task):
    graph = build(tmp_path, task, 'graph')
    current = build(tmp_path, task, 'current')
    for section in ('data', 'env', 'training', 'offline', 'advantage', 'evaluation', 'pbrs', 'critic'):
        assert graph[section] == current[section], section
    changed = {key for key in graph['model'].keys() | current['model'].keys()
               if graph['model'].get(key) != current['model'].get(key)}
    assert changed == {'use_joint_graph_encoder', 'use_edge_relation_encoder',
                       'joint_graph_edge_dim', 'joint_graph_dropout'}
    assert graph['model']['joint_graph_edge_dim'] == 32
    assert graph['model']['use_joint_graph_encoder'] and not graph['model']['use_edge_relation_encoder']
    assert current['model']['use_edge_relation_encoder'] and not current['model']['use_joint_graph_encoder']
    for cfg in (graph, current):
        assert cfg['model']['use_resource_decoder'] and cfg['model']['use_agda_v2']
        assert not cfg['model'].get('use_directed_road_profile', False)
        assert not cfg['model'].get('use_directed_score_mixer', False)


@pytest.mark.parametrize('task,global_batch', [('vrptw', 40), ('evrptw', 32)])
@pytest.mark.parametrize('encoder_variant', ['graph', 'current'])
def test_world_size_conversion_preserves_main_search_and_archive_intake_budgets(tmp_path, task, global_batch, encoder_variant):
    cfg = build(tmp_path, task, encoder_variant)
    train, offline, protocol = cfg['training'], cfg['offline'], cfg['experiment_protocol']
    assert train['num_envs_per_gpu'] * 2 == global_batch
    assert train['n_traj'] == 50
    assert train['ppo_update_epochs'] == 5 and train['num_minibatches'] == 4
    assert train['gradient_accumulation_steps'] == 1
    assert train['target_kl'] is None and train['learning_rate'] == 1e-4
    assert train.get('ppo_warmup_epochs', 0) == 0 and offline['method'] == 'sl_ppo'
    assert protocol['world_size'] == 2
    assert protocol['global_instances_per_rollout'] == global_batch
    assert protocol['global_trajectories_per_rollout'] == global_batch * 50
    assert protocol['global_instances_per_optimizer_step'] == global_batch // 4
    assert offline['exploration_instances'] == 4 and offline['exploration_trajectories'] == 8
    assert offline['exploration_interval'] == 5
    assert offline['exploration_instances'] * offline['exploration_trajectories'] * 2 == 64
    assert offline['policy_replay_max_new_routes'] == 16  #32 globally, NOT32/rank.
    assert offline['policy_replay_capacity'] == 3
    assert offline['policy_replay_exploration_capacity'] == 4  #Per instance, do not halve.
    assert offline['policy_replay_fraction'] == .25
    actual_candidates = 2 * min(offline['policy_replay_max_candidates'], train['num_envs_per_gpu'] // 4)
    remote = reference(task, encoder_variant)
    expected_candidates = min(remote['offline']['policy_replay_max_candidates'], global_batch // 4)
    assert actual_candidates == expected_candidates
    assert train['reward_norm_mode'] == 'physical_shared_popart' and train['gamma'] == 1.


@pytest.mark.parametrize('section,key,value', [
    ('training', 'ppo_update_epochs', 3),
    ('training', 'n_traj', 25),
    ('training', 'num_minibatches', 2),
    ('training', 'gradient_accumulation_steps', 2),
    ('training', 'learning_rate', 2e-4),
    ('training', 'gamma', .99),
    ('training', 'target_kl', .02),
    ('training', 'reward_norm_mode', 'legacy'),
    ('training', 'ppo_loss_reduction', 'legacy_step_mean'),
    ('model', 'n_encode_layers', 3),
    ('model', 'use_rdi_v2', False),
    ('model', 'use_agda_v2', False),
    ('model', 'decoder_observation_mode', 'feasible'),
    ('env', 'observation_coordinate_mode', 'legacy_minmax'),
    ('env', 'prefer_explicit_edge_matrices', False),
    ('env', 'reward_contract', 'legacy'),
    ('offline', 'sl_coef', .35),
    ('offline', 'branch_exploration_enabled', False),
    ('offline', 'exploration_interval', 10),
    ('offline', 'policy_replay_capacity', 1),
    ('offline', 'policy_replay_weight', .2),
    ('evaluation', 'eval_n_traj', 100),
    ('evaluation', 'eval_seed', 3011),
    ('advantage', 'sl_expert_candidate_weight', .3),
])
def test_reference_guard_rejects_unadvertised_algorithm_or_eval_changes(tmp_path, section, key, value):
    cfg = build(tmp_path)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        launch.verify_reference_config(cfg, task='vrptw', encoder_variant='graph', world_size=2)


@pytest.mark.parametrize('section,key,value', [
    ('training', 'num_envs_per_gpu', 40),
    ('offline', 'exploration_instances', 8),
    ('offline', 'policy_replay_max_new_routes', 32),
])
def test_reference_guard_does_not_allow_per_rank_budget_to_silently_double(tmp_path, section, key, value):
    cfg = build(tmp_path)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        launch.verify_reference_config(cfg, task='vrptw', encoder_variant='graph', world_size=2)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_preflight_retains_actual_training_and_validation_memory_shape(tmp_path, task):
    cfg = build(tmp_path, task)
    trial = shared.preflight_config(cfg, tmp_path / 'preflight')
    assert trial['training']['epochs'] == 2
    for key in ('num_envs_per_gpu', 'n_traj', 'ppo_update_epochs', 'num_minibatches', 'ppo_step_chunk_size'):
        assert trial['training'][key] == cfg['training'][key]
    assert trial['evaluation']['eval_batch_size'] == cfg['evaluation']['eval_batch_size'] == 16
    assert trial['evaluation']['eval_n_traj'] == cfg['evaluation']['eval_n_traj'] == 50
    assert trial['evaluation']['eval_limit'] == 16
    assert trial['offline']['exploration_interval'] == 1
    assert trial['offline']['exploration_instances'] == 4
    assert trial['offline']['policy_replay_max_new_routes'] == 16
    shared.scratch.assert_scratch(trial)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
def test_reference_seed_controls_train_sampling_and_validation(tmp_path, task):
    cfg = build(tmp_path, task)
    assert cfg['training']['post_init_seed'] == 3011
    assert cfg['evaluation']['eval_seed'] == 17003011


@pytest.mark.parametrize('seed', [3009, 3010, 3012])
def test_reference_guard_rejects_changed_training_seed(tmp_path, seed):
    with pytest.raises(ValueError, match='post_init_seed|eval_seed'):
        build(tmp_path, seed=seed)


def preflight_rows():
    return [dict(epoch=epoch, optimizer_steps_epoch=20,
                 amp_skipped_steps_epoch=0, grad_norm_nonfinite_count=0,
                 grad_norm=.75, gpu_peak_allocated_bytes=epoch * 1024,
                 gpu_peak_reserved_bytes=epoch * 2048,
                 parameter_sync=dict(parameter_sync_checksum_max_diff=0.,
                     optimizer_steps_rank_max_diff=0,
                     amp_skipped_steps_rank_max_diff=0, amp_scale_rank_max_diff=0.))
            for epoch in [1, 2]]


def write_preflight(tmp_path, rows_by_rank=None):
    output = tmp_path / 'preflight' / 'monitoring'
    output.mkdir(parents=True)
    for rank, rows in enumerate(rows_by_rank or [preflight_rows(), preflight_rows()]):
        (output / f'monitor_rank_{rank}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))
    return {'output_dir': str(output.parent)}


def test_preflight_accepts_two_healthy_epochs_per_rank_and_reports_memory(tmp_path):
    report = launch.validate_preflight(write_preflight(tmp_path), 2)
    assert set(report) == {'0', '1'}
    for rank in report.values():
        assert rank['epochs'] == 2 and rank['successful_updates'] == 40
        assert rank['amp_skips'] == rank['nonfinite_gradient_steps'] == 0
        assert rank['parameter_checksum_max_diff'] == 0
        assert rank['peak_allocated_bytes'] == 2048
        assert rank['peak_reserved_bytes'] == 4096


@pytest.mark.parametrize('field,value', [
    ('optimizer_steps_epoch', 19),
    ('optimizer_steps_epoch', None),
    ('amp_skipped_steps_epoch', 1),
    ('grad_norm_nonfinite_count', 1),
    ('grad_norm', float('nan')),
    ('grad_norm', float('inf')),
])
def test_preflight_rejects_unsuccessful_updates_skips_or_nonfinite_gradients(tmp_path, field, value):
    rows = [preflight_rows(), preflight_rows()]
    rows[1][1][field] = value
    spec = write_preflight(tmp_path, rows)
    with pytest.raises(ValueError, match='rank 1'):
        launch.validate_preflight(spec, 2)


@pytest.mark.parametrize('field', ['parameter_sync_checksum_max_diff',
    'optimizer_steps_rank_max_diff', 'amp_skipped_steps_rank_max_diff',
    'amp_scale_rank_max_diff'])
@pytest.mark.parametrize('missing', [False, True])
def test_preflight_rejects_missing_or_failed_sync_at_each_epoch(tmp_path, field, missing):
    rows = [preflight_rows(), preflight_rows()]
    if missing:
        del rows[0][0]['parameter_sync'][field]
    else:
        rows[0][0]['parameter_sync'][field] = .125
    spec = write_preflight(tmp_path, rows)
    with pytest.raises(ValueError, match='synchronization'):
        launch.validate_preflight(spec, 2)


@pytest.mark.parametrize('epochs', [[], [1], [2], [1, 1, 2], [2, 1], [1, 2, 3]])
def test_preflight_rejects_incomplete_duplicated_or_out_of_order_epochs(tmp_path, epochs):
    rows = [preflight_rows(), [dict(preflight_rows()[0], epoch=epoch) for epoch in epochs]]
    spec = write_preflight(tmp_path, rows)
    with pytest.raises(ValueError, match='exactly epochs 1 and 2'):
        launch.validate_preflight(spec, 2)


def test_preflight_requires_both_rank_monitor_files(tmp_path):
    spec = write_preflight(tmp_path, [preflight_rows()])
    with pytest.raises(FileNotFoundError):
        launch.validate_preflight(spec, 2)


@pytest.mark.parametrize('task', ['vrptw', 'evrptw'])
@pytest.mark.parametrize('encoder_variant', ['graph', 'current'])
def test_prepare_records_distinct_arm_and_provenance_builder_without_launching(
        scratch_prepared, monkeypatch, task, encoder_variant):
    state = scratch_prepared
    monkeypatch.setattr(shared, 'CODE_ROOT', state.root)
    data = state.root.parent / 'AAAI_Dataset'
    for split, count in [('train', 5000), ('val', 1000)]:
        folder = data / 'dataset' / task / split / 'Cus100'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'instances.pkl').write_bytes(b'CPU configuration fixture only')
        (folder / 'metadata.json').write_text(json.dumps(dict(
            num_instances=count, num_customers=100,
            num_charging_stations=20 if task == 'evrptw' else 0)))
        (folder / ('expert_solutions.csv' if split == 'train' else 'gurobi_summary.csv')).write_text(
            'instance_id\nfixture_0\n')
    args = shared.make_parser().parse_args([
        '--task', task, '--variant', 'optimized', '--encoder-variant', encoder_variant,
        '--seed', '3011', '--gpus', '0,1', '--prepare-only', '--data-root', str(data),
        '--batch-per-gpu', '20' if task == 'vrptw' else '16',
        '--chunk-size', '24', '--expert-chunk-size', '64',
        '--base-config', str(ROOT / 'configs/experiments/physics_exploration_vrptw100.yaml')])
    run = shared.prepare(args, config_builder=launch.build_config, arm_label=encoder_variant)
    manifest = json.loads((run / 'manifest.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert set(manifest['arms']) == set(status['arms']) == {encoder_variant}
    assert manifest['init_checkpoint'] is None and manifest['initialization_mode'] == 'scratch'
    assert manifest['hardware_at_prepare'] is None and not state.launches
    assert manifest['protocol']['require_preflight_health']
    assert manifest['protocol']['arm'] == encoder_variant
    assert manifest['protocol']['reference_training_commit'] == launch.TRAINING_COMMIT
    assert manifest['protocol']['global_policy_replay_max_new_routes'] == 32
    spec = manifest['arms'][encoder_variant]
    for stage in [spec, spec['preflight']]:
        assert '--nproc-per-node=2' in stage['command']
        assert 'offline2online.train' in stage['command']
        cfg = yaml.safe_load(Path(stage['config']).read_text())
        assert cfg['model']['use_joint_graph_encoder'] == (encoder_variant == 'graph')
        assert cfg['offline']['policy_replay_max_new_routes'] == 16
        assert cfg['experiment_protocol']['require_preflight_health']
        shared.scratch.assert_scratch(cfg)
    shared.shared.verify_manifest(manifest)


@pytest.mark.parametrize('encoder_variant', ['graph', 'current'])
def test_protocol_describes_actual_encoder_instead_of_inherited_template(tmp_path, encoder_variant):
    cfg = build(tmp_path, encoder_variant=encoder_variant)
    recorded = cfg['experiment_protocol']['model_integration']
    assert recorded['use_joint_graph_encoder'] == cfg['model']['use_joint_graph_encoder']
    assert recorded['use_edge_relation_encoder'] == cfg['model']['use_edge_relation_encoder']
    assert recorded['active_edge_state_dim'] == (32 if encoder_variant == 'graph' else 16)
