from argparse import Namespace
import importlib.util
import math
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/run_plugin_comparison.py'
spec = importlib.util.spec_from_file_location('plugin_comparison', path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_long_two_gpu_protocol_is_equal_across_models(tmp_path):
    args = Namespace(problem='cvrp', customers=50, data_root=tmp_path / 'dataset', init_checkpoint=tmp_path / 'ppo.pt',
                     epochs=1000, num_envs=64, n_traj=50, learning_rate=5e-5 * math.sqrt(2),
                     eval_interval=20, eval_batch_size=128, seed=3009, eval_limit=None, expert_limit=None,
                     num_minibatches=4, checkpoint_interval=50, lr_warmup_epochs=20, lr_min=1e-5, monitor_interval=20)
    cfg = launcher.build_long_configs(args, tmp_path / 'experiment')
    a, b = cfg['baseline'], cfg['optimized']
    assert a['training']['latest_checkpoint_interval'] == b['training']['latest_checkpoint_interval'] == 5
    assert a['data'] == b['data']
    assert a['offline']['init_checkpoint_path'] == b['offline']['init_checkpoint_path']
    for key in ('num_envs_per_gpu', 'n_traj', 'num_minibatches', 'ppo_update_epochs', 'learning_rate',
                'lr_schedule', 'lr_warmup_epochs', 'lr_min', 'entropy_initial_coef', 'entropy_final_coef', 'amp_init_scale'):
        assert a['training'][key] == b['training'][key]
    assert not a['model']['use_rdi_v2']
    assert b['model']['use_rdi_v2'] and b['model']['use_agda_v2']
    assert not b['model']['use_residual_edge_bias']
    assert b['advantage']['sl_advantage_scale_mode'] == 'relative'
    assert b['offline']['policy_replay_warmup_epochs'] == 25
    assert b['offline']['policy_replay_weight'] == .1


@pytest.mark.parametrize('left,right', [('0,1', '1,2'), ('0', '1,2'), ('0,x', '1,2')])
def test_gpu_assignments_cannot_overlap_or_silently_run_one_rank(left, right):
    with pytest.raises(ValueError):
        launcher.parse_gpu_pairs(left, right)


def test_failed_epoch_19_is_not_a_completed_1000_epoch_run():
    manifest = {'protocol': {'epochs': 1000}, 'arms': {'baseline': {}, 'optimized': {}}}
    rows = {arm: [{'epoch': str(epoch)} for epoch in range(1, 20)] for arm in manifest['arms']}
    evaluations = {arm: [{'epoch': '0', 'eval_status': 'ok'}] for arm in manifest['arms']}
    status = {'exit_code': 1, 'arms': {arm: {'exit_code': 1} for arm in manifest['arms']}}
    progress = launcher.training_progress(manifest, status, rows, evaluations)
    assert progress['target_epochs'] == 1000
    assert progress['run_state'] == 'failed'
    for arm in progress['arms'].values():
        assert arm['completed_training_epochs'] == 19
        assert arm['latest_training_epoch'] == 19
        assert arm['latest_validation_epoch'] == 0
        assert arm['remaining_epochs'] == 981
        assert arm['process_state'] == 'failed'


def test_success_requires_every_target_epoch_and_both_clean_exits():
    manifest = {'protocol': {'epochs': 3}, 'arms': {'baseline': {}, 'optimized': {}}}
    rows = {'baseline': [{'epoch': epoch} for epoch in (1, 2, 3)],
            'optimized': [{'epoch': epoch} for epoch in (1, 3, 3)]}
    status = {'exit_code': 0, 'arms': {arm: {'exit_code': 0} for arm in manifest['arms']}}
    assert launcher.training_progress(manifest, status, rows, {})['run_state'] == 'failed'
    rows['optimized'] = rows['baseline']
    assert launcher.training_progress(manifest, status, rows, {})['run_state'] == 'completed'
    running = {'arms': {arm: {'pid': 1234} for arm in manifest['arms']}}
    assert launcher.training_progress(manifest, running, rows, {})['run_state'] == 'running'
    status['interrupted_signal'] = 15
    assert launcher.training_progress(manifest, status, rows, {})['run_state'] == 'interrupted'


def test_session_clock_reset_accumulates_only_retained_recorded_work():
    rows = [
        {'epoch': '1', 'run_session_id': 'original', 'run_elapsed_seconds': '100'},
        {'epoch': '20', 'run_session_id': 'original', 'run_elapsed_seconds': '1000'},
        {'epoch': '21', 'run_session_id': 'resumed', 'run_elapsed_seconds': '80'},
        {'epoch': '22', 'run_session_id': 'resumed', 'run_elapsed_seconds': '130'},
        {'epoch': '23', 'run_session_id': 'resumed_again', 'run_elapsed_seconds': '90'},
    ]
    recorded = launcher.recorded_session_times(rows)
    assert [row['recorded_active_session_seconds'] for row in recorded] == [100, 1000, 1080, 1130, 1220]
    for original, updated in zip(rows, recorded):
        assert updated['run_elapsed_seconds'] == original['run_elapsed_seconds']
        assert updated['run_session_id'] == original['run_session_id']
        assert 'recorded_active_session_seconds' not in original


def test_report_exposes_progress_lineage_and_session_time(tmp_path):
    import csv
    import json

    manifest = {'comparison': 'test', 'git_commit': 'abc',
                'protocol': {'epochs': 3, 'lr_warmup_epochs': 20}, 'arms': {}}
    rows = [dict(epoch=epoch, eval_status='ok', epoch_wall_time_s=10, eval_wall_time_s=1,
                 run_elapsed_seconds=elapsed, run_session_id=session,
                 eval_feasible_rate=1, eval_avg_min_objective_distance_km=100 - epoch)
            for epoch, elapsed, session in [(1, 100, 'first'), (2, 200, 'first'), (3, 30, 'resumed')]]
    for arm in ('baseline', 'optimized'):
        folder = tmp_path / arm
        folder.mkdir()
        manifest['arms'][arm] = {'log_dir': str(folder), 'resume_checkpoint': 'epoch_2.pt',
                                 'source_experiment': 'original'}
        for filename in ('train_log.csv', 'eval_log.csv'):
            with (folder / filename).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    status = {'exit_code': 0, 'arms': {arm: {'exit_code': 0} for arm in manifest['arms']}}
    launcher.update_long_report(tmp_path, manifest, status)
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert next(iter(report)) == 'progress'
    assert report['progress']['run_state'] == 'completed'
    assert report['arms']['baseline']['completed_training_epochs'] == 3
    assert report['arms']['optimized']['lineage']['resume_checkpoint'] == 'epoch_2.pt'
    assert [row['recorded_active_session_seconds'] for row in report['validation_vs_wall_time']['optimized']] == [100, 200, 230]
    assert 'excludes downtime' in report['validation_wall_time_scope']
