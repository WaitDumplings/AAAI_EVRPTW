"""The JSON tracker stays read-only and distinguishes missing fresh KL from zero."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_comparison as watch


def snapshot():
    return {'state': 'running', 'protocol': {'epochs': 300, 'ppo_update_epochs': 5},
        'arms': {'legacy': {'state': 'running', 'gpu': 0, 'completed_training_epochs': 11,
            'latest_train_row': {'epoch': '11', 'approx_kl': '.006', 'learning_rate': '1e-5'}},
                 'physics': {'state': 'running', 'gpu': 1, 'completed_training_epochs': 10,
            'latest_train_row': {'epoch': '10', 'approx_kl': '.007', 'learning_rate': '1e-5'}}}}


def render(tmp_path, capsys, data):
    path = tmp_path/'comparison.json'
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    watch.render(path)
    output = capsys.readouterr().out
    assert path.read_bytes() == before
    return output


def diagnostic_rows(output):
    section = output.split('PPO update diagnostics (latest training row)\n', 1)[1]
    rows = {}
    for line in section.splitlines()[2:]:
        fields = line.split()
        if fields and fields[0] in ('legacy', 'physics'):
            rows[fields[0]] = fields[1:]
    return rows


def test_legacy_json_keeps_old_table_and_footer_without_new_diagnostics(tmp_path, capsys):
    output = render(tmp_path, capsys, snapshot())
    assert 'Train KL' in output and '0.00600' in output and '0.00700' in output
    assert 'PPO update diagnostics' not in output and 'Fresh KL' not in output
    assert 'KL is the logged training aggregate.' in output
    assert 'No common complete validation is recorded' in output


def test_new_fields_add_small_table_without_changing_main_table(tmp_path, capsys):
    original = snapshot()
    before = render(tmp_path, capsys, original)
    changed = copy.deepcopy(original)
    changed['arms']['physics']['latest_train_row'].update(
        ppo_passes_executed='3', post_update_kl='.018', post_update_kl_time_s='1.23456')
    output = render(tmp_path, capsys, changed)
    assert output.split('\nPPO update diagnostics')[0].rstrip() == before.split('\nNo common complete validation')[0].rstrip()
    rows = diagnostic_rows(output)
    assert rows['physics'] == ['10', '3', '0.01800', '1.235']
    assert rows['legacy'] == ['11', '--', '--', '--']
    assert 'updated policy on sampled rollout actions' in output
    assert 'Train KL is the update aggregate' in output


@pytest.mark.parametrize('missing', [None, '', 'nan', 'inf'])
def test_unsampled_epochs_do_not_report_zero_fresh_kl_or_zero_monitor_cost(tmp_path, capsys, missing):
    data = snapshot()
    data['arms']['legacy']['latest_train_row'].update(
        ppo_passes_executed=5, post_update_kl=missing, post_update_kl_time_s=0.)
    rows = diagnostic_rows(render(tmp_path, capsys, data))
    assert rows['legacy'] == ['11', '5', '--', '--']
    assert rows['physics'] == ['10', '--', '--', '--']


def test_real_zero_kl_is_shown_and_missing_time_is_not_fabricated(tmp_path, capsys):
    data = snapshot()
    data['arms']['physics']['latest_train_row']['post_update_kl'] = 0.
    rows = diagnostic_rows(render(tmp_path, capsys, data))
    assert rows['physics'] == ['10', '--', '0.00000', '--']


def test_preflight_diagnostics_never_appear_as_formal_training(tmp_path, capsys):
    data = snapshot()
    data['arms']['physics']['stage'] = 'preflight'
    data['arms']['physics']['preflight_progress'] = {'latest_train_row': {
        'ppo_passes_executed': 2, 'post_update_kl': .9}}
    output = render(tmp_path, capsys, data)
    assert 'running/preflight' in output
    assert 'PPO update diagnostics' not in output


def test_separate_dual_runs_only_show_common_completed_epoch(tmp_path, capsys):
    protocol = dict(task='vrptw100', seed=3011, epochs=1500, global_batch=64,
        n_traj=50, ppo_update_epochs=5, eval_interval=50, eval_n_traj=50, validation_instances=1000)
    row = {watch.DISTANCE: 230, 'eval_feasible_rate': 1, 'eval_num_instances': 1000}
    first = dict(protocol=protocol, arms={'original': {}}, matched_validation_epochs={
        '50': {'original': row}, '100': {'original': row}})
    second = dict(protocol=protocol, arms={'optimized': {}}, matched_validation_epochs={
        '50': {'optimized': {**row, watch.DISTANCE: 225}}})
    output = render(tmp_path, capsys, watch.combined_snapshot(first, second))
    assert 'Latest common complete validation: epoch 50' in output
    assert '230.00' in output and '225.00' in output
    assert 'epoch 100 (km' not in output
    second['protocol'] = {**protocol, 'task': 'evrptw100'}
    with pytest.raises(ValueError, match='protocol task'):
        watch.combined_snapshot(first, second)
