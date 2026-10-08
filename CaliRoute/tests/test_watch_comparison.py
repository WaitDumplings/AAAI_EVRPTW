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
