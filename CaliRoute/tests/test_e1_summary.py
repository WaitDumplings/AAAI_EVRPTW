"""Cross-server E1 summaries consume accepted tests, never select runs by test cost.

Fixtures are artifact-contract examples, not trained models or performance claims.
They exercise the real lifecycle acceptance helper using deterministic file hashes.
"""
import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from e1.configs import CONTROLLED, METHODS, SCHEMA, digest
from e1.summary import summarize_campaign


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _contract():
    return dict(schema=SCHEMA, training_seed=3009, online_loops=1500, global_batch=64,
                total_instance_exposures=96000, n_traj=50, controlled_ppo_passes=4,
                gamma=.99, eval_K=50, source_content_sha256='identical-source',
                expert_pool_semantic_sha256='identical-sorted-id-routes-km',
                dataset_sha256={split: split + '-instances-hash' for split in ('train', 'val', 'test')})


def _campaign(root, name, accepted=(), *, cost=100., failures=()):
    path = root / name
    path.mkdir()
    index = path / 'assets/instance_index.csv'
    index.parent.mkdir()
    index.write_text('instance_id,split\n' + ''.join(f'test_{i},test\n' for i in range(1000)))
    audit_path = path / 'assets/audit.json'
    _json(audit_path, dict(test_data_ready=True,
        splits={'test': {'files': {'instances.pkl': {'sha256': 'test-instances-hash'}}}},
        outputs={'instance_index.csv': {'sha256': digest(index)}}))
    manifest = dict(schema=SCHEMA, campaign=str(path), code_root=str(path / 'source'),
                    source={'content_sha256': 'identical-source'}, training_seed=3009,
                    audit_path=str(audit_path), comparison_contract=_contract(), methods={})
    for method in METHODS:
        run = path / 'runs' / method
        run.mkdir(parents=True)
        cfg = dict(data={'num_customers': 100}, run_name=method,
                   experiment_protocol={'training_seed': 3009})
        config_path = run / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        manifest['methods'][method] = dict(run_dir=str(run), config=str(config_path),
            config_sha256=digest(config_path), server='2080ti_a' if method in CONTROLLED else 'a6000')
        if method not in accepted:
            continue
        checkpoint = (path / 'source/results/checkpoints/Cus_100_CS_0' / method / 'seed_3009/checkpoint_best.pt'
                      if method in CONTROLLED else run / 'native_checkpoints/best.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'artifact-hash-fixture-not-a-trained-checkpoint')
        checkpoint_sha = digest(checkpoint)
        rows = []
        for index in range(1000):
            good = index not in failures
            rows.append(dict(method=method, instance_id=f'test_{index}', training_seed=3009,
                split='test', checkpoint_id=checkpoint_sha, requested_K=50, actual_K=50,
                feasible=good, completed=good, truncated=False,
                recomputed_cost_km=cost if good else None, num_routes=5 if good else None,
                runtime=.2, timing_batch_size=8, runtime_kind='batch_amortized',
                timing_scope='preprocess+encode+decode+validate', cost_mismatch_candidate_count=0))
        _write_result(run, rows, checkpoint_sha)
    _json(path / 'manifest.json', manifest)
    return path


def _write_result(run, rows, checkpoint_sha):
    folder = run / 'test'
    folder.mkdir(exist_ok=True)
    rows_path = folder / 'instances.jsonl'
    rows_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    _json(folder / 'manifest.json', dict(accepted=True, instances_sha256=digest(rows_path),
        test_data_sha256='test-instances-hash', checkpoint_sha256=checkpoint_sha))


def _mutate_rows(campaign, method, change):
    folder = campaign / 'runs' / method / 'test'
    rows = [json.loads(line) for line in (folder / 'instances.jsonl').read_text().splitlines()]
    change(rows)
    checkpoint_sha = json.loads((folder / 'manifest.json').read_text())['checkpoint_sha256']
    _write_result(folder.parent, rows, checkpoint_sha)


def _mutate_manifest(campaign, change):
    path = campaign / 'manifest.json'
    manifest = json.loads(path.read_text())
    change(manifest)
    _json(path, manifest)


def test_empty_campaign_has_seven_missing_methods_and_no_false_completion(tmp_path):
    campaign = _campaign(tmp_path, 'empty')
    result = summarize_campaign(campaign)
    assert not result['complete_seven_method_test'] and result['accepted_methods'] == []
    assert set(result['methods']) == set(METHODS)
    assert all(row['state'] == 'test_not_completed' for row in result['methods'].values())
    assert result['available_method_common_ids'] == 0
    with (campaign / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 and all(row['mean_distance_km'] == '' for row in rows)
    assert json.loads((campaign / 'summary.json').read_text()) == result


def test_merge_seven_methods_and_write_only_primary_campaign(tmp_path):
    paths = [
        _campaign(tmp_path, 'machine_a', ('ppo_base', 'ppo_rdi_agda')),
        _campaign(tmp_path, 'machine_b', ('awbc', 'dapg')),
        _campaign(tmp_path, 'machine_c', ('slppo',)),
        _campaign(tmp_path, 'machine_d', ('rrnco', 'radar')),
    ]
    result = summarize_campaign(paths[0], additional_campaigns=paths[1:])
    assert result['complete_seven_method_test']
    assert result['accepted_methods'] == list(METHODS)
    assert result['available_method_common_ids'] == result['available_method_common_feasible_ids'] == 1000
    assert result['common_feasible_mean_km'] == dict.fromkeys(METHODS, 100.)
    assert all(not (path / 'summary.json').exists() for path in paths[1:])
    with (paths[0] / 'summary.csv').open() as stream:
        rows = {row['method']: row for row in csv.DictReader(stream)}
    assert len(rows) == 7 and rows['radar']['campaign'] == str(paths[3])
    assert rows['radar']['server'] == 'a6000'
    assert rows['ppo_base']['timing_batch_sizes'] == '[8]'
    assert 'between-training-seed' in result['methods']['ppo_base']['statistical_scope']
    assert not any('std' in key for key in rows['radar'])


def test_path_sensitive_audits_do_not_block_semantically_identical_campaigns(tmp_path):
    first = _campaign(tmp_path, 'one', ('ppo_base',))
    second = _campaign(tmp_path, 'two', ('slppo',))
    _mutate_manifest(first, lambda m: m.update(created_at='yesterday', inputs=[{'sha256': 'audit1'}]))
    _mutate_manifest(second, lambda m: m.update(created_at='today', inputs=[{'sha256': 'audit2'}]))
    result = summarize_campaign(first, additional_campaigns=[second])
    assert result['accepted_methods'] == ['ppo_base', 'slppo']
    assert not result['complete_seven_method_test']


@pytest.mark.parametrize('field,value', [
    ('source_content_sha256', 'different-source'), ('training_seed', 3010),
    ('online_loops', 1600), ('total_instance_exposures', 102400), ('global_batch', 128),
    ('n_traj', 100), ('controlled_ppo_passes', 5), ('gamma', 1.), ('eval_K', 51),
    ('expert_pool_semantic_sha256', 'different-id-route-cost'),
    ('dataset_sha256', {'train': 'other', 'val': 'val-instances-hash', 'test': 'test-instances-hash'}),
])
def test_incompatible_source_data_seed_or_budget_rejected(tmp_path, field, value):
    first = _campaign(tmp_path, 'one', ('ppo_base',))
    second = _campaign(tmp_path, 'two', ('slppo',))
    def change(m):
        m['comparison_contract'][field] = value
        if field == 'source_content_sha256': m['source']['content_sha256'] = value
        if field == 'training_seed': m['training_seed'] = value
    _mutate_manifest(second, change)
    with pytest.raises(ValueError, match='Incompatible comparison_contract.*' + field):
        summarize_campaign(first, additional_campaigns=[second])
    assert not (first / 'summary.json').exists()


def test_duplicate_accepted_method_errors_without_selecting_cheaper_test(tmp_path):
    first = _campaign(tmp_path, 'one', ('slppo',), cost=100.)
    second = _campaign(tmp_path, 'two', ('slppo',), cost=1.)
    previous = summarize_campaign(first)
    with pytest.raises(ValueError, match='Duplicate accepted test results.*slppo'):
        summarize_campaign(first, additional_campaigns=[second])
    assert json.loads((first / 'summary.json').read_text()) == previous


def test_same_campaign_twice_is_not_a_second_run(tmp_path):
    first = _campaign(tmp_path, 'one')
    with pytest.raises(ValueError, match='Duplicate campaign path'):
        summarize_campaign(first, additional_campaigns=[first])


def test_rows_without_acceptance_manifest_are_not_a_completed_test(tmp_path):
    first = _campaign(tmp_path, 'one', ('slppo',))
    (first / 'runs/slppo/test/manifest.json').unlink()
    result = summarize_campaign(first)
    assert result['methods']['slppo']['state'] == 'test_unaccepted'
    assert result['accepted_methods'] == []
    assert 'mean_distance_km' not in result['methods']['slppo']


@pytest.mark.parametrize('mutation', [
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(instance_id='test_1'),
    lambda rows: rows[0].update(instance_id='outside-audited-test'),
    lambda rows: rows[0].update(actual_K=49),
    lambda rows: rows[0].update(method='radar'),
    lambda rows: rows[0].update(training_seed=3010),
    lambda rows: rows[0].update(checkpoint_id='unselected-checkpoint'),
    lambda rows: rows[0].update(feasible=False, recomputed_cost_km=0.),
    lambda rows: rows[0].update(truncated=True),
])
def test_invalid_or_partial_results_report_unaccepted_and_never_enter_means(tmp_path, mutation):
    first = _campaign(tmp_path, 'one', ('slppo',))
    _mutate_rows(first, 'slppo', mutation)
    result = summarize_campaign(first)
    assert result['methods']['slppo']['state'] == 'test_unaccepted'
    assert result['accepted_methods'] == [] and result['available_method_common_ids'] == 0


def test_changed_accepted_file_is_rejected_by_real_lifecycle_hash_check(tmp_path):
    first = _campaign(tmp_path, 'one', ('slppo',))
    path = first / 'runs/slppo/test/instances.jsonl'
    path.write_text(path.read_text() + '\n')
    result = summarize_campaign(first)
    assert 'changed' in result['methods']['slppo']['acceptance_error']


def test_one_accepted_and_one_unaccepted_attempt_uses_only_accepted(tmp_path):
    first = _campaign(tmp_path, 'one', ('slppo',), cost=100.)
    second = _campaign(tmp_path, 'two', ('slppo',), cost=1.)
    (second / 'runs/slppo/test/manifest.json').unlink()
    result = summarize_campaign(first, additional_campaigns=[second])
    row = result['methods']['slppo']
    assert row['mean_distance_km'] == 100.
    assert [a['state'] for a in row['campaign_attempts']] == ['test_completed', 'test_unaccepted']


def test_gurobi_gap_preserves_negatives_and_reports_both_denominators(tmp_path):
    first = _campaign(tmp_path, 'one', ('slppo',), failures=(2,))
    def change(rows):
        rows[0]['recomputed_cost_km'] = 90.
        rows[1]['recomputed_cost_km'] = 110.
    _mutate_rows(first, 'slppo', change)
    reference = tmp_path / 'gurobi.csv'
    reference.write_text('instance_id,objective_distance_km,feasible\n'
                         'test_0,100,True\ntest_1,100,True\ntest_2,100,True\n'
                         'test_3,100,False\ntest_4,nan,True\ntest_5,0,True\n')
    result = summarize_campaign(first, gurobi_path=reference)
    row = result['methods']['slppo']
    assert row['feasible_count'] == 999 and row['failure_count'] == 1
    assert row['gurobi_incumbent_available'] == 3
    assert row['model_feasible_and_gurobi_incumbent_count'] == 2
    assert row['negative_incumbent_gap_count'] == 1
    assert row['mean_gap_to_gurobi_incumbent_pct'] == pytest.approx(0., abs=1e-10)
    assert 'NOT a certified optimality gap' in row['gap_definition']


def test_gurobi_duplicate_id_rejected_even_when_first_cost_unusable(tmp_path):
    campaign = _campaign(tmp_path, 'one')
    reference = tmp_path / 'gurobi.csv'
    reference.write_text('instance_id,objective_distance_km\ntest_0,nan\ntest_0,100\n')
    with pytest.raises(ValueError, match='Duplicate Gurobi ID'):
        summarize_campaign(campaign, gurobi_path=reference)


def test_common_feasible_subset_is_instance_matched_and_retains_failures(tmp_path):
    first = _campaign(tmp_path, 'one', ('ppo_base',), failures=(0,), cost=100.)
    second = _campaign(tmp_path, 'two', ('slppo',), failures=(1,), cost=80.)
    result = summarize_campaign(first, additional_campaigns=[second])
    assert result['available_method_common_ids'] == 1000
    assert result['available_method_common_feasible_ids'] == 998
    assert result['common_feasible_mean_km'] == {'ppo_base': 100., 'slppo': 80.}
    assert result['methods']['slppo']['instances'] == 1000
    assert result['methods']['ppo_base']['failure_count'] == 1


def test_missing_comparison_contract_cannot_silently_merge_legacy_artifacts(tmp_path):
    campaign = _campaign(tmp_path, 'one')
    _mutate_manifest(campaign, lambda m: m.pop('comparison_contract'))
    with pytest.raises(ValueError, match='Missing complete comparison_contract'):
        summarize_campaign(campaign)


def test_copied_campaign_rebases_owned_paths_without_editing_frozen_files(tmp_path, monkeypatch):
    import shutil
    import e1.summary as summary_module
    first = _campaign(tmp_path, 'collector', ('ppo_base',))
    original = _campaign(tmp_path, 'remote_server', ('slppo', 'radar'))
    # Match the actual runtime layout: results is an absolute symlink to the
    # owned artifacts directory and becomes dangling after the server move.
    shutil.move(original / 'source/results', original / 'artifacts')
    (original / 'source/results').symlink_to(original / 'artifacts', target_is_directory=True)
    _mutate_manifest(original, lambda m: m.update(artifacts_root=str(original / 'artifacts')))
    frozen_manifest = (original / 'manifest.json').read_bytes()
    frozen_cfg = (original / 'runs/slppo/config.yaml').read_bytes()
    moved = tmp_path / 'collected_from_remote'
    shutil.move(original, moved)
    assert not (moved / 'source/results').exists()  # Stale old-machine symlink.
    real_accept = summary_module.assert_accepted_test_result
    rebased = []
    def inspect_rebased(manifest, method):
        if method in ('slppo', 'radar'):
            assert manifest['campaign'] == str(moved)
            assert manifest['code_root'] == str(moved / 'source')
            assert manifest['audit_path'] == str(moved / 'assets/audit.json')
            assert manifest['artifacts_root'] == str(moved / 'artifacts')
            assert manifest['methods'][method]['run_dir'] == str(moved / 'runs' / method)
            assert manifest['methods'][method]['config'] == str(moved / 'runs' / method / 'config.yaml')
            rebased.append(method)
        return real_accept(manifest, method)
    monkeypatch.setattr(summary_module, 'assert_accepted_test_result', inspect_rebased)
    result = summarize_campaign(first, additional_campaigns=[moved])
    assert result['accepted_methods'] == ['ppo_base', 'slppo', 'radar']
    assert rebased == ['slppo', 'radar']
    assert result['methods']['slppo']['instance_file'].startswith(str(moved))
    assert (moved / 'manifest.json').read_bytes() == frozen_manifest
    assert (moved / 'runs/slppo/config.yaml').read_bytes() == frozen_cfg
    assert not original.exists()


def test_moved_primary_campaign_writes_summary_to_actual_location(tmp_path):
    import shutil
    original = _campaign(tmp_path, 'remote', ('ppo_base',))
    moved = tmp_path / 'collector'
    shutil.move(original, moved)
    result = summarize_campaign(moved)
    assert result['accepted_methods'] == ['ppo_base']
    assert (moved / 'summary.json').exists() and (moved / 'summary.csv').exists()
    assert not original.exists()
