"""Evidence exports verify immutable inputs while snapshotting append-only logs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'export_run_evidence.py'
SPEC = importlib.util.spec_from_file_location('export_run_evidence', SCRIPT)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def save_manifest(run, manifest):
    (run / 'manifest.json').write_text(json.dumps(manifest))


def fixture_run(tmp_path, name='run'):
    run = tmp_path / name
    root = run / 'source' / 'CaliRoute'
    source = {
        'caliroute/model.py': b'def forward(x):\n    return x\n',
        'configs/default.yaml': b'gamma: 1.0\n',
        'README.md': b'# Source version\n',
        '.gitignore': b'results/\n',
        'assets/initial.pt': b'WEIGHT_BYTES_MUST_NOT_BE_EXPORTED',
        'data/raw.json': b'{"secret": "DATA_BYTES_MUST_NOT_BE_EXPORTED"}',
    }
    for relative, data in source.items():
        put(root / relative, data)
    files = {key: sha(value) for key, value in source.items()}
    source_spec = dict(files=files, content_sha256=sha(json.dumps(files, sort_keys=True).encode()), git_commit='abc123')
    dataset = put(tmp_path / f'{name}_dataset' / 'instances.pkl', b'DATASET_BYTES_MUST_NOT_BE_EXPORTED')
    config = put(run / 'graph' / 'config.yaml', b'run_name: scratch\ntraining:\n  epochs: 1500\n')
    preflight = put(run / 'graph' / 'preflight' / 'config.yaml', b'run_name: preflight\ntraining:\n  epochs: 2\n')
    logs = tmp_path / f'{name}_logs'
    put(logs / 'train_log.csv', b'epoch,train_kl\n1,0.001\n')
    put(logs / 'eval_log.csv', b'epoch,distance\n0,350\n')
    put(logs / 'train_log_rank_1.csv', b'RANK1_MUST_NOT_BE_EXPORTED')
    manifest = dict(
        initialization_mode='scratch', init_checkpoint=None, code_root=str(root), source=source_spec,
        inputs={'task/train/instances.pkl': {'path': str(dataset), 'sha256': sha(dataset.read_bytes())}},
        arms={'graph': dict(config=str(config), config_sha256=sha(config.read_bytes()),
                            code_root=str(root), log_dir=str(logs), checkpoint_dir=str(run/'checkpoints'),
                            preflight=dict(config=str(preflight), config_sha256=sha(preflight.read_bytes()),
                                           code_root=str(root)))},
        additional_sources={})
    save_manifest(run, manifest)
    put(run / 'status.json', b'{"state":"running","epoch":12}')
    put(run / 'comparison.json', b'{"arms":{"graph":{"epoch":12}}}')
    return run, manifest


def report_at(package):
    index = json.loads(package.read_text())
    report = json.loads((package.parent / index['runs'][0]['verification']).read_text())
    return index, report


def assert_inventory(package):
    index = json.loads(package.read_text())
    actual = {str(path.relative_to(package.parent)) for path in package.parent.rglob('*') if path.is_file()}
    assert actual == set(index['files']) | {'evidence_manifest.json'}
    for relative, entry in index['files'].items():
        data = (package.parent / relative).read_bytes()
        assert sha(data) == entry['sha256']
        assert len(data) == entry['bytes']


def test_default_export_checks_every_frozen_file_without_exporting_source_data_or_weights(tmp_path):
    run, manifest = fixture_run(tmp_path)
    package = evidence.export_run_evidence([run], tmp_path / 'export')
    index, report = report_at(package)
    assert_inventory(package)
    assert report['verified']
    assert len(report['sources']['main']['files']) == len(manifest['source']['files'])
    assert all(item['verified'] for item in report['sources']['main']['files'].values())
    assert report['sources']['main']['exported_files'] == []
    assert report['inputs']['task/train/instances.pkl']['verified']
    assert report['inputs']['task/train/instances.pkl']['exported'] is False
    assert set(report['configs']) == {'graph', 'graph/preflight'}
    assert report['live_snapshots']['status.json']['snapshot_only']
    assert report['logs']['graph/train_log.csv']['present']
    assert not index['policy']['datasets_exported']
    assert not index['policy']['checkpoints_exported']
    assert not index['policy']['live_snapshots_are_cross_file_atomic']
    contents = b''.join(path.read_bytes() for path in package.parent.rglob('*') if path.is_file())
    for secret in (b'DATASET_BYTES_MUST_NOT_BE_EXPORTED', b'WEIGHT_BYTES_MUST_NOT_BE_EXPORTED',
                   b'DATA_BYTES_MUST_NOT_BE_EXPORTED', b'RANK1_MUST_NOT_BE_EXPORTED'):
        assert secret not in contents
    assert not any('/source/' in name for name in index['files'])


def test_source_export_is_verified_allowlisted_subset(tmp_path):
    run, _ = fixture_run(tmp_path)
    package = evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    index, report = report_at(package)
    exported = set(report['sources']['main']['exported_files'])
    assert exported == {'caliroute/model.py', 'configs/default.yaml', 'README.md'}
    assert set(report['sources']['main']['excluded_from_source_export']) == {'.gitignore', 'assets/initial.pt', 'data/raw.json'}
    assert_inventory(package)
    assert len([item for item in index['files'].values() if item['kind'] == 'verified_source']) == 3


def test_two_runs_same_basename_remain_separate(tmp_path):
    first, _ = fixture_run(tmp_path/'first')
    second, _ = fixture_run(tmp_path/'second')
    package = evidence.export_run_evidence([first, second], tmp_path/'both')
    index = json.loads(package.read_text())
    assert [item['directory'] for item in index['runs']] == ['runs/01_run', 'runs/02_run']
    assert_inventory(package)


@pytest.mark.parametrize('kind', ['source', 'input', 'config', 'preflight'])
def test_hash_mismatch_fails_and_removes_only_the_new_export(tmp_path, kind):
    run, manifest = fixture_run(tmp_path)
    path = {
        'source': Path(manifest['code_root'])/'caliroute/model.py',
        'input': Path(manifest['inputs']['task/train/instances.pkl']['path']),
        'config': Path(manifest['arms']['graph']['config']),
        'preflight': Path(manifest['arms']['graph']['preflight']['config']),
    }[kind]
    path.write_bytes(b'CHANGED AFTER MANIFEST')
    with pytest.raises(evidence.EvidenceError, match='SHA256 mismatch'):
        evidence.export_run_evidence([run], tmp_path/'failed')
    assert not (tmp_path/'failed').exists()
    assert (run/'manifest.json').exists()


def test_existing_directory_is_untouched(tmp_path):
    run, _ = fixture_run(tmp_path)
    output = tmp_path/'existing'
    sentinel = put(output/'keep.txt', b'keep')
    with pytest.raises(FileExistsError):
        evidence.export_run_evidence([run], output)
    assert sentinel.read_bytes() == b'keep'


def test_dangling_output_symlink_is_not_overwritten(tmp_path):
    run, _ = fixture_run(tmp_path)
    output = tmp_path/'alias'
    output.symlink_to(tmp_path/'missing')
    with pytest.raises(FileExistsError):
        evidence.export_run_evidence([run], output)
    assert output.is_symlink()


@pytest.mark.parametrize('relative', ['../outside.py', '/outside.py', 'safe/../../outside.py', 'safe\\outside.py'])
def test_manifest_path_traversal_is_rejected(tmp_path, relative):
    run, manifest = fixture_run(tmp_path)
    manifest['source']['files'] = {relative: sha(b'no')}
    manifest['source'].pop('content_sha256')
    save_manifest(run, manifest)
    with pytest.raises(evidence.EvidenceError, match='Unsafe source-relative path'):
        evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    assert not (tmp_path/'export').exists()


@pytest.mark.parametrize('include_source', [False, True])
def test_source_symlink_cannot_escape_declared_frozen_root(tmp_path, include_source):
    run, manifest = fixture_run(tmp_path)
    root = Path(manifest['code_root'])
    external = put(tmp_path/'outside.py', (root/'caliroute/model.py').read_bytes())
    (root/'caliroute/model.py').unlink()
    (root/'caliroute/model.py').symlink_to(external)
    with pytest.raises(evidence.EvidenceError, match='escapes its declared root'):
        evidence.export_run_evidence([run], tmp_path/'export', include_source=include_source)


def test_internal_symlink_cannot_smuggle_excluded_data(tmp_path):
    run, manifest = fixture_run(tmp_path)
    root = Path(manifest['code_root'])
    (root/'config_alias.json').symlink_to(root/'data/raw.json')
    manifest['source']['files']['config_alias.json'] = sha((root/'data/raw.json').read_bytes())
    manifest['source'].pop('content_sha256')
    save_manifest(run, manifest)
    package = evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    _, report = report_at(package)
    assert 'config_alias.json' in report['sources']['main']['excluded_from_source_export']


def test_input_json_in_source_tree_is_not_copied(tmp_path):
    run, manifest = fixture_run(tmp_path)
    root = Path(manifest['code_root'])
    input_path = put(root/'examples'/'records.json', b'{"private_records":42}')
    manifest['source']['files']['examples/records.json'] = sha(input_path.read_bytes())
    manifest['source'].pop('content_sha256')
    manifest['inputs']['private_json'] = {'path':str(input_path), 'sha256':sha(input_path.read_bytes())}
    save_manifest(run, manifest)
    package = evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    _, report = report_at(package)
    assert 'examples/records.json' not in report['sources']['main']['exported_files']


def test_additional_historical_source_is_verified_and_exported(tmp_path):
    run, manifest = fixture_run(tmp_path)
    old = run/'historical'
    source = put(old/'old.py', b'print("historical")\n')
    manifest['additional_sources']['original'] = {'code_root':str(old), 'source':{'files':{'old.py':sha(source.read_bytes())}}}
    manifest['arms']['graph']['preflight']['code_root'] = str(old)
    save_manifest(run, manifest)
    package = evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    _, report = report_at(package)
    assert report['sources']['additional_original']['files']['old.py']['verified']
    assert report['sources']['additional_original']['exported_files'] == ['old.py']


def test_unverified_execution_root_is_rejected(tmp_path):
    run, manifest = fixture_run(tmp_path)
    alternate = tmp_path/'other_code'
    alternate.mkdir()
    manifest['arms']['graph']['code_root'] = str(alternate)
    save_manifest(run, manifest)
    with pytest.raises(evidence.EvidenceError, match='Unverified execution source'):
        evidence.export_run_evidence([run], tmp_path/'export')


def test_config_cannot_copy_arbitrary_outside_file(tmp_path):
    run, manifest = fixture_run(tmp_path)
    external = put(tmp_path/'outside.yaml', b'secret: 1\n')
    manifest['arms']['graph'].update(config=str(external), config_sha256=sha(external.read_bytes()))
    save_manifest(run, manifest)
    with pytest.raises(evidence.EvidenceError, match='configuration escapes'):
        evidence.export_run_evidence([run], tmp_path/'export')


def test_running_partial_status_and_csv_are_preserved_as_once_read_snapshots(tmp_path):
    run, manifest = fixture_run(tmp_path)
    (run/'status.json').write_bytes(b'{"state":')
    log = Path(manifest['arms']['graph']['log_dir'])/'train_log.csv'
    log.write_bytes(b'epoch,loss\n1,0.')
    package = evidence.export_run_evidence([run], tmp_path/'export')
    index, report = report_at(package)
    assert report['live_snapshots']['status.json']['valid_json'] is False
    assert report['logs']['graph/train_log.csv']['possibly_partial_final_line'] is True
    assert (package.parent/index['runs'][0]['directory']/'status.json').read_bytes() == b'{"state":'
    assert_inventory(package)


def test_missing_live_files_and_logs_do_not_invalidate_prepared_run(tmp_path):
    run, manifest = fixture_run(tmp_path)
    (run/'status.json').unlink()
    (run/'comparison.json').unlink()
    for name in evidence.LOG_NAMES:
        (Path(manifest['arms']['graph']['log_dir'])/name).unlink()
    package = evidence.export_run_evidence([run], tmp_path/'export')
    _, report = report_at(package)
    assert report['verified']
    assert report['live_snapshots']['status.json']['present'] is False
    assert report['logs']['graph/train_log.csv']['present'] is False


def test_recorded_source_mapping_hash_is_also_checked(tmp_path):
    run, manifest = fixture_run(tmp_path)
    manifest['source']['content_sha256'] = '0'*64
    save_manifest(run, manifest)
    with pytest.raises(evidence.EvidenceError, match='Source mapping SHA256 mismatch'):
        evidence.export_run_evidence([run], tmp_path/'export')


def test_legacy_files_mapping_without_aggregate_remains_supported(tmp_path):
    run, manifest = fixture_run(tmp_path)
    manifest['source'].pop('content_sha256')
    save_manifest(run, manifest)
    package = evidence.export_run_evidence([run], tmp_path/'export')
    assert report_at(package)[1]['verified']


def test_binary_disguised_as_source_is_never_exported(tmp_path):
    run, manifest = fixture_run(tmp_path)
    root = Path(manifest['code_root'])
    binary = put(root/'bad.py', b'\x00\x01binary')
    manifest['source']['files']['bad.py'] = sha(binary.read_bytes())
    manifest['source'].pop('content_sha256')
    save_manifest(run, manifest)
    with pytest.raises(evidence.EvidenceError, match='binary NUL'):
        evidence.export_run_evidence([run], tmp_path/'export', include_source=True)


def test_cli_uses_only_standard_library_and_accepts_multiple_runs(tmp_path):
    first, _ = fixture_run(tmp_path, 'first')
    second, _ = fixture_run(tmp_path, 'second')
    output = tmp_path/'cli'
    result = subprocess.run([sys.executable, '-S', str(SCRIPT), str(first), str(second),
                             '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Export complete:' in result.stdout
    assert len(json.loads((output/'evidence_manifest.json').read_text())['runs']) == 2


def test_input_hardlink_cannot_be_exported_as_source_json(tmp_path):
    run, manifest = fixture_run(tmp_path)
    root = Path(manifest['code_root'])
    input_path = Path(manifest['inputs']['task/train/instances.pkl']['path'])
    alias = root/'renamed_records.json'
    alias.hardlink_to(input_path)
    manifest['source']['files']['renamed_records.json'] = sha(alias.read_bytes())
    manifest['source'].pop('content_sha256')
    save_manifest(run, manifest)
    package = evidence.export_run_evidence([run], tmp_path/'export', include_source=True)
    _, report = report_at(package)
    assert 'renamed_records.json' in report['sources']['main']['excluded_from_source_export']
