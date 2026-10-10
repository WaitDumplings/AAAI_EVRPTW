"""Local E1 artifact ownership, run locks and final-test acceptance contracts.

These functions never start workers, alter checkpoints, read Gurobi outputs, or
request a GPU. Checkpoints are trusted local files, loaded explicitly on CPU.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import socket

import yaml

from e1.configs import CONTROLLED, EXPERT_METHODS, SCHEMA, content_hash, digest


def _read(path):
    return json.loads(Path(path).read_text())


def _config(manifest, method):
    if manifest.get('schema') != SCHEMA or method not in manifest['methods']:
        raise ValueError('Unknown E1 campaign/method')
    spec = manifest['methods'][method]
    if digest(spec['config']) != spec['config_sha256']:
        raise ValueError('Frozen method configuration changed')
    return spec, yaml.safe_load(Path(spec['config']).read_text())


def _checkpoint_dir(manifest, cfg):
    return (Path(manifest.get('artifacts_root', str(Path(manifest['code_root']) / 'results'))) / 'checkpoints'
            / f'Cus_{cfg["data"]["num_customers"]}_CS_0' / cfg['run_name']
            / f'seed_{cfg["experiment_protocol"]["training_seed"]}')


def _selected_checkpoint(manifest, method, spec, cfg):
    if method in CONTROLLED:
        return _checkpoint_dir(manifest, cfg) / 'checkpoint_best.pt'
    return Path(spec['run_dir']) / 'native_checkpoints/best.pt'


def assert_can_start(manifest, method):
    """Refuse scratch writes over formal artifacts; preflight/smoke are isolated."""
    spec, cfg = _config(manifest, method)
    run = Path(spec['run_dir'])
    artifacts = [run / name for name in ('training_result.json', 'train.log', 'monitoring',
        'validation', 'native_manifest.json', 'native_resolved.json', 'native_status.json',
        'native_source', 'native_harness', 'native_data', 'native_checkpoints',
        'native_train.jsonl', 'native_validation.jsonl', 'native_evaluations', 'test')]
    if method in CONTROLLED:
        checkpoint = _checkpoint_dir(manifest, cfg)
        artifacts.append(checkpoint)
        artifacts.append(Path(manifest['code_root']) / 'results/logs'
            / f'Cus_{cfg["data"]["num_customers"]}_CS_0' / cfg['run_name']
            / f'seed_{cfg["experiment_protocol"]["training_seed"]}')
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError('Formal E1 artifacts already exist; use explicit resume: ' + ', '.join(existing))


@contextmanager
def acquire_run_lock(run_dir):
    """Hold on worker rank zero for the full lifecycle, independently of GPUs."""
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / '.e1_run.lock').open('a+')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'E1 run is already owned by another process: {folder}') from error
        handle.seek(0); handle.truncate()
        json.dump(dict(pid=os.getpid(), hostname=socket.gethostname()), handle)
        handle.flush()
        yield handle
    finally:
        # Never unlink a lock file: doing so can allow a new inode to bypass
        # another process holding the same path's old inode.
        handle.close()


def _check_protocol(saved, expected, manifest, method):
    identity = dict(schema=SCHEMA, method=method, training_seed=manifest['training_seed'],
        data_audit_sha256=expected.get('data_audit_sha256'),
        expert_pool_sha256=expected.get('expert_pool_sha256'),
        source_content_sha256=manifest['source']['content_sha256'])
    if not identity['data_audit_sha256']:
        raise ValueError('Campaign is missing its audited data identity')
    if method in EXPERT_METHODS and not identity['expert_pool_sha256']:
        raise ValueError('Expert method is missing its audited train expert identity')
    for key, value in identity.items():
        if saved.get(key) != value or expected.get(key) != value:
            raise ValueError(f'Checkpoint E1 identity mismatch: {key}')
    if saved.get('smoke_only') or saved.get('hardware_preflight_only'):
        raise ValueError('Smoke/preflight checkpoint cannot initialize a formal E1 run')


def assert_checkpoint_owner(manifest, method, checkpoint):
    """Validate campaign identity even for identical copies in backup directories.

    Returns the loaded checkpoint so callers can inspect actual progress without
    loading weights twice. It does not restore a model or optimizer.
    """
    import torch
    spec, cfg = _config(manifest, method)
    path = Path(checkpoint)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must be a state dictionary with provenance')
    if method in CONTROLLED:
        saved_cfg = payload.get('config', {})
        if saved_cfg.get('run_name') != cfg['run_name']:
            raise ValueError('Checkpoint run_name does not belong to this formal run')
        if payload.get('seed') != manifest['training_seed']:
            raise ValueError('Checkpoint training seed changed')
        _check_protocol(saved_cfg.get('experiment_protocol', {}), cfg['experiment_protocol'], manifest, method)
    else:
        run = Path(spec['run_dir'])
        resolved = _read(run / 'native_resolved.json')
        native = _read(run / 'native_manifest.json')
        if content_hash(resolved) != native['config_sha256']:
            raise ValueError('Native resolved configuration changed')
        if resolved.get('run_id') != cfg['run_id'] or resolved.get('training_seed') != manifest['training_seed']:
            raise ValueError('Native checkpoint run_id/training seed does not belong to this run')
        _check_protocol(resolved.get('experiment_protocol', {}), cfg['experiment_protocol'], manifest, method)
        if payload.get('schema') != 'aaai_e1_native_checkpoint_v1' or payload.get('method') != method:
            raise ValueError('Native checkpoint method/schema mismatch')
        expected = dict(config_sha256=native['config_sha256'], source_sha256=native['source']['sha256'],
            harness_sha256=native['harness']['sha256'],
            dataset_hashes={key: value['npz_sha256'] for key, value in native['datasets'].items()})
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f'Native checkpoint identity mismatch: {key}')
    return payload


def assert_test_ready(manifest, method, checkpoint):
    """Require completed formal budget and frozen validation-selected weights."""
    spec, cfg = _config(manifest, method)
    run = Path(spec['run_dir'])
    selected = _selected_checkpoint(manifest, method, spec, cfg)
    selected_sha = digest(selected)
    if digest(checkpoint) != selected_sha:
        raise ValueError('Final test requires the validation-selected best checkpoint or its identical backup')
    best = assert_checkpoint_owner(manifest, method, checkpoint)
    if method in CONTROLLED:
        result = _read(run / 'training_result.json')
        if result.get('state') != 'completed':
            raise ValueError('Formal training must finish before final test')
        final_path = _checkpoint_dir(manifest, cfg) / 'checkpoint_final.pt'
        if digest(result['last_checkpoint']) != digest(final_path):
            raise ValueError('Training completion points to a different final checkpoint')
        final = assert_checkpoint_owner(manifest, method, final_path)
        total_epochs = cfg['training']['epochs']
        state = final.get('training_resume_state', {})
        if final.get('epoch') != total_epochs or state.get('completed_epoch') != total_epochs:
            raise ValueError('Actual formal checkpoint epoch has not completed its budget')
        if state.get('evaluation_pending', True):
            raise ValueError('Final validation is still pending')
        budget = state.get('experiment_budget', {})
        exposures = cfg['experiment_protocol']['total_instance_exposures']
        attempts = exposures * cfg['training']['n_traj']
        if budget.get('instance_exposures') != exposures or budget.get('sampled_trajectories') != attempts:
            raise ValueError('Actual online instance/trajectory budget has not completed')
        if state.get('best_validation_selection', {}).get('epoch') != best.get('epoch'):
            raise ValueError('Best checkpoint disagrees with the final validation selection')
    else:
        result = _read(run / 'native_status.json')
        if result.get('state') != 'completed' or result.get('training_complete') is not True:
            raise ValueError('Native formal training must finish before final test')
        final_path = run / 'native_checkpoints/last.pt'
        final = assert_checkpoint_owner(manifest, method, final_path)
        exposures = cfg['instance_exposures']
        if (result.get('total_instance_exposures') != exposures
                or result.get('counters', {}).get('instance_exposures') != exposures
                or final.get('counters', {}).get('instance_exposures') != exposures):
            raise ValueError('Native actual instance exposure budget has not completed')
        if result.get('checkpoint', {}).get('sha256') != digest(final_path):
            raise ValueError('Native completion checkpoint hash changed')
        if (final.get('best_metric') is None or best.get('best_metric') != final.get('best_metric')
                or result.get('best_metric') != final.get('best_metric')):
            raise ValueError('Native best checkpoint disagrees with final validation selection')
    return dict(checkpoint_sha256=selected_sha, final_checkpoint_sha256=digest(final_path),
                formal_training_complete=True, method=method)


def assert_accepted_test_result(manifest, method):
    """A rows file alone is not an accepted test; verify its completion marker."""
    spec, cfg = _config(manifest, method)
    run = Path(spec['run_dir']); folder = run / 'test'
    record = _read(folder / 'manifest.json')
    if record.get('accepted') is not True:
        raise ValueError('Test result has no successful acceptance marker')
    rows_path = folder / 'instances.jsonl'
    if record.get('instances_sha256') != digest(rows_path):
        raise ValueError('Accepted test instance output changed')
    audit = _read(manifest['audit_path'])
    expected_test_sha = audit['splits']['test']['files']['instances.pkl']['sha256']
    if not audit.get('test_data_ready') or record.get('test_data_sha256') != expected_test_sha:
        raise ValueError('Test result has a different or unverified data identity')
    index = Path(manifest['campaign']) / 'assets/instance_index.csv'
    expected_index_sha = audit['outputs']['instance_index.csv']['sha256']
    if digest(index) != expected_index_sha:
        raise ValueError('Frozen test instance index changed')
    with index.open() as stream:
        expected_ids = {row['instance_id'] for row in csv.DictReader(stream) if row['split'] == 'test'}
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    ids = [str(row['instance_id']) for row in rows]
    if len(rows) != 1000 or len(set(ids)) != 1000 or set(ids) != expected_ids:
        raise ValueError('Accepted test must contain exactly the 1000 audited unique instance IDs')
    selected_sha = digest(_selected_checkpoint(manifest, method, spec, cfg))
    if record.get('checkpoint_sha256') != selected_sha:
        raise ValueError('Test result does not use the frozen validation-selected checkpoint')
    for row in rows:
        if (row.get('method') != method or row.get('training_seed') != manifest['training_seed']
                or row.get('split') != 'test' or row.get('checkpoint_id') != selected_sha
                or row.get('requested_K') != 50 or row.get('actual_K') != 50):
            raise ValueError('Accepted test row method/seed/split/checkpoint/K contract mismatch')
    return record
