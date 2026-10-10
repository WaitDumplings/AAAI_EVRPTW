#!/usr/bin/env python3
"""Export verified experiment evidence without datasets, weights, or GPU access.

Usage:
    python scripts/export_run_evidence.py RUN_DIR [RUN_DIR ...] --output NEW_DIR
    python scripts/export_run_evidence.py RUN_DIR --output NEW_DIR --include-source

Frozen sources, inputs and configs must match their manifest hashes. Live JSON
and rank-0 CSV logs are read once and exported as timestamped snapshots. Their
package hashes establish the exported bytes, not cross-file snapshot atomicity.
Only the standard library is required; nothing imports the training stack.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys


SOURCE_SUFFIXES = frozenset({'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.toml'})
SOURCE_EXCLUDED_PARTS = frozenset({
    'results', 'assets', 'data', 'dataset', 'datasets', 'aaai_dataset',
    'checkpoints', 'checkpoint', 'weights', 'weight', '.git', '__pycache__',
})
LOG_NAMES = ('train_log.csv', 'eval_log.csv')
LIVE_NAMES = ('status.json', 'comparison.json')
SCHEMA = 'caliroute_run_evidence_v1'


class EvidenceError(ValueError):
    """A run cannot be exported without violating its evidence contract."""


def now():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checksum(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise EvidenceError(f'Missing or invalid SHA256 for {label}')
    return value.lower()


def _component(value, label):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or '/' in value or '\\' in value or '\x00' in value):
        raise EvidenceError(f'Unsafe {label}: {value!r}')
    return value


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise EvidenceError(f'Unsafe source-relative path: {value!r}')
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {'', '.', '..'} for part in value.split('/')):
        raise EvidenceError(f'Unsafe source-relative path: {value!r}')
    return relative


def _resolve(value, base, label):
    if not isinstance(value, (str, Path)) or not str(value):
        raise EvidenceError(f'Missing path for {label}')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def _contained(path, root, label):
    actual = path.resolve(strict=True)
    try:
        actual.relative_to(root)
    except ValueError as error:
        raise EvidenceError(f'{label} escapes its declared root: {path} -> {actual}') from error
    return actual


def _read(path, *, expected=None, capture=True):
    """Hash exactly the bytes captured for copying, or stream large inputs."""
    if not path.is_file():
        raise EvidenceError(f'Not a regular file: {path}')
    digest = hashlib.sha256()
    chunks = [] if capture else None
    size = 0
    with path.open('rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise EvidenceError(f'Not a regular file: {path}')
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if capture:
                chunks.append(chunk)
    actual = digest.hexdigest()
    if expected is not None and actual != _checksum(expected, str(path)):
        raise EvidenceError(f'SHA256 mismatch for {path}: expected {expected}, actual {actual}')
    return (b''.join(chunks) if capture else None), {'sha256': actual, 'bytes': size}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def _mapping(data, label):
    try:
        result = json.loads(data)
    except (UnicodeDecodeError, ValueError) as error:
        raise EvidenceError(f'Invalid JSON in {label}: {error}') from error
    if not isinstance(result, dict):
        raise EvidenceError(f'{label} must contain a JSON object')
    return result


def _copy_bytes(output, relative, data, inventory, *, original=None, kind):
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        handle.write(data)
    item = {'sha256': _sha(data), 'bytes': len(data), 'kind': kind}
    if original is not None:
        item['original_path'] = str(original)
    inventory[relative.as_posix()] = item


def _source_files(source, label):
    if not isinstance(source, Mapping):
        raise EvidenceError(f'Missing source mapping for {label}')
    files = source.get('files', source.get('file_sha256'))
    if not isinstance(files, Mapping) or not files:
        raise EvidenceError(f'Missing recorded source files for {label}')
    result = {}
    for relative, checksum in files.items():
        _relative(relative)
        if isinstance(checksum, Mapping):
            checksum = checksum.get('sha256')
        result[relative] = _checksum(checksum, f'{label}/{relative}')
    # source_snapshot uses this exact encoding for the aggregate file mapping.
    if source.get('content_sha256') is not None:
        actual = _sha(json.dumps(result, sort_keys=True).encode())
        if actual != _checksum(source['content_sha256'], f'{label} source mapping'):
            raise EvidenceError(f'Source mapping SHA256 mismatch for {label}')
    return result


def _source_allowed(relative):
    return (relative.suffix.lower() in SOURCE_SUFFIXES
            and not any(part.lower() in SOURCE_EXCLUDED_PARTS for part in relative.parts))


def _text_source(data, label):
    try:
        data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise EvidenceError(f'Allowlisted source is not UTF-8 text: {label}') from error
    if b'\x00' in data:
        raise EvidenceError(f'Allowlisted source contains binary NUL bytes: {label}')


def _stages(value, prefix=()):
    """Find recorded formal/preflight/warmup configs, without interpreting code."""
    if not isinstance(value, Mapping):
        return
    if 'config' in value:
        yield prefix, value
    for name, item in value.items():
        if isinstance(item, Mapping):
            yield from _stages(item, prefix + (_component(name, 'stage name'),))


def _export_one(run_dir, destination, output, inventory, include_source):
    manifest_path = _contained(run_dir / 'manifest.json', run_dir, 'manifest')
    manifest_data, manifest_hash = _read(manifest_path)
    manifest = _mapping(manifest_data, manifest_path)
    _copy_bytes(output, destination / 'manifest.json', manifest_data, inventory,
                original=manifest_path, kind='frozen_manifest')
    report = {'schema': SCHEMA, 'run_dir': str(run_dir), 'started_at_utc': now(),
              'manifest': manifest_hash, 'sources': {}, 'inputs': {}, 'configs': {},
              'live_snapshots': {}, 'logs': {}, 'warnings': [],
              'checkpoints': 'Standalone checkpoint references are retained but not opened or verified. Files recorded in source.files are hashed for source integrity even if they are excluded checkpoint assets. No checkpoint bytes are exported.'}
    inputs = manifest.get('inputs')
    if not isinstance(inputs, Mapping) or not inputs:
        raise EvidenceError('Manifest must declare nonempty inputs for SHA256 verification')
    input_paths = {}
    for name, item in inputs.items():
        if not isinstance(item, Mapping):
            raise EvidenceError(f'Invalid input entry: {name}')
        input_paths[name] = _resolve(item.get('path'), run_dir, f'input {name}')
    input_files = set(input_paths.values())
    input_identities = {(path.stat().st_dev, path.stat().st_ino) for path in input_files}
    roots = {}
    sources = {'main': {'code_root': manifest.get('code_root'), 'source': manifest.get('source')}}
    additional = manifest.get('additional_sources') or {}
    if not isinstance(additional, Mapping):
        raise EvidenceError('additional_sources must be a mapping')
    for name, item in additional.items():
        _component(name, 'additional source name')
        sources[f'additional_{name}'] = item
    for name, item in sources.items():
        if not isinstance(item, Mapping):
            raise EvidenceError(f'Invalid source entry: {name}')
        root = _resolve(item.get('code_root'), run_dir, f'{name} code_root')
        if not root.is_dir():
            raise EvidenceError(f'Source root is not a directory: {root}')
        roots[name] = root
        files = _source_files(item.get('source'), name)
        checks = {}
        exported, excluded = [], []
        for relative, expected in files.items():
            rel = _relative(relative)
            path = _contained(root / rel, root, 'source file')
            file_stat = path.stat()
            copy_source = (include_source and _source_allowed(rel)
                           and _source_allowed(path.relative_to(root)) and path not in input_files
                           and (file_stat.st_dev, file_stat.st_ino) not in input_identities)
            content, checked = _read(path, expected=expected, capture=copy_source)
            checks[relative] = {**checked, 'expected_sha256': expected, 'verified': True}
            if copy_source:
                _text_source(content, path)
                _copy_bytes(output, destination / 'source' / name / rel, content,
                            inventory, original=path, kind='verified_source')
                exported.append(relative)
            elif include_source:
                excluded.append(relative)
        report['sources'][name] = {'code_root': str(root), 'git_commit': item['source'].get('git_commit'),
            'verified': True, 'files': checks, 'exported_files': exported,
            'excluded_from_source_export': excluded}
    for name, item in inputs.items():
        path = input_paths[name]
        _, checked = _read(path, expected=_checksum(item.get('sha256'), f'input {name}'), capture=False)
        report['inputs'][name] = {'path': str(path), **checked, 'expected_sha256': item['sha256'],
                                 'verified': True, 'exported': False}
    arms = manifest.get('arms')
    if not isinstance(arms, Mapping) or not arms:
        raise EvidenceError('Manifest must declare a nonempty arms mapping')
    declared_roots = set(roots.values())
    for arm, value in arms.items():
        _component(arm, 'arm name')
        if not isinstance(value, Mapping) or 'config' not in value:
            raise EvidenceError(f'Arm {arm} must declare its formal config')
        for parts, stage in _stages(value):
            stage_name = '/'.join((arm, *parts))
            code_root = _resolve(stage.get('code_root', value.get('code_root', manifest['code_root'])),
                                 run_dir, f'{stage_name} execution code_root')
            if code_root not in declared_roots:
                raise EvidenceError(f'Unverified execution source for {stage_name}: {code_root}')
            config_path = _resolve(stage['config'], run_dir, f'{stage_name} config')
            config_path = _contained(config_path, run_dir, 'configuration')
            if config_path.suffix.lower() not in {'.yaml', '.yml', '.json'}:
                raise EvidenceError(f'Unexpected config file suffix: {config_path}')
            data, checked = _read(config_path, expected=_checksum(stage.get('config_sha256'), f'{stage_name} config'))
            target = destination / 'arms' / arm
            for part in parts:
                target /= part
            _copy_bytes(output, target / ('config' + config_path.suffix.lower()), data, inventory,
                        original=config_path, kind='verified_config')
            report['configs'][stage_name] = {'path': str(config_path), **checked,
                'expected_sha256': stage['config_sha256'], 'verified': True}
            log_dir = stage.get('log_dir')
            if not log_dir:
                report['warnings'].append(f'{stage_name}: no recorded rank-0 log directory')
                continue
            log_root = Path(log_dir).expanduser()
            log_root = (log_root if log_root.is_absolute() else run_dir / log_root).resolve()
            for log_name in LOG_NAMES:
                log_path = log_root / log_name
                label = f'{stage_name}/{log_name}'
                if not log_path.exists():
                    report['logs'][label] = {'present': False, 'path': str(log_path)}
                    continue
                actual = _contained(log_path, log_root, 'rank-0 log')
                content, captured = _read(actual)
                _copy_bytes(output, target / 'logs' / log_name, content, inventory,
                            original=actual, kind='rank0_log_snapshot')
                report['logs'][label] = {**captured, 'present': True, 'path': str(actual),
                    'captured_at_utc': now(), 'snapshot_only': True,
                    'possibly_partial_final_line': bool(content and not content.endswith(b'\n'))}
    for name in LIVE_NAMES:
        path = run_dir / name
        if not path.exists():
            report['live_snapshots'][name] = {'present': False}
            continue
        path = _contained(path, run_dir, 'live metadata')
        content, captured = _read(path)
        _copy_bytes(output, destination / name, content, inventory,
                    original=path, kind='live_metadata_snapshot')
        entry = {**captured, 'present': True, 'snapshot_only': True, 'captured_at_utc': now()}
        try:
            json.loads(content)
            entry['valid_json'] = True
        except (UnicodeDecodeError, ValueError):
            entry['valid_json'] = False
            report['warnings'].append(f'{name}: preserved in-progress bytes; JSON parsing failed')
        report['live_snapshots'][name] = entry
    report.update(verified=True, completed_at_utc=now())
    _copy_bytes(output, destination / 'verification.json', _json_bytes(report), inventory,
                kind='verification_report')
    return {'source_run_dir': str(run_dir), 'directory': destination.as_posix(),
            'verification': (destination / 'verification.json').as_posix(),
            'verified_source_files': sum(len(item['files']) for item in report['sources'].values()),
            'verified_inputs': len(report['inputs']), 'verified_configs': len(report['configs'])}


def export_run_evidence(run_dirs, output, *, include_source=False):
    """Create a new evidence directory; remove this export if verification fails.

    The output is exclusively reserved before copying. ``evidence_manifest.json``
    is written last as the completion marker; an existing output is never reused.
    No checkpoint or input dataset bytes are included, even with include_source.
    """
    run_dirs = [Path(path).expanduser().resolve(strict=True) for path in run_dirs]
    if not run_dirs or len(set(run_dirs)) != len(run_dirs):
        raise EvidenceError('Provide one or more distinct run directories')
    if any(not path.is_dir() for path in run_dirs):
        raise EvidenceError('Each run must be a directory containing manifest.json')
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f'Output must not already exist: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation also closes a concurrent-create race.
    owned = output.stat()
    inventory = {}
    bundle = {'schema': SCHEMA, 'created_at_utc': now(), 'include_source': bool(include_source),
        'policy': {'datasets_exported': False, 'checkpoints_exported': False,
            'source_verified_in_full': True, 'source_export_is_allowlisted_subset': bool(include_source),
            'live_snapshots_are_cross_file_atomic': False,
            'live_snapshot_note': 'Each status/comparison/log file was read once. Files can represent slightly different instants while training continues.'},
        'runs': [], 'files': inventory}
    try:
        for index, run_dir in enumerate(run_dirs, 1):
            destination = Path('runs') / f'{index:02d}_{run_dir.name}'
            bundle['runs'].append(_export_one(run_dir, destination, output, inventory, include_source))
        bundle['completed_at_utc'] = now()
        marker = output / 'evidence_manifest.json'
        with marker.open('xb') as handle:
            handle.write(_json_bytes(bundle))
    except BaseException:
        # Remove only the directory this call created, never an existing output.
        try:
            current = output.lstat()
            if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                shutil.rmtree(output)
        except OSError:
            pass
        raise
    return output / 'evidence_manifest.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', type=Path, nargs='+', help='Run directories containing frozen manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are rejected')
    parser.add_argument('--include-source', action='store_true', help='Also copy verified allowlisted source text; never datasets or checkpoints')
    args = parser.parse_args(argv)
    try:
        report = export_run_evidence(args.run_dirs, args.output, include_source=args.include_source)
    except (OSError, EvidenceError) as error:
        parser.exit(1, f'Cannot export evidence: {error}\n')
    print(f'Export complete: {report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
