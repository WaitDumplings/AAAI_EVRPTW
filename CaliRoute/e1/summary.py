"""Merge accepted E1 tests without choosing checkpoints or runs by test score."""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

from e1.configs import METHODS, SCHEMA, content_hash
from e1.evaluation import atomic_json, summarize_rows
from e1.lifecycle import assert_accepted_test_result

_CONTRACT_FIELDS = {
    'schema', 'training_seed', 'online_loops', 'global_batch',
    'total_instance_exposures', 'n_traj', 'controlled_ppo_passes', 'gamma',
    'eval_K', 'source_content_sha256', 'expert_pool_semantic_sha256', 'dataset_sha256',
}
_GAP_DEFINITION = ('100*(model physical km / feasible Gurobi incumbent km - 1); '
                   'NOT a certified optimality gap')
_CSV_FIELDS = (
    'method', 'state', 'training_seed', 'instances', 'feasible_count', 'failure_count',
    'feasible_rate', 'mean_distance_km', 'mean_vehicle_count', 'mean_runtime_s',
    'timing_batch_sizes', 'runtime_kinds', 'timing_scopes', 'server', 'requested_K',
    'actual_K', 'gurobi_incumbent_available', 'model_feasible_and_gurobi_incumbent_count',
    'mean_gap_to_gurobi_incumbent_pct', 'negative_incumbent_gap_count',
    'available_method_common_feasible_ids', 'common_feasible_mean_km',
    'cost_mismatch_candidates', 'checkpoint_id', 'campaign', 'instance_file',
    'acceptance_error',
)


def _references(path):
    records = {}; seen = set()
    if path is None:
        return records
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            identity = str(row['instance_id'])
            if identity in seen:
                raise ValueError(f'Duplicate Gurobi ID: {identity}')
            seen.add(identity)
            try:
                cost = float(row['objective_distance_km'])
            except (ValueError, TypeError, KeyError):
                continue
            if not math.isfinite(cost) or cost <= 0:
                continue
            feasible = str(row.get('verified_feasible', row.get('feasible', 'true'))).lower()
            if feasible in ('false', '0', 'no'):
                continue
            records[identity] = cost
    return records


def _different_fields(left, right, prefix=''):
    differences = []
    for key in sorted(set(left) | set(right)):
        name = f'{prefix}.{key}' if prefix else key
        if key not in left or key not in right:
            differences.append(name)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            differences.extend(_different_fields(left[key], right[key], name))
        elif left[key] != right[key]:
            differences.append(name)
    return differences


def _load_campaign(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError(f'Unknown E1 campaign schema: {path}')
    unknown = set(manifest['methods']) - set(METHODS)
    if unknown:
        raise ValueError(f'Unknown E1 methods in {path}: {sorted(unknown)}')
    contract = manifest.get('comparison_contract')
    if not isinstance(contract, dict) or _CONTRACT_FIELDS - set(contract):
        raise ValueError(f'Missing complete comparison_contract in {path}; regenerate the campaign manifest')
    if contract['schema'] != SCHEMA or contract['training_seed'] != manifest['training_seed']:
        raise ValueError(f'Campaign seed/schema disagrees with comparison_contract: {path}')
    if contract['source_content_sha256'] != manifest['source']['content_sha256']:
        raise ValueError(f'Campaign source disagrees with comparison_contract: {path}')
    if not isinstance(contract['dataset_sha256'], dict) or set(contract['dataset_sha256']) != {'train', 'val', 'test'}:
        raise ValueError(f'Comparison contract must identify train, val and test data: {path}')
    # Hashing also refuses NaN/Infinity, rather than letting them bypass equality.
    content_hash(contract)
    return _rebase_manifest(manifest, path)


def _rebase_manifest(manifest, campaign):
    """Resolve copied campaign assets without editing frozen manifest/config bytes.

    The artifact tree is local to a campaign. Its YAML may retain the original
    server's dataset paths: acceptance hashes YAML bytes and does not load that
    remote dataset. An explicit artifacts_root avoids stale absolute symlinks
    under source/CaliRoute/results when complete campaign trees are copied.
    """
    result = copy.deepcopy(manifest)
    recorded = Path(manifest['campaign'])

    def rebase(value):
        try:
            suffix = Path(value).relative_to(recorded)
        except ValueError:
            return value
        return str(campaign / suffix)

    result['campaign'] = str(campaign)
    for key in ('code_root', 'audit_path', 'artifacts_root'):
        if key in result:
            result[key] = rebase(result[key])
    for spec in result['methods'].values():
        for key in ('config', 'run_dir'):
            spec[key] = rebase(spec[key])
    return result


def _read_accepted_rows(manifest, method, path):
    acceptance = assert_accepted_test_result(manifest, method)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [str(row['instance_id']) for row in rows]
    # Lifecycle verifies audited IDs, checkpoint, seed, split and exact K. Keep
    # these cheap local guards so a malformed row cannot pollute aggregate km.
    if len(rows) != 1000 or len(set(identities)) != 1000:
        raise ValueError('Accepted E1 test must have 1000 unique instance IDs')
    for row in rows:
        row['instance_id'] = str(row['instance_id'])
        if row['method'] != method or not isinstance(row['feasible'], bool):
            raise ValueError('Invalid method or feasible flag in accepted test row')
        if row['feasible']:
            cost = row['recomputed_cost_km']
            if isinstance(cost, bool) or not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError('Feasible test cost must be finite physical km')
            row['recomputed_cost_km'] = float(cost)
            if not row['completed'] or row['truncated']:
                raise ValueError('Feasible test row must be complete and not truncated')
        elif row['recomputed_cost_km'] is not None:
            raise ValueError('Failed test instance must retain a null cost')
        if not math.isfinite(float(row['runtime'])) or float(row['runtime']) < 0:
            raise ValueError('Test runtime must be finite and nonnegative')
    return acceptance, rows


def _summarize_method(rows, references):
    summary = summarize_rows(rows)
    identities = [row['instance_id'] for row in rows]
    matched = [(row['recomputed_cost_km'] / references[row['instance_id']] - 1) * 100
               for row in rows if row['feasible'] and row['instance_id'] in references]
    summary.update(
        state='test_completed',
        timing_batch_sizes=sorted(set(row['timing_batch_size'] for row in rows)),
        runtime_kinds=sorted(set(row['runtime_kind'] for row in rows)),
        timing_scopes=sorted(set(row['timing_scope'] for row in rows)),
        checkpoint_id=rows[0]['checkpoint_id'],
        gurobi_incumbent_available=sum(identity in references for identity in identities),
        model_feasible_and_gurobi_incumbent_count=len(matched),
        mean_gap_to_gurobi_incumbent_pct=sum(matched) / len(matched) if matched else None,
        negative_incumbent_gap_count=sum(gap < 0 for gap in matched),
        gap_definition=_GAP_DEFINITION,
    )
    return summary


def _write_csv(path, result):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for method in METHODS:
            info = result['methods'][method]
            row = {key: info.get(key) for key in _CSV_FIELDS}
            row.update(method=method, training_seed=result['training_seed'],
                       available_method_common_feasible_ids=result['available_method_common_feasible_ids'],
                       common_feasible_mean_km=result['common_feasible_mean_km'].get(method))
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, sort_keys=True, allow_nan=False)
            writer.writerow(row)
    temporary.replace(path)


def summarize_campaign(campaign, *, gurobi_path=None, additional_campaigns=()):
    """Merge one accepted test per method across semantically identical campaigns.

    Complete campaign trees copied from another server are rebased in memory
    to their supplied directories. Frozen manifest/config bytes remain intact;
    acceptance requires local checkpoint/test/index artifacts, not remote data.
    """
    campaign = Path(campaign).resolve()
    paths = [campaign, *(Path(path).resolve() for path in additional_campaigns)]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate campaign path; supply each campaign only once')
    manifests = [_load_campaign(path) for path in paths]
    contract = manifests[0]['comparison_contract']
    for path, manifest in zip(paths[1:], manifests[1:]):
        differences = _different_fields(contract, manifest['comparison_contract'])
        if differences:
            raise ValueError(f'Incompatible comparison_contract in {path}: {", ".join(differences)}')
    references = _references(gurobi_path)
    attempts = {method: [] for method in METHODS}
    accepted = {method: [] for method in METHODS}
    for path, manifest in zip(paths, manifests):
        for method, spec in manifest['methods'].items():
            rows_path = Path(spec['run_dir']) / 'test/instances.jsonl'
            record_path = rows_path.with_name('manifest.json')
            attempt = dict(campaign=str(path), instance_file=str(rows_path),
                           test_manifest=str(record_path), state='test_not_completed')
            attempts[method].append(attempt)
            if not rows_path.exists() and not record_path.exists():
                continue
            try:
                acceptance, rows = _read_accepted_rows(manifest, method, rows_path)
                summary = _summarize_method(rows, references)
            except (OSError, ValueError, TypeError, KeyError) as error:
                attempt.update(state='test_unaccepted', acceptance_error=str(error))
                continue
            attempt['state'] = 'test_completed'
            summary.update(campaign=str(path), instance_file=str(rows_path),
                           test_manifest=str(record_path), acceptance=acceptance,
                           server=spec.get('server'))
            accepted[method].append((summary, rows))
    duplicates = {method: [item[0]['campaign'] for item in results]
                  for method, results in accepted.items() if len(results) > 1}
    if duplicates:
        raise ValueError('Duplicate accepted test results; choosing a run using test scores is forbidden: '
                         + json.dumps(duplicates, sort_keys=True))
    methods = {}; paired = {}
    for method in METHODS:
        if accepted[method]:
            summary, rows = accepted[method][0]
            paired[method] = {row['instance_id']: row for row in rows}
        else:
            rejected = [attempt for attempt in attempts[method] if attempt['state'] == 'test_unaccepted']
            summary = dict(state='test_unaccepted' if rejected else 'test_not_completed')
            if rejected:
                summary['acceptance_error'] = '; '.join(
                    f"{attempt['campaign']}: {attempt['acceptance_error']}" for attempt in rejected)
        summary['campaign_attempts'] = attempts[method]
        methods[method] = summary
    identity_sets = [set(rows) for rows in paired.values()]
    common = set.intersection(*identity_sets) if identity_sets else set()
    common_feasible = {identity for identity in common
                       if all(rows[identity]['feasible'] for rows in paired.values())}
    result = dict(
        schema='aaai_e1_test_summary_v1', training_seed=manifests[0]['training_seed'],
        comparison_contract=contract, comparison_contract_sha256=content_hash(contract),
        campaigns=[str(path) for path in paths], methods=methods,
        accepted_methods=[method for method in METHODS if method in paired],
        gurobi_summary_path=str(Path(gurobi_path).resolve()) if gurobi_path else None,
        available_method_common_ids=len(common),
        available_method_common_feasible_ids=len(common_feasible),
        common_feasible_mean_km={method: sum(rows[i]['recomputed_cost_km'] for i in common_feasible)
                                / len(common_feasible) if common_feasible else None
                                for method, rows in paired.items()},
        statistical_scope='Single training seed. Means are over instances, not repeated training seeds.',
        runtime_comparison_scope='Reported timing retains batch size/scope and assigned server; '
                                 'native A6000 and controlled 2080 Ti runtimes are not a hardware-matched speed comparison.',
        complete_seven_method_test=all(methods[method]['state'] == 'test_completed' for method in METHODS),
    )
    # No output is written before every duplicate/contract check has passed.
    _write_csv(campaign / 'summary.csv', result)
    atomic_json(campaign / 'summary.json', result)
    return result
