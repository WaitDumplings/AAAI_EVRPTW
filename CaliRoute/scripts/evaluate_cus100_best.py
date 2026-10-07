#!/usr/bin/env python3
"""Evaluate a validation-selected Cus100 checkpoint once on the frozen test set.

No training, checkpoint selection, or hyperparameter search takes place here.
GPU visibility is controlled by the caller with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from offline2online.input_normalization import (checkpoint_profile, signature as input_normalization_signature,
                                             configure as configure_input_normalization)

from offline2online.model_integration import (checkpoint_profile as integration_checkpoint_profile,
    signature as model_integration_signature, configure as configure_model_integration)
EVAL_SEED = 17_003_009
NUM_INSTANCES = 1000


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_output(path):
    """Allow a caller-owned console log, but never overwrite evaluation artifacts."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'Output is not a directory: {path}')
        occupied = [item.name for item in path.iterdir() if item.name != 'console.log']
        if occupied:
            raise ValueError(f'Refusing to overwrite existing evaluation artifacts in {path}: {occupied}')
    else:
        path.mkdir(parents=True)


def validate_checkpoint(checkpoint, problem, seed, selection):
    cfg = checkpoint.get('config', {})
    checkpoint_profile(checkpoint)
    integration_checkpoint_profile(checkpoint)
    data = cfg.get('data', {})
    if data.get('problem_type') != problem or int(data.get('num_customers', -1)) != 100:
        raise ValueError(f'Checkpoint must be {problem} Cus100; got {data.get("problem_type")}, {data.get("num_customers")}')
    if int(data.get('num_charging_stations', 0)) != 0:
        raise ValueError('CVRP/VRPTW checkpoint must have zero charging stations')
    if int(checkpoint.get('seed', -1)) != seed:
        raise ValueError('Checkpoint training seed does not match --seed')
    epoch = int(checkpoint.get('epoch', -1))
    if epoch < 1 or epoch > int(cfg.get('training', {}).get('epochs', 0)):
        raise ValueError('Checkpoint has no valid completed training epoch')
    if int(selection.get('epoch', -1)) != epoch:
        raise ValueError('best_checkpoint.json epoch does not match the supplied checkpoint')
    if selection.get('selection') != 'feasibility_then_distance_v1':
        raise ValueError('Expected the training validation-only best-checkpoint selection metadata')
    if 'model_state_dict' not in checkpoint or not cfg.get('model'):
        raise ValueError('Checkpoint is missing its full model state/config')
    return cfg, epoch


def validate_metadata(metadata, problem):
    if (metadata.get('split') != 'test'
            or str(metadata.get('problem_class', '')).lower() != problem
            or int(metadata.get('num_customers', -1)) != 100
            or int(metadata.get('num_instances', -1)) != NUM_INSTANCES):
        raise ValueError(f'Expected frozen test metadata for {problem} Cus100 with {NUM_INSTANCES} instances')


def validate_ids(instance_ids, reference_rows):
    if len(instance_ids) != NUM_INSTANCES or len(set(instance_ids)) != NUM_INSTANCES:
        raise ValueError(f'Expected exactly {NUM_INSTANCES} distinct frozen test instances')
    reference_ids = [row['instance_id'] for row in reference_rows]
    if len(reference_ids) != NUM_INSTANCES or len(set(reference_ids)) != NUM_INSTANCES:
        raise ValueError('Gurobi summary must contain exactly 1000 unique instance IDs')
    if set(reference_ids) != set(instance_ids):
        raise ValueError('Gurobi summary IDs do not exactly match this frozen test bundle')


def build_eval_config(original, dataset, reference, output):
    cfg = copy.deepcopy(original)
    cfg['run_name'] = str(original.get('run_name', 'CALIROUTE')) + '_FROZEN_BEST_TEST'
    evaluation = cfg.setdefault('evaluation', {})
    evaluation.update(eval_path=str(dataset), gurobi_summary_path=str(reference),
                      eval_output_dir=str(output / 'routes'), eval_seed=EVAL_SEED,
                      eval_n_traj=50, eval_batch_size=32, eval_decode_mode='sample',
                      eval_max_steps=201, eval_save_routes=True)
    evaluation.pop('eval_limit', None)
    evaluation.pop('eval_num_batches', None)
    return cfg


def constructor_kwargs(cfg, agent_class, device='cuda:0'):
    """Preserve all supported model kwargs and the training factory's aliases."""
    input_normalization_signature(cfg)
    model_integration_signature(cfg)
    model = cfg['model']
    parameters = inspect.signature(agent_class.__init__).parameters
    result = {key: model[key] for key in parameters if key in model and key not in {'self', 'device', 'name'}}
    # The common routing backbone retains its internal EVRPTW representation name.
    result['device'] = device
    critic = cfg.get('critic', {})
    result['use_decomposed_critic'] = bool(model.get('use_decomposed_critic', critic.get(
        'use_decomposed_critic', cfg.get('training', {}).get('use_decomposed_critic', True))))
    for suffix in ('delta_k', 'delta_v', 'delta_action_key', 'action_bias'):
        key = 'dynamic_decision_' + suffix
        result[key] = bool(model.get(key, model.get('dynamic_' + suffix, model.get(suffix, True))))
    mode = str(model.get('distance_injection', 'encoder')).lower().replace('-', '_')
    if mode not in {'encoder', 'encoder_bias', 'road_encoder', 'none', 'off', 'no'}:
        raise ValueError(f'Unsupported checkpoint distance_injection={mode!r}')
    result['use_encoder_distance_bias'] = bool(model.get('use_encoder_distance_bias', mode in {
        'encoder', 'encoder_bias', 'road_encoder'}))
    return result


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def summarize_rows(rows, reference_rows, instance_ids):
    ids = [row['instance_id'] for row in rows]
    if len(ids) != len(instance_ids) or len(set(ids)) != len(ids) or set(ids) != set(instance_ids):
        raise ValueError('Exported routes do not exactly cover the frozen test instance IDs')
    references = {row['instance_id']: row for row in reference_rows}
    output = []
    for row in rows:
        validation = row.get('route_validation', {})
        if validation.get('checked') is not True:
            raise ValueError(f'Independent route validation was not implemented/executed for {row["instance_id"]}: {validation}')
        feasible = row.get('feasible') is True
        if feasible and validation.get('valid') is not True:
            raise ValueError(f'Feasible route failed independent validation: {row["instance_id"]}')
        objective = finite_number(row.get('objective_distance_km')) if feasible else None
        if feasible and (objective is None or objective <= 0):
            raise ValueError(f'Feasible route has invalid objective distance: {row["instance_id"]}')
        reference = references[row['instance_id']]
        reference_feasible = str(reference.get('feasible', '')).strip().lower() in {'true', '1', 'yes', 'y'}
        reference_objective = finite_number(reference.get('objective_distance_km')) if reference_feasible else None
        if reference_objective is not None and reference_objective <= 0:
            reference_objective = None
        exported_reference = finite_number(row.get('reference_objective_distance_km'))
        if reference_objective is not None and (exported_reference is None or not math.isclose(
                reference_objective, exported_reference, rel_tol=1e-10, abs_tol=1e-7)):
            raise ValueError(f'Evaluator reference differs from the declared Gurobi summary: {row["instance_id"]}')
        gap = objective - reference_objective if objective is not None and reference_objective is not None else None
        output.append({'instance_id': row['instance_id'], 'feasible': feasible,
                       'environment_feasible': row.get('environment_feasible'),
                       'objective_distance_km': objective, 'reference_objective_distance_km': reference_objective,
                       'distance_gap_km': gap,
                       'relative_gap_pct': 100 * gap / reference_objective if gap is not None else None,
                       'route_validation_checked': True, 'route_validation_valid': validation.get('valid') is True,
                       'gurobi_status_name': reference.get('status_name'), 'runtime_s': finite_number(row.get('runtime_s'))})
    def mean(key):
        values = [row[key] for row in output if row[key] is not None]
        return statistics.fmean(values) if values else None
    summary = {'num_instances': len(output), 'feasible_instances': sum(row['feasible'] for row in output),
               'feasible_rate': sum(row['feasible'] for row in output) / len(output),
               'mean_distance_km': mean('objective_distance_km'), 'mean_distance_gap_km': mean('distance_gap_km'),
               'mean_relative_gap_pct': mean('relative_gap_pct'),
               'reference_instances_with_feasible_incumbent': sum(row['reference_objective_distance_km'] is not None for row in output),
               'paired_feasible_gap_instances': sum(row['distance_gap_km'] is not None for row in output),
               'independent_route_validation_checked': len(output),
               'independent_route_validation_passes': sum(row['route_validation_valid'] for row in output),
               'distance_scope': 'feasible model solutions only; FR includes every frozen test instance',
               'gap_scope': 'same-instance feasible model/Gurobi incumbents; mean of individual (model-reference)/reference percentages',
               'gap_sign': 'positive means model longer; negative means model shorter',
               'reference_scope': '2-hour Gurobi incumbents; TIME_LIMIT results are not certified optima'}
    return summary, output


def source_provenance():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE_ROOT, text=True)
    paths = sorted({Path(__file__).resolve(), *CODE_ROOT.joinpath('offline2online').rglob('*.py'),
                    *CODE_ROOT.joinpath('caliroute').rglob('*.py'),
                    *CODE_ROOT.joinpath('EVRPTW_Benchmark/Reinforcement_Learning/TERRAN').rglob('*.py')})
    return {'code_root': str(CODE_ROOT), 'git_commit': commit, 'git_status_porcelain': status,
            'source_sha256': {str(path.relative_to(CODE_ROOT)): digest(path) for path in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-root', type=Path, required=True)
    parser.add_argument('--gurobi-root', type=Path, required=True)
    parser.add_argument('--problem', choices=['cvrp', 'vrptw'], required=True)
    parser.add_argument('--seed', type=int, default=3009)
    args = parser.parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    dataset = args.test_root.resolve() / args.problem / 'test/Cus100'
    reference = args.gurobi_root.resolve() / args.problem / 'test/Cus100/gurobi_summary.csv'
    selection_path = checkpoint_path.with_name('best_checkpoint.json')
    # Validate input provenance before creating a CUDA context or touching output.
    metadata = json.loads((dataset / 'metadata.json').read_text())
    validate_metadata(metadata, args.problem)
    dataset_hash = digest(dataset / 'instances.pkl')
    if dataset_hash != metadata.get('bundle_sha256'):
        raise ValueError('Frozen test bundle SHA-256 does not match its metadata')
    selection = json.loads(selection_path.read_text())
    checkpoint_hash = digest(checkpoint_path)
    import torch
    import yaml
    from offline2online.instance_adapter import iter_adapted_instances
    from offline2online.models import Agent
    from offline2online.trainer import evaluate_fixed_dataset, _validate_dataset_metadata
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    original, epoch = validate_checkpoint(checkpoint, args.problem, args.seed, selection)
    _validate_dataset_metadata(dataset, num_customers=100, num_charging_stations=0, label='frozen_test')
    instance_ids = [instance.instance_id for instance in iter_adapted_instances(
        dataset, num_customers=100, num_charging_stations=0, problem_type=args.problem)]
    with reference.open(newline='') as stream:
        references = list(csv.DictReader(stream))
    validate_ids(instance_ids, references)
    prepare_output(output)
    cfg = build_eval_config(original, dataset, reference, output)
    kwargs = constructor_kwargs(cfg, Agent)
    manifest = {'created_at_utc': now(), 'checkpoint': str(checkpoint_path), 'checkpoint_epoch': epoch,
                'checkpoint_sha256': checkpoint_hash, 'selection_metadata': selection,
                'selection_metadata_path': str(selection_path), 'selection_metadata_sha256': digest(selection_path),
                'source_training_run': original.get('run_name'), 'training_seed': args.seed, 'eval_seed': EVAL_SEED,
                'dataset_path': str(dataset), 'dataset_sha256': dataset_hash,
                'metadata_sha256': digest(dataset / 'metadata.json'), 'gurobi_summary': str(reference),
                'gurobi_summary_sha256': digest(reference), 'instance_ids': instance_ids,
                'source': source_provenance(), 'model_constructor': kwargs,
                'input_normalization_signature': input_normalization_signature(cfg),
                'model_integration_signature': model_integration_signature(cfg),
                'protocol': {'problem': args.problem, 'customers': 100, 'num_instances': NUM_INSTANCES,
                             'split': 'frozen test', 'selection': 'validation-selected checkpoint fixed before this evaluation',
                             'training_or_tuning': False, 'strict_checkpoint_load': True, 'decode': 'sample',
                             'n_traj': 50, 'eval_batch_size': 32, 'max_steps': 201,
                             'selection_per_instance': 'shortest feasible of 50', 'precision': 'FP32; no autocast',
                             'reference_scope': '2-hour Gurobi incumbents; TIME_LIMIT results are not certified optima'},
                'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    write_json(output / 'manifest.json', manifest)
    (output / 'checkpoint_config.yaml').write_text(yaml.safe_dump(original, sort_keys=False))
    (output / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('A visible CUDA GPU is required; set CUDA_VISIBLE_DEVICES in the caller')
        agent = Agent(**kwargs).to('cuda:0')
        configure_input_normalization(agent, cfg)
        configure_model_integration(agent, cfg)
        agent.load_state_dict(checkpoint['model_state_dict'], strict=True)
        agent.eval()
        del checkpoint
        print(f'{now()} {args.problem} Cus100: evaluating validation best epoch {epoch} on all 1000 frozen test instances', flush=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = evaluate_fixed_dataset(agent, cfg, seed=args.seed, epoch=epoch, device='cuda:0')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if metrics.get('eval_status') != 'ok' or metrics.get('eval_num_instances') != NUM_INSTANCES:
            raise ValueError(f'Evaluator did not complete all frozen test instances: {metrics}')
        route_path = output / 'routes' / f'epoch_{epoch:04d}.jsonl'
        rows = [json.loads(line) for line in route_path.read_text().splitlines() if line.strip()]
        summary, per_instance = summarize_rows(rows, references, instance_ids)
        if digest(checkpoint_path) != checkpoint_hash or digest(reference) != manifest['gurobi_summary_sha256']:
            raise ValueError('Checkpoint or reference summary changed during evaluation')
        with (output / 'per_instance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_instance[0]))
            writer.writeheader()
            writer.writerows(per_instance)
        summary.update(status='completed', problem=args.problem, checkpoint=str(checkpoint_path),
                       checkpoint_epoch=epoch, checkpoint_sha256=checkpoint_hash, eval_seed=EVAL_SEED,
                       evaluation_wall_seconds=elapsed, routes_path=str(route_path), routes_sha256=digest(route_path),
                       metrics=metrics, manifest_path=str(output / 'manifest.json'), finished_at_utc=now())
        write_json(output / 'summary.json', summary)
        print(json.dumps(finite_json(summary), allow_nan=False), flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                           'error': str(error), 'checkpoint_epoch': epoch, 'at_utc': now()})
        raise


if __name__ == '__main__':
    main()
