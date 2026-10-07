"""Validate and import an exact single-GPU reward/norm continuation.

This module never stops or launches processes. A completed predecessor supplies
an immutable full checkpoint, committed logs, and its original configuration.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import yaml


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def validation_epochs(epochs, eval_interval, inherited=()):
    """Initial, periodic, terminal and already-committed resume evaluations."""
    positive_integer(epochs, 'epochs')
    positive_integer(eval_interval, 'eval_interval')
    inherited = list(inherited)
    if any(isinstance(e, bool) or not isinstance(e, int) or not 0 <= e <= epochs for e in inherited):
        raise ValueError('Inherited validation epochs must lie within the training horizon')
    return sorted({0, epochs, *range(eval_interval, epochs + 1, eval_interval), *inherited})


def required_validation_epochs(spec, epoch):
    """Read the frozen schedule, retaining nonperiodic predecessor end points.

    Older manifests did not store an explicit schedule; their configuration
    still records the interval and (for extensions) the predecessor horizon.
    """
    cfg = yaml.safe_load(Path(spec['config']).read_text())
    interval = positive_integer(cfg['evaluation']['eval_interval'], 'eval_interval')
    protocol = cfg.get('experiment_protocol', {})
    inherited = list(protocol.get('inherited_validation_epochs', []))
    boundary = protocol.get('continuation_from_epoch', spec.get('restored_through_epoch'))
    if boundary is not None:
        inherited.append(positive_integer(boundary, 'continuation_from_epoch'))
    expected = validation_epochs(epoch, interval, inherited)
    declared = spec.get('required_validation_epochs')
    if declared is not None:
        if (not isinstance(declared, list)
                or any(isinstance(e, bool) or not isinstance(e, int) or not 0 <= e <= epoch for e in declared)
                or declared != sorted(set(declared)) or not set(expected).issubset(declared)):
            raise ValueError('Declared validation schedule is incomplete or invalid')
        return declared
    return expected


def _canonical_sections(cfg):
    result = copy.deepcopy(cfg)
    aliases = {
        'sl_candidate_margin': 'sl_candidate_incumbent_margin',
        'sl_candidate_gate_eta': 'sl_candidate_incumbent_eta',
        'sl_candidate_use_current_incumbent_gate': 'sl_candidate_use_current_incumbent',
        'sl_candidate_use_memory_incumbent_gate': 'sl_candidate_use_memory_incumbent',
        'sl_use_expert_candidate': 'sl_candidate_use_expert_candidate',
    }
    for section in ('offline', 'advantage'):
        values = result.setdefault(section, {})
        for public, internal in aliases.items():
            if public in values:
                values.setdefault(internal, values[public])
    return result


def validate_extension_checkpoint(checkpoint, cfg, target_epochs, seed):
    """Accept only a completed, fully restorable fixed-schedule predecessor.

    ``cfg`` is the predecessor's original launch configuration, before changing
    its horizon or output paths. Runtime alias additions are canonicalized.
    """
    import torch
    from offline2online.reward_normalization import MODE, signature

    epoch = positive_integer(checkpoint.get('epoch', -1), 'checkpoint epoch')
    horizon = positive_integer(cfg.get('training', {}).get('epochs', -1), 'source epochs')
    positive_integer(cfg.get('evaluation', {}).get('eval_interval'), 'eval_interval')
    if epoch != horizon:
        raise ValueError('Extension requires a completed previous horizon')
    if positive_integer(target_epochs, 'target epochs') <= epoch:
        raise ValueError('Target epochs must exceed the completed previous horizon')
    if int(checkpoint.get('seed', -1)) != int(seed):
        raise ValueError('Extension seed changed')
    original, saved = _canonical_sections(cfg), _canonical_sections(checkpoint.get('config', {}))
    for section in ('data', 'model', 'env', 'critic', 'advantage', 'offline', 'training', 'evaluation', 'pbrs'):
        before, after = original.get(section, {}), saved.get(section, {})
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after or before[key] != after[key]:
                raise ValueError(f'Checkpoint algorithm configuration differs: {section}.{key}')
    train = original['training']
    if train.get('lr_schedule', 'constant') != 'constant' or int(train.get('lr_warmup_epochs', 0)) != 0:
        raise ValueError('Extension requires constant learning rate without warmup')
    initial = float(train.get('entropy_initial_coef', train.get('ent_coef', .01)))
    final = float(train.get('entropy_final_coef', initial))
    if not math.isfinite(initial) or initial < 0 or initial != final:
        raise ValueError('Extension requires equal constant entropy endpoints')
    if not math.isfinite(float(train.get('learning_rate', 0))) or float(train.get('learning_rate', 0)) <= 0:
        raise ValueError('Invalid constant learning rate')
    if train.get('post_init_seed', seed) != seed:
        raise ValueError('Configured training seed differs')
    if not checkpoint.get('model_state_dict') or not isinstance(checkpoint.get('optimizer_state_dict'), dict):
        raise ValueError('Full model and optimizer state are required')
    optimizer = checkpoint['optimizer_state_dict']
    if not optimizer.get('state') or not optimizer.get('param_groups'):
        raise ValueError('Optimizer moments and parameter groups are required')
    state = checkpoint.get('training_resume_state', {})
    if (state.get('world_size') != 1 or len(state.get('ranks', [])) != 1
            or not state.get('sampler_state_complete') or state.get('completed_epoch') != epoch
            or state.get('next_training_epoch') != epoch + 1):
        raise ValueError('Complete single-rank epoch-boundary state is required')
    if state.get('evaluation_pending') is not False:
        raise ValueError('Checkpoint validation is pending; use the completed final checkpoint')
    rank = state['ranks'][0]
    if rank.get('rank') != 0 or any(k not in rank for k in (
            'rng', 'sampler', 'expert_rng', 'policy_route_pool', 'policy_best_objectives',
            'scaler', 'optimizer_steps', 'amp_skipped_steps', 'sample_count_offset')):
        raise ValueError('Incomplete rank-zero training state')
    if any(rank['rng'].get(k) is None for k in ('python', 'numpy', 'torch')):
        raise ValueError('Training RNG state is incomplete')
    sampler = rank['sampler']
    if (not sampler.get('supported') or sampler.get('class') not in
            {'AdaptedFixedDatasetInstancePool', 'SolutionPrioritySampler'}
            or sampler.get('rng') is None or not sampler.get('attributes')
            or any(k not in sampler['attributes'] for k in ('order', 'cursor', 'sample_count'))):
        raise ValueError('Sampler state cannot be restored completely')
    if train.get('mixed_precision') and not rank['scaler']:
        raise ValueError('AMP scaler state is missing')
    if original.get('offline', {}).get('policy_replay_enabled') and rank['policy_route_pool'] is None:
        raise ValueError('Policy replay state is missing')
    if original.get('offline', {}).get('expert_solution_path') and rank['expert_rng'] is None:
        raise ValueError('Expert replay RNG state is missing')
    norm = checkpoint.get('reward_normalization_state')
    if train.get('reward_norm_mode', 'legacy') == MODE:
        if not isinstance(norm, dict) or norm.get('signature') != signature(original):
            raise ValueError('Reward normalization signature is missing or changed')
        for name, required in (
            ('actor', {'beta', 'minimum', 'sample_count', 'update_count', 'second_moment'}),
            ('critic', {'beta', 'minimum', 'sample_count', 'update_count', 'mean', 'variance'}),
            ('normalized_head', {'weight', 'bias'})):
            values = norm.get(name, {})
            if set(values) != required or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in values.values()):
                raise ValueError(f'Incomplete or nonfinite normalization state: {name}')
        for name, minimum in (('actor', 'actor_min_scale'), ('critic', 'critic_min_std')):
            values = norm[name]
            if (int(values['update_count']) != epoch or float(values['sample_count']) <= 0
                    or float(values['beta']) != norm['signature']['beta']
                    or float(values['minimum']) != norm['signature'][minimum]):
                raise ValueError(f'Normalization statistics are inconsistent: {name}')
        if float(norm['actor']['second_moment']) < 0 or float(norm['critic']['variance']) < 0:
            raise ValueError('Normalization variance must be nonnegative')
    elif norm is not None:
        raise ValueError('Legacy continuation cannot load normalized training state')
    return epoch


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def import_single_gpu_history(old_spec, new_spec, epoch):
    """Copy checkpoint-committed history and historical best without overwrites."""
    old_logs, new_logs = Path(old_spec['log_dir']), Path(new_spec['log_dir'])
    old_output, new_output = Path(old_spec['output_dir']), Path(new_spec['output_dir'])
    old_checkpoints, new_checkpoints = Path(old_spec['checkpoint_dir']), Path(new_spec['checkpoint_dir'])
    if new_logs.exists() or new_checkpoints.exists():
        raise ValueError('Resume destination logs/checkpoints already exist')
    eval_epochs = required_validation_epochs(old_spec, epoch)
    planned = []
    for filename in ('train_log.csv', 'eval_log.csv'):
        source = old_logs / filename
        with source.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row['epoch']) <= epoch]
        epochs = [int(row['epoch']) for row in rows]
        expected = list(range(1, epoch + 1)) if filename == 'train_log.csv' else eval_epochs
        if not fields or epochs != expected:
            raise ValueError(f'Missing or duplicate committed epoch history: {source}')
        if filename == 'eval_log.csv' and any(row.get('eval_status') != 'ok' or int(row.get('eval_num_instances', 0)) != 1000 for row in rows):
            raise ValueError('Every imported validation must be successful on all 1000 instances')
        planned.append((source, new_logs / filename, ('csv', fields, rows)))
    for e in eval_epochs:
        source = old_output / 'evaluations' / f'epoch_{e:04d}.jsonl'
        if not source.is_file():
            raise ValueError(f'Missing committed validation routes: {source}')
        planned.append((source, new_output / 'evaluations' / source.name, None))
    source = old_output / 'monitoring/monitor_rank_0.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row['epoch']) <= epoch]
    if any(row.get('rank', 0) != 0 for row in rows) or len({int(row['epoch']) for row in rows}) != len(rows):
        raise ValueError('Monitor history contains duplicate epochs or multiple ranks')
    planned.append((source, new_output / 'monitoring' / source.name, ('jsonl', rows)))
    meta_source = old_checkpoints / 'best_checkpoint.json'
    meta = json.loads(meta_source.read_text())
    if int(meta['epoch']) <= 0 or int(meta['epoch']) not in eval_epochs:
        raise ValueError('Historical best lies outside the committed validation history')
    for filename in ('best_checkpoint.json', 'checkpoint_best.pt'):
        source = old_checkpoints / filename
        if not source.is_file():
            raise ValueError(f'Missing historical best: {source}')
        planned.append((source, new_checkpoints / filename, None))
    if any(destination.exists() for _, destination, _ in planned):
        raise ValueError('Refusing to overwrite existing continuation artifacts')
    records = {}
    for source, destination, content in planned:
        before = _digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            shutil.copy2(source, destination)
        elif content[0] == 'csv':
            with destination.open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=content[1])
                writer.writeheader()
                writer.writerows(content[2])
        else:
            destination.write_text(''.join(json.dumps(row) + '\n' for row in content[1]))
        if _digest(source) != before:
            raise ValueError(f'Predecessor history changed while copying: {source}')
        records[str(destination)] = dict(source=str(source), source_sha256=before, copied_sha256=_digest(destination))
        if content is not None:
            records[str(destination)]['rows'] = len(content[-1])
    return records
