#!/usr/bin/env python3
"""Run an immutable original CaliRoute revision from random initialization.

Evaluation instrumentation adds fixed isolated RNG, optional epoch-zero eval,
route exports, and independent physical feasibility checks. The archived model,
optimizer, losses and environment files are imported unchanged. Under torchrun,
an explicitly declared external optimizer-boundary synchronization layer adds
data parallelism; the historical revision has no native DDP implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

# This standalone NumPy helper imports no project model or environment code.
_helper_spec = importlib.util.spec_from_file_location(
    '_original_eval_validation', Path(__file__).with_name('original_eval_validation.py'))
_validation = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_validation)
_runtime_spec = importlib.util.spec_from_file_location(
    '_original_external_distributed', Path(__file__).with_name('original_distributed.py'))
_distributed = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(_distributed)
_warmup_spec = importlib.util.spec_from_file_location(
    '_original_ppo_warmup', Path(__file__).with_name('ppo_warmup.py'))
_warmup = importlib.util.module_from_spec(_warmup_spec)
_warmup_spec.loader.exec_module(_warmup)

PROJECT_PREFIXES = ('offline2online', 'EVRPTW_Benchmark', 'evrptw_core',
                    'evrptw_hierarchy', 'caliroute', 'ablation')
CHECKPOINT_KEYS = frozenset({'init_checkpoint', 'init_checkpoint_path',
                           'initial_checkpoint', 'initial_checkpoint_path',
                           'resume_checkpoint', 'resume_checkpoint_path',
                           'reference_checkpoint', 'reference_checkpoint_path',
                           'pretrained_checkpoint', 'pretrained_checkpoint_path',
                           'pretrained_path'})


def assert_scratch_config(cfg):
    """Fail before any checkpoint can be opened, including nested aliases."""
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key)
                label = prefix + name
                if (name in CHECKPOINT_KEYS or name.endswith('_checkpoint_path')) and item not in (None, ''):
                    raise ValueError(f'Original scratch baseline forbids initialization/resume/reference checkpoints: {label}')
                if name == 'resume_start_epoch' and item not in (None, 1):
                    raise ValueError(f'Original scratch baseline must start at epoch 1: {label}')
                visit(item, label + '.')
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, prefix + f'{index}.')
    visit(cfg)


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError, TypeError):
        return False


def assert_project_origins(source_root):
    """Reject mixed historical/current imports, including environments/models."""
    source_root = Path(source_root).resolve()
    origins = {}
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in PROJECT_PREFIXES):
            continue
        filename = getattr(module, '__file__', None)
        paths = [filename] if filename is not None else list(getattr(module, '__path__', ()))
        if any(not _inside(path, source_root) for path in paths):
            raise RuntimeError(f'Original baseline imported project code outside its archive: {name}: {paths}')
        if filename is not None:
            origins[name] = str(Path(filename).resolve())
    return origins


def _import_original(source_root):
    source_root = Path(source_root).resolve()
    if not (source_root / 'offline2online/trainer.py').is_file():
        raise FileNotFoundError('source-root must contain the archived offline2online/trainer.py')
    assert_project_origins(source_root)
    current_code_root = Path(__file__).resolve().parents[1]
    # Remove the adapter's current checkout and cwd from module search. The
    # archive may itself be inside current results; it is reinserted explicitly.
    sys.path[:] = [str(source_root)] + [entry for entry in sys.path
        if entry and not _inside(entry, current_code_root) and Path(entry).resolve() != Path.cwd().resolve()]
    os.environ['EVRPTW_DB_ROOT'] = str(source_root)
    trainer = importlib.import_module('offline2online.trainer')
    origins = assert_project_origins(source_root)
    if not any(name.startswith('offline2online.models') for name in origins):
        raise RuntimeError('Original trainer did not load its archived model module')
    if not any(name.startswith('EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env') for name in origins):
        raise RuntimeError('Original trainer did not load its archived environment module')
    return trainer, origins


@contextmanager
def isolated_evaluation_rng(seed, device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            torch.random.default_generator.manual_seed(int(seed))
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _write_metadata(path, record):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _load_validation_sidecar(trainer, runtime_cfg):
    """Read only edge seconds dropped by the original classical data adapter."""
    adapter = importlib.import_module('offline2online.instance_adapter')
    path = trainer._resolve_path(runtime_cfg.get('evaluation', {}).get('eval_path'))
    if path is None or not path.exists():
        return {}
    data = runtime_cfg.get('data', {})
    problem = trainer.problem_type_from_config(runtime_cfg)
    sidecar = {}
    for index, payload in enumerate(adapter.iter_instance_payloads(
            path, num_customers=int(data.get('num_customers', 15)),
            num_charging_stations=trainer.num_charging_stations_for_problem(data, problem))):
        identity = str(payload.get('instance_id') or f'adapted_{index:06d}')
        values = {name: payload[name] for name in
                  ('travel_time_matrix_s', 'time_matrix_s', 'travel_time_matrix', 'energy_matrix_kwh', 'edge_energy_kwh')
                  if payload.get(name) is not None}
        if 'energy_matrix_kwh' not in values and 'edge_energy_kwh' in values:
            values['energy_matrix_kwh'] = values['edge_energy_kwh']
        if values:
            if identity in sidecar:
                raise ValueError(f'Duplicate validation instance with explicit time matrix: {identity}')
            sidecar[identity] = values
    return sidecar


def run_original_scratch(source_root, config_path, seed, device):
    source_root, config_path = Path(source_root).resolve(), Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Original baseline configuration must be a mapping')
    assert_scratch_config(cfg)
    warmup_schedule = _warmup.PPOWarmupSchedule(cfg)
    trainer, origins = _import_original(source_root)
    runtime = _distributed.OriginalDistributedRuntime(cfg, seed, device, config_path.parent)
    context, device = runtime.context, runtime.device
    warmup_runtime = _warmup.OriginalPPOWarmupRuntime(warmup_schedule, is_primary=context.is_primary)
    fixed_seed = int(cfg.get('evaluation', {}).get('eval_seed', 17_000_000 + int(seed)))
    if fixed_seed < 0:
        raise ValueError('eval_seed must be nonnegative')
    initial_eval = bool(cfg.get('evaluation', {}).get('eval_before_training', False))
    metadata_path = config_path.with_name('original_adapter.json' if context.is_primary else f'original_adapter_rank_{context.rank}.json')
    problem = trainer.problem_type_from_config(cfg)
    validation_problem = 'evrptw' if problem == 'evrptw' else 'vrptw'
    metadata = {
        'adapter_schema': 'original_scratch_eval_instrumentation_v3',
        'distributed_runtime': runtime.describe(),
        'source_root': str(source_root), 'config': str(config_path),
        'initialization': 'random; checkpoint loading forbidden', 'seed': int(seed),
        'eval_seed': fixed_seed, 'eval_before_training': initial_eval,
        'source_modules': origins,
        'source_sha256': {str(Path(path).relative_to(source_root)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in sorted(set(origins.values())) if Path(path).is_file()},
        'evaluation_adapter': 'fixed torch/Python/NumPy seed with isolated RNG; original epoch-dependent env seed offset cancelled',
        'evaluation_feasibility_source': f'environment_success_and_independent_{validation_problem}_route_validation',
        'independent_route_validation': True, 'per_instance_route_export': True,
        'evaluation_physics': 'authoritative directed time/energy matrices when available; otherwise D/v and D*c; original training/decode environment unchanged',
        'trajectory_distribution_metrics': 'original environment-only feasibility; independent validation applies to the selected minimum route',
        'monitoring_capabilities': {'original_train_csv': True, 'global_rank_metrics': context.enabled,
                                   'post_update_kl': False, 'plugin_gradient_diagnostics': False},
        'training_model_optimizer_loss_environment': 'unchanged archived model/loss/environment implementation; external gradient synchronization and constant-zero auxiliary backward guard when distributed; optional explicit PPO warmup phase gates',
        'checkpoint_selection': 'unchanged original trainer; epoch zero is not eligible',
        'ppo_warmup_schedule': dict(epochs=warmup_schedule.epochs, configured_method=warmup_schedule.method,
                                   helper_sha256=hashlib.sha256(Path(_warmup.__file__).read_bytes()).hexdigest(),
                                   transition='same model and optimizer; no checkpoint reload/reset',
                                   adapter='external epoch predicate/auxiliary gates; archived source files unchanged'),
        'state': 'starting', 'completed_eval_epochs': [],
    }
    _write_metadata(metadata_path, metadata)
    print('[OriginalScratchAdapter] ' + json.dumps(metadata, sort_keys=True), flush=True)
    captured = {}
    original_agent = trainer.Agent
    original_evaluate = trainer.evaluate_fixed_dataset
    original_rollout_eval = trainer._rollout_eval_batch_min_median
    original_scale = trainer._configure_dataset_reward_scale
    original_writer = csv.DictWriter
    load_functions = {name: getattr(trainer, name) for name in ('_load_agent_checkpoint', '_load_training_checkpoint') if hasattr(trainer, name)}

    class CapturedAgent(original_agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if 'agent' in captured:
                raise RuntimeError('Scratch adapter expected exactly one original policy; reference policies are unsupported')
            captured['agent'] = self
            runtime.model_ready(self, trainer.set_seed)

    def configure_scale(runtime_cfg, pool):
        result = original_scale(runtime_cfg, pool)
        captured['runtime_cfg'] = runtime_cfg
        return result

    def evaluate_primary(agent, runtime_cfg, seed, epoch, device):
        del seed
        problem = trainer.problem_type_from_config(runtime_cfg)
        if problem not in ('vrptw', 'cvrptw', 'evrptw'):
            raise ValueError('Original measurement adapter supports VRPTW and EVRPTW evaluation')
        if 'validation_sidecar' not in captured:
            captured['validation_sidecar'] = _load_validation_sidecar(trainer, runtime_cfg)
        captured['evaluation_rows'] = []
        captured['eval_epoch'] = int(epoch)
        modes = [(module, module.training) for module in agent.modules()]
        started = time.perf_counter()
        try:
            with isolated_evaluation_rng(fixed_seed, device):
                # f388343 adds epoch * 1,000,000 before env.reset; cancel it.
                row = original_evaluate(agent, runtime_cfg, seed=fixed_seed - int(epoch) * 1_000_000,
                                        epoch=epoch, device=device)
        finally:
            for module, mode in modes:
                module.training = mode
        rows = captured.pop('evaluation_rows')
        output_dir = trainer._resolve_path(runtime_cfg.get('evaluation', {}).get('eval_output_dir'))
        if output_dir is None:
            output_dir = config_path.parent / 'evaluations'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'epoch_{int(epoch):04d}.jsonl'
        temporary = output_path.with_name('.' + output_path.name + '.tmp')
        with temporary.open('w') as handle:
            for instance_row in rows:
                handle.write(json.dumps(_json_finite(instance_row), allow_nan=False) + '\n')
        temporary.replace(output_path)
        row.update(eval_seed=fixed_seed, eval_rng_isolated=True,
                   eval_feasibility_source=metadata['evaluation_feasibility_source'],
                   eval_environment_feasible_rate=float(np.mean([item['environment_feasible'] for item in rows])) if rows else float('nan'),
                   eval_independent_valid_rate=float(np.mean([item['independently_valid'] for item in rows])) if rows else float('nan'),
                   eval_instance_results_path=str(output_path))
        metadata['completed_eval_epochs'].append(int(epoch))
        metadata['latest_eval_wall_time_s'] = time.perf_counter() - started
        metadata['latest_eval'] = dict(epoch=int(epoch), **row)
        _write_metadata(metadata_path, metadata)
        return row

    def evaluate(agent, runtime_cfg, seed, epoch, device):
        if not context.enabled:
            return evaluate_primary(agent, runtime_cfg, seed, epoch, device)
        packet = None
        if context.is_primary:
            try:
                packet = {'row': evaluate_primary(agent, runtime_cfg, seed, epoch, device)}
            except Exception as error:
                # Deliver the failure instead of leaving the other rank waiting
                # until the long validation collective timeout expires.
                packet = {'error': f'{type(error).__name__}: {error}'}
        packet = context.broadcast_object(packet)
        if 'error' in packet:
            raise RuntimeError('Original primary evaluation failed: ' + packet['error'])
        if not context.is_primary:
            metadata['completed_eval_epochs'].append(int(epoch))
            metadata['latest_eval'] = dict(epoch=int(epoch), **packet['row'])
            metadata['evaluation_executed_on_rank'] = 0
            _write_metadata(metadata_path, metadata)
        return packet['row']

    def rollout_eval(agent, envs, *args, **kwargs):
        rows = original_rollout_eval(agent, envs, *args, **kwargs)
        if len(rows) != len(envs):
            raise RuntimeError('Original evaluation did not return one row per environment')
        for row, wrapped in zip(rows, envs):
            env = wrapped.unwrapped
            success = env.terminated & (env.served_customers == env.num_customers) & (env.last == 0)
            index = _validation.select_original_best_index(
                env.objective_distance_km, success, env.served_customers)
            routes = env.get_routes()
            instance = env.instance
            row.update(instance_id=str(instance.instance_id), epoch=captured['eval_epoch'],
                       eval_seed=fixed_seed, selected_trajectory_index=index,
                       routes=routes[index] if index is not None else [],
                       environment_feasible=bool(row['feasible']),
                       selected_trajectory_complete=bool(success[index]) if index is not None else False)
            # Old get_routes appends a depot to an unfinished current route.
            # Export the actual prefix, not a synthetic unexecuted depot return.
            if index is not None and env.route_has_customer[index] and env.current_routes[index]:
                row['routes'][-1] = list(env.current_routes[index])
            raw = captured['validation_sidecar'].get(str(instance.instance_id))
            if validation_problem == 'evrptw':
                result = _validation.validate_evrptw_route(
                    instance, row, raw_payload=raw,
                    charging_mode=captured['runtime_cfg'].get('env', {}).get('charging_mode', 'fixed_full'))
            else:
                result = _validation.validate_vrptw_route(instance, row, raw_payload=raw)
            row.update(route_validation=result, independently_valid=bool(result['valid']),
                       feasible=bool(row['environment_feasible'] and result['valid']),
                       feasibility_source=metadata['evaluation_feasibility_source'])
            captured['evaluation_rows'].append(row)
        return rows

    class InstrumentedWriter(original_writer):
        def __init__(self, handle, fieldnames, *args, **kwargs):
            fields = list(fieldnames)
            self._is_train = 'policy_loss' in fields and 'reward_mean' in fields
            if self._is_train and warmup_schedule.epochs:
                fields.extend(key for key in _warmup.PHASE_FIELDS if key not in fields)
            if self._is_train and context.enabled:
                fields.extend(key for key in _distributed.EXTRA_TRAIN_FIELDS if key not in fields)
            self._is_eval = ('eval_status' in fields and 'eval_avg_objective_distance_km' in fields
                             and 'policy_loss' not in fields and 'reward_mean' not in fields)
            if self._is_eval:
                for key in ('eval_seed', 'eval_rng_isolated', 'eval_feasibility_source',
                            'eval_environment_feasible_rate', 'eval_independent_valid_rate',
                            'eval_instance_results_path'):
                    if key not in fields:
                        fields.append(key)
            self._handle = handle
            super().__init__(handle, fields, *args, **kwargs)

        def writerow(self, row):
            if self._is_train and context.enabled and isinstance(row.get('epoch'), (int, np.integer)):
                row = dict(row)
                row.update(runtime.finish_epoch(row))
            if self._is_train and warmup_schedule.epochs and isinstance(row.get('epoch'), (int, np.integer)):
                row = warmup_runtime.finish_epoch(trainer, row, seed)
            return super().writerow(row)

        def writeheader(self):
            result = super().writeheader()
            if self._is_eval and initial_eval:
                if captured.get('initial_eval_written'):
                    raise RuntimeError('Original epoch-zero evaluation was requested twice')
                if 'agent' not in captured or 'runtime_cfg' not in captured:
                    raise RuntimeError('Cannot instrument original epoch zero before finalized model/data initialization')
                row = evaluate(captured['agent'], captured['runtime_cfg'], seed=seed, epoch=0, device=device)
                self.writerow({'epoch': 0, **row})
                self._handle.flush()
                captured['initial_eval_written'] = True
            return result

    def forbid_checkpoint(*args, **kwargs):
        raise RuntimeError('Original scratch adapter forbids every initialization/resume/reference checkpoint load')

    runtime.install(trainer)
    warmup_runtime.install(trainer)
    trainer.Agent = CapturedAgent
    trainer._configure_dataset_reward_scale = configure_scale
    trainer.evaluate_fixed_dataset = evaluate
    trainer._rollout_eval_batch_min_median = rollout_eval
    csv.DictWriter = InstrumentedWriter
    for name in load_functions:
        setattr(trainer, name, forbid_checkpoint)
    started = time.perf_counter()
    try:
        metadata['state'] = 'running'
        checkpoint = trainer.train_from_config(cfg, seed=int(seed), device=device)
        assert_project_origins(source_root)
        final_difference = runtime.finalize()
        if context.enabled:
            checkpoint = runtime.last_checkpoint_path
        if initial_eval and not captured.get('initial_eval_written'):
            raise RuntimeError('Original trainer completed without required epoch-zero evaluation')
        metadata.update(state='completed', checkpoint_final=str(checkpoint),
                        expert_observation_storage=dict(runtime.expert_storage),
                        final_parameter_max_abs_difference=final_difference,
                        optimizer_steps=context.optimizer_steps if context.enabled else None,
                        amp_skipped_steps=context.amp_skipped_steps if context.enabled else None,
                        adapter_wall_time_s=time.perf_counter() - started)
        _write_metadata(metadata_path, metadata)
        print(f'Saved original from-scratch final checkpoint: {checkpoint}', flush=True)
        return Path(checkpoint)
    except BaseException as error:
        metadata.update(state='failed', error=f'{type(error).__name__}: {error}',
                        adapter_wall_time_s=time.perf_counter() - started)
        _write_metadata(metadata_path, metadata)
        raise
    finally:
        trainer.Agent = original_agent
        trainer._configure_dataset_reward_scale = original_scale
        trainer.evaluate_fixed_dataset = original_evaluate
        trainer._rollout_eval_batch_min_median = original_rollout_eval
        csv.DictWriter = original_writer
        for name, function in load_functions.items():
            setattr(trainer, name, function)
        warmup_runtime.uninstall(trainer)
        runtime.uninstall(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_original_scratch(args.source_root, args.config, args.seed, args.device)


if __name__ == '__main__':
    main()
