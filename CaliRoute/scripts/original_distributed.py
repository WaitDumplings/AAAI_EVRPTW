"""External synchronous runtime for unmodified f388343 model/loss sources.

This is an explicitly declared port of optimizer-boundary synchronization, not
native historical DDP and not an assertion of single-process bitwise equivalence.
Only the standalone communication utility is loaded from the current checkout;
no current trainer, model, environment, or data adapter is imported.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

_SUPPORT_PATH = Path(__file__).resolve().parents[1] / 'offline2online/distributed.py'
_spec = importlib.util.spec_from_file_location('_original_standalone_sync_support', _SUPPORT_PATH)
support = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = support
_spec.loader.exec_module(support)
_storage_spec = importlib.util.spec_from_file_location(
    '_original_static_expert_storage', Path(__file__).with_name('original_distributed_storage.py'))
_storage = importlib.util.module_from_spec(_storage_spec)
_storage_spec.loader.exec_module(_storage)


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


class OriginalDistributedRuntime:
    def __init__(self, cfg, seed, device, output_dir, context=None):
        self.cfg, self.seed = cfg, int(seed)
        # Let torchrun terminate peers when a worker/collective fails. Evaluation
        # errors are explicitly broadcast; NCCL errors must also fail the job.
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
        self.context = context or support.DistributedContext.initialize(
            device, timeout_minutes=int(cfg.get('training', {}).get('distributed_timeout_minutes', 180)))
        self.device = self.context.device
        self.sampling_seed = support.rank_seed(self.seed, self.context.rank)
        self.output_dir = Path(output_dir)
        self.originals = {}
        self.pool = self.expert = self.agent = self.scaler = self.optimizer = None
        self.policy_best = None
        self.epoch = 0
        self.attempted_steps = self.epoch_attempted_steps = 0
        self.empty_gradient_steps = 0
        self.sampled_ids = []
        self.last_epoch_metrics = {}
        self.last_checkpoint_path = None
        self.started = time.perf_counter()
        self.final_max_difference = None
        self.primary_run_name = str(cfg.get('run_name', 'O2O_TERRAN_FULL'))
        self.share_static = bool(cfg.get('offline', {}).get('original_share_static_expert_observations', False))
        self.expert_storage = {'enabled': self.share_static, 'state': 'not_loaded'}
        if self.context.enabled:
            if not cfg.get('data', {}).get('train_dataset_path'):
                raise ValueError('Original distributed adapter requires a fixed training dataset')
            if int(cfg.get('offline', {}).get('bc_warmup_epochs', 0)) > 0:
                raise ValueError('Original distributed scratch adapter does not support BC-only warmup epochs')
            if not self.context.is_primary:
                # Original code writes best_checkpoint.json directly, outside its
                # save_checkpoint function. A separate run subpath prevents races.
                cfg['run_name'] = f'{self.primary_run_name}/rank_{self.context.rank}'

    def describe(self):
        return {
            'enabled': self.context.enabled, 'rank': self.context.rank,
            'world_size': self.context.world_size, 'sampling_seed': self.sampling_seed,
            'native_historical_ddp': False,
            'runtime': 'external optimizer-boundary synchronous gradient averaging' if self.context.enabled else 'original single-process runtime',
            'objective': 'mean_of_rank_local_masked_objectives; not concatenated-global valid-action/route normalization',
            'expert_and_memory_scope': 'rank-local expert replay RNG, priority state and incumbent memory; no cross-rank archive merge',
            'sampling_scope': 'independent rank seeds over the full train pool; overlap measured, not guaranteed disjoint',
            'checkpoint_scope': 'rank0 model/optimizer checkpoint plus all-rank diagnostic state; exact resume unsupported and checkpoint loading forbidden',
            'sync_helper': str(_SUPPORT_PATH),
            'sync_helper_sha256': hashlib.sha256(_SUPPORT_PATH.read_bytes()).hexdigest(),
            'expert_storage': dict(self.expert_storage),
            'empty_auxiliary_losses': 'constant zero auxiliary losses skip backward; optimizer boundaries still synchronize',
            'non_amp_nonfinite_gradients': 'all ranks skip the invalid optimizer step',
        }

    def model_ready(self, agent, set_seed):
        self.agent = agent
        if self.context.enabled:
            # CapturedAgent.__init__ runs before the original outer .to(device).
            agent.to(self.device)
            self.context.broadcast_module(agent)
            set_seed(self.sampling_seed)

    def optimizer_step(self, original_step, optimizer, agent, max_grad_norm, scaler, amp_enabled):
        if not self.context.enabled:
            return original_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
        self.optimizer, self.scaler = optimizer, scaler
        self.attempted_steps += 1
        self.epoch_attempted_steps += 1
        if amp_enabled and scaler is not None:
            # A rank with only remotely contributed gradients may not yet have
            # initialized its scaler. Use the public API before checking parity.
            scaler.scale(torch.zeros((), device=self.device))
            scales = self.context.gather_objects(float(scaler.get_scale()))
            if len(set(scales)) != 1:
                raise RuntimeError(f'Original distributed AMP scales diverged: {scales}')
        if not self.context.synchronize_gradients(agent):
            self.empty_gradient_steps += 1
            return
        gradients = [parameter.grad for parameter in agent.parameters() if parameter.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([gradient.detach().float().norm() for gradient in gradients]))
        scale_before = float(scaler.get_scale()) if amp_enabled and scaler is not None else 1.
        grad_norm = float(norm.item()) / scale_before
        if not amp_enabled and not np.isfinite(grad_norm):
            skipped = True
        else:
            # The ORIGINAL unscale, clip, AdamW step, and scaler update run here.
            # Overflow already propagated through averaged scaled gradients.
            original_step(optimizer, agent, max_grad_norm, scaler, amp_enabled)
            skipped = bool(amp_enabled and scaler is not None and scaler.get_scale() < scale_before)
        self.context.record_step(grad_norm, skipped, max_grad_norm=max_grad_norm)

    def _patch(self, trainer, name, value):
        self.originals[name] = getattr(trainer, name)
        setattr(trainer, name, value)

    def install(self, trainer):
        runtime = self
        original_expert = trainer._load_expert_buffer

        def load_expert(cfg, seed, *args, **kwargs):
            storage = _storage.StaticExpertStorage() if runtime.share_static else None
            offline_data = sys.modules['offline2online.offline_data']
            with storage.instrument(offline_data) if storage is not None else nullcontext():
                runtime.expert = original_expert(
                    cfg, support.rank_seed(seed, runtime.context.rank) if runtime.context.enabled else seed,
                    *args, **kwargs)
            runtime.expert_storage = storage.metrics() if storage is not None else {'enabled': False, 'state': 'original_copies'}
            if storage is not None:
                print('[OriginalExpertStorage] ' + json.dumps(dict(rank=runtime.context.rank, **runtime.expert_storage)), flush=True)
            return runtime.expert

        if self.share_static or self.context.enabled:
            self._patch(trainer, '_load_expert_buffer', load_expert)
        if not self.context.enabled:
            return
        original_pool = trainer.AdaptedFixedDatasetInstancePool
        original_priority = trainer.SolutionPrioritySampler
        original_step = trainer._optimizer_step
        original_backward = trainer._backward
        original_rollout = trainer.collect_rollout
        original_candidates = trainer._prepare_sl_expert_candidates
        original_save = trainer.save_checkpoint

        class RankedPool(original_pool):
            def __post_init__(self):
                self.seed = support.rank_seed(self.seed, runtime.context.rank)
                super().__post_init__()
                runtime.pool = self

        class RankedPriority(original_priority):
            def __init__(self, *args, seed, **kwargs):
                super().__init__(*args, seed=support.rank_seed(seed, runtime.context.rank), **kwargs)
                runtime.pool = self

        def step(*args, **kwargs):
            return runtime.optimizer_step(original_step, *args, **kwargs)

        def backward(loss, scaler, amp_enabled):
            if loss.requires_grad:
                original_backward(loss, scaler, amp_enabled)

        def rollout(agent, envs, *args, **kwargs):
            runtime.epoch += 1
            runtime.context.reset_epoch()
            runtime.epoch_attempted_steps = 0
            runtime.epoch_started = time.perf_counter()
            if str(runtime.device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(runtime.device)
            if kwargs.get('seed') is not None:
                kwargs['seed'] = support.rank_seed(kwargs['seed'], runtime.context.rank)
            batch = original_rollout(agent, envs, *args, **kwargs)
            runtime.sampled_ids = [str(env.unwrapped.instance.instance_id) for env in envs]
            return batch

        def candidates(agent, batch, cfg, envs, expert_buffer, policy_best_objectives, device):
            runtime.policy_best = policy_best_objectives
            return original_candidates(agent, batch, cfg, envs, expert_buffer, policy_best_objectives, device)

        def save(path, agent, optimizer, cfg, epoch, seed):
            # Scheduled/final calls are unconditional; best calls are consistent
            # because every rank receives the same primary validation row.
            states = runtime.context.gather_objects(runtime.capture_state())
            primary_path = runtime.context.broadcast_object(str(path) if runtime.context.is_primary else None)
            runtime.last_checkpoint_path = primary_path
            if runtime.context.is_primary:
                original_save(path, agent, optimizer, cfg, epoch, seed)
                companion = Path(path).with_suffix('.distributed.pt')
                temporary = companion.with_name('.' + companion.name + '.tmp')
                torch.save({'epoch': int(epoch), 'world_size': runtime.context.world_size,
                            'exact_resume_supported': False, 'runtime': runtime.describe(), 'ranks': states}, temporary)
                temporary.replace(companion)

        for name, value in [('AdaptedFixedDatasetInstancePool', RankedPool), ('SolutionPrioritySampler', RankedPriority),
                            ('_optimizer_step', step), ('_backward', backward), ('collect_rollout', rollout),
                            ('_prepare_sl_expert_candidates', candidates),
                            ('save_checkpoint', save)]:
            self._patch(trainer, name, value)

    def capture_state(self):
        state = {'rank': self.context.rank, 'sampling_seed': self.sampling_seed,
                 'rng': support.capture_rng_state(self.device),
                 'optimizer_steps': self.context.optimizer_steps, 'amp_skipped_steps': self.context.amp_skipped_steps,
                 'attempted_optimizer_steps': self.attempted_steps,
                 'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                 'sampled_instance_ids': list(self.sampled_ids),
                 'last_epoch_metrics': dict(self.last_epoch_metrics),
                 'policy_best_objectives': copy.deepcopy(self.policy_best),
                 'exact_resume_supported': False, 'expert_observation_storage': dict(self.expert_storage)}
        if self.pool is not None:
            state['pool'] = {'class': type(self.pool).__name__, 'rng': copy.deepcopy(self.pool.rng.bit_generator.state),
                             'attributes': {key: copy.deepcopy(getattr(self.pool, key)) for key in support._POOL_KEYS if hasattr(self.pool, key)}}
        if self.expert is not None:
            state['expert_rng'] = copy.deepcopy(self.expert.rng.bit_generator.state)
        return state

    def finish_epoch(self, row):
        if not self.context.enabled:
            return {}
        epoch = int(row['epoch'])
        if epoch != self.epoch:
            raise RuntimeError(f'Original distributed epoch/rollout mismatch: {epoch} vs {self.epoch}')
        train = self.cfg['training']
        envs = int(train['num_envs_per_gpu'])
        minibatches = max(1, min(int(train.get('num_minibatches', 4)), envs))
        accumulation = max(1, int(train.get('gradient_accumulation_steps', 1)))
        expected_attempts = int(train.get('ppo_update_epochs', 3)) * ((minibatches + accumulation - 1) // accumulation)
        counts = self.context.gather_objects(self.epoch_attempted_steps)
        if any(count != expected_attempts for count in counts):
            raise RuntimeError(f'Original optimizer boundary count mismatch: {counts}; expected {expected_attempts}')
        metrics = self.context.metrics(
            num_envs=envs, n_traj=int(train['n_traj']),
            effective_instances=envs / minibatches * min(accumulation, minibatches),
            train_seconds=max(float(row.get('epoch_wall_time_s', 0.)) - float(row.get('eval_wall_time_s', 0.)), 1e-9),
            learning_rate=float(self.optimizer.param_groups[0]['lr']), samples_seen=int(row.get('samples_seen', 0)))
        metrics.update(support.rollout_instance_coverage(self.context, self.sampled_ids))
        metrics.update(support.parameter_sync_diagnostics(self.context, self.agent, self.scaler))
        for key in ('parameter_sync_checksum_max_diff', 'optimizer_steps_rank_max_diff', 'amp_skipped_steps_rank_max_diff', 'amp_scale_rank_max_diff'):
            if metrics[key] != 0:
                raise RuntimeError(f'Original distributed state diverged: {key}={metrics[key]}')
        metrics.update(sampling_seed=self.sampling_seed, optimizer_attempts_epoch=self.epoch_attempted_steps,
                       optimizer_attempts=self.attempted_steps, empty_gradient_steps=self.empty_gradient_steps,
                       original_external_distributed_runtime=True,
                       rank_local_auxiliary_memory=True, sampled_instance_ids=list(self.sampled_ids),
                       run_elapsed_seconds=time.perf_counter() - self.started,
                       expert_observation_storage=dict(self.expert_storage))
        if str(self.device).startswith('cuda'):
            metrics.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                           gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        self.last_epoch_metrics = metrics
        monitor = Path(train.get('monitor_output_dir') or self.output_dir / 'monitoring')
        monitor.mkdir(parents=True, exist_ok=True)
        with (monitor / f'monitor_rank_{self.context.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(_finite_json(dict(row, **metrics)), allow_nan=False) + '\n')
        return metrics

    def finalize(self):
        if self.context.enabled:
            # Exact elementwise final check, stronger than the inexpensive epoch
            # checksums. One model-sized buffer; no extra decoder/sample budget.
            maximum = torch.zeros((), device=self.device)
            for parameter in self.agent.parameters():
                primary = parameter.detach().clone()
                dist.broadcast(primary, src=0)
                maximum = torch.maximum(maximum, (parameter.detach() - primary).abs().max())
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            self.final_max_difference = float(maximum.item())
            if not np.isfinite(self.final_max_difference) or self.final_max_difference != 0:
                raise RuntimeError(f'Original final model parameters differ across ranks: {self.final_max_difference}')
            self.context.barrier()
        return self.final_max_difference

    def uninstall(self, trainer):
        for name, original in self.originals.items():
            setattr(trainer, name, original)
        self.context.close()


EXTRA_TRAIN_FIELDS = list(support.DISTRIBUTED_TRAIN_FIELDS) + [
    'sampling_seed', 'optimizer_attempts_epoch', 'optimizer_attempts', 'empty_gradient_steps',
    'original_external_distributed_runtime', 'rank_local_auxiliary_memory',
    'global_unique_instances_per_rollout', 'global_duplicate_instances_per_rollout',
    'parameter_sync_checksum_max_diff', 'optimizer_steps_rank_max_diff',
    'amp_skipped_steps_rank_max_diff', 'amp_scale_rank_max_diff', 'amp_scale',
    'run_elapsed_seconds', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes',
]
