"""Explicit synchronous data parallelism for policies with custom cached forwards.

Only optimizer boundaries communicate: rollout lengths and expert candidate counts
may differ by rank. Gradients define the *mean of rank-local masked objectives*,
not the mean over a concatenated set of variable-count valid steps/routes. Every
rank has the same number of environments and optimizer boundaries. This preserves
the existing per-rank loss normalization, including trajectory/chunk weighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from collections import deque
import copy
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def rank_seed(seed: int, rank: int) -> int:
    return (int(seed) + 1_000_003 * int(rank)) % (2**32)


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    owns_process_group: bool = False
    optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    epoch_optimizer_steps: int = 0
    epoch_amp_skipped_steps: int = 0
    epoch_gradient_sync_sec: float = 0.0
    epoch_grad_norms: list[float] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, requested_device=None, timeout_minutes: int = 30):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available() and (requested_device is None or str(requested_device).startswith("cuda"))
        device = f"cuda:{local_rank}" if use_cuda and world_size > 1 else str(requested_device or ("cuda" if use_cuda else "cpu"))
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        owns = False
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=timeout_minutes))
            owns = True
        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        return cls(rank=rank, world_size=world_size, local_rank=local_rank, device=device, owns_process_group=owns)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def broadcast_module(self, module):
        if self.enabled:
            for value in module.state_dict().values():
                if torch.is_tensor(value):
                    dist.broadcast(value, src=0)

    def broadcast_object(self, obj):
        objects = [obj if self.is_primary else None]
        if self.enabled:
            dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def gather_objects(self, obj):
        if not self.enabled:
            return [obj]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def synchronize_gradients(self, module, bucket_bytes: int = 25 * 1024 * 1024) -> bool:
        """Average scaled gradients, including ranks with locally unused parameters.

        All ranks must call exactly once per optimizer boundary. A globally unused
        parameter stays grad=None so AdamW does not update/decay it. Synchronizing
        before AMP unscale propagates overflow to every rank and keeps scaler state
        and skipped optimizer steps consistent without private GradScaler internals.
        """
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return False
        if not self.enabled:
            return any(p.grad is not None for p in params)
        # Separate queued backward kernels from synchronization wall time.
        if params[0].is_cuda:
            torch.cuda.synchronize(params[0].device)
        started = time.perf_counter()
        present = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.int32)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        active = present.cpu().tolist()
        bucket, size = [], 0

        def flush():
            if not bucket:
                return
            flattened = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
            dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
            flattened.div_(self.world_size)
            offset = 0
            for p in bucket:
                averaged = flattened[offset:offset + p.numel()].view_as(p)
                if p.grad is None:
                    p.grad = averaged.clone()
                else:
                    p.grad.copy_(averaged)
                offset += p.numel()

        for p, used in zip(params, active):
            if not used:
                p.grad = None
                continue
            if bucket and (size + p.numel() * p.element_size() > bucket_bytes or p.dtype != bucket[0].dtype or p.device != bucket[0].device):
                flush()
                bucket, size = [], 0
            bucket.append(p)
            size += p.numel() * p.element_size()
        flush()
        # Include completed NCCL work, not just host enqueue latency.
        if params[0].is_cuda:
            torch.cuda.synchronize(params[0].device)
        self.epoch_gradient_sync_sec += time.perf_counter() - started
        return any(active)

    def reset_epoch(self):
        self.epoch_optimizer_steps = 0
        self.epoch_amp_skipped_steps = 0
        self.epoch_gradient_sync_sec = 0.0
        self.epoch_grad_norms.clear()

    def record_step(self, grad_norm: float, skipped: bool):
        self.epoch_grad_norms.append(float(grad_norm))
        if skipped:
            self.amp_skipped_steps += 1
            self.epoch_amp_skipped_steps += 1
        else:
            self.optimizer_steps += 1
            self.epoch_optimizer_steps += 1

    def metrics(self, *, num_envs, n_traj, effective_instances, train_seconds, learning_rate, samples_seen):
        # Epoch throughput uses the slowest rank, not rank zero's clock alone.
        elapsed = torch.tensor(float(train_seconds), device=self.device, dtype=torch.float64)
        samples = torch.tensor(int(samples_seen), device=self.device, dtype=torch.int64)
        if self.enabled:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(samples, op=dist.ReduceOp.SUM)
        seconds = max(float(elapsed.item()), 1e-9)
        return {
            "distributed_rank": self.rank,
            "distributed_world_size": self.world_size,
            "distributed_objective": "mean_of_rank_local_masked_objectives",
            "global_num_envs": int(num_envs * self.world_size),
            "global_samples_seen": int(samples.item()),
            "global_trajectories_per_rollout": int(num_envs * n_traj * self.world_size),
            "global_effective_instances_per_optimizer_step": float(effective_instances * self.world_size),
            "optimizer_steps": self.optimizer_steps,
            "optimizer_steps_epoch": self.epoch_optimizer_steps,
            "amp_skipped_steps": self.amp_skipped_steps,
            "amp_skipped_steps_epoch": self.epoch_amp_skipped_steps,
            "grad_norm": float(np.mean(self.epoch_grad_norms)) if self.epoch_grad_norms else 0.0,
            "grad_norm_max": float(np.max(self.epoch_grad_norms)) if self.epoch_grad_norms else 0.0,
            "learning_rate": float(learning_rate),
            "distributed_gradient_sync_sec": self.epoch_gradient_sync_sec,
            "global_instances_per_sec": num_envs * self.world_size / seconds,
            "global_trajectories_per_sec": num_envs * n_traj * self.world_size / seconds,
            "distributed_train_wall_time_s": seconds,
        }

    def close(self):
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


DISTRIBUTED_TRAIN_FIELDS = list(DistributedContext().metrics(num_envs=1, n_traj=1, effective_instances=1, train_seconds=1, learning_rate=0, samples_seen=0))


def capture_rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if str(device).startswith("cuda") else None}


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and str(device).startswith("cuda"):
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


_POOL_KEYS = ("order", "cursor", "sample_count", "priority", "gap_ema", "vehicle_gap_ema", "last_update_epoch", "num_updates", "current_epoch", "_buffer", "_last_sample_indices", "_last_sample_stats")
_SUPPORTED_POOLS = {"AdaptedFixedDatasetInstancePool", "SolutionPrioritySampler"}


def capture_local_training_state(context, pool, expert_buffer, route_pool, policy_best_objectives, scaler, sample_count_offset=0):
    supported = type(pool).__name__ in _SUPPORTED_POOLS
    sampler = {"class": type(pool).__name__, "supported": supported}
    if supported:
        sampler["attributes"] = {key: copy.deepcopy(getattr(pool, key)) for key in _POOL_KEYS if hasattr(pool, key)}
        sampler["rng"] = copy.deepcopy(pool.rng.bit_generator.state)
    return {
        "rank": context.rank, "rng": capture_rng_state(context.device), "sampler": sampler,
        "expert_rng": copy.deepcopy(expert_buffer.rng.bit_generator.state) if expert_buffer is not None else None,
        "policy_route_pool": route_pool.state_dict() if route_pool is not None else None,
        "policy_best_objectives": dict(policy_best_objectives),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "optimizer_steps": context.optimizer_steps, "amp_skipped_steps": context.amp_skipped_steps,
        "sample_count_offset": int(sample_count_offset),
    }


def restore_local_training_state(state, context, pool, expert_buffer, route_pool, policy_best_objectives, scaler):
    sampler = state["sampler"]
    if sampler.get("supported") and sampler["class"] == type(pool).__name__:
        for key, value in sampler["attributes"].items():
            setattr(pool, key, copy.deepcopy(value))
        pool.rng.bit_generator.state = sampler["rng"]
    if expert_buffer is not None and state.get("expert_rng") is not None:
        expert_buffer.rng.bit_generator.state = state["expert_rng"]
    if route_pool is not None and state.get("policy_route_pool") is not None:
        route_pool.load_state_dict(state["policy_route_pool"])
    policy_best_objectives.update(state.get("policy_best_objectives", {}))
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    context.optimizer_steps = int(state.get("optimizer_steps", 0))
    context.amp_skipped_steps = int(state.get("amp_skipped_steps", 0))
    restore_rng_state(state["rng"], context.device)
    return int(state.get("sample_count_offset", 0))


def rollout_instance_coverage(context, local_ids):
    """Small epoch-boundary gather; duplicates count repeated draws, not pairs."""
    per_rank = context.gather_objects(list(local_ids))
    known = [str(instance_id) for ids in per_rank for instance_id in ids if instance_id not in (None, '')]
    unique = len(set(known))
    return {
        'global_unique_instances_per_rollout': unique,
        'global_duplicate_instances_per_rollout': len(known) - unique,
        'global_instance_ids_observed': len(known),
        'global_instance_ids_missing': sum(map(len, per_rank)) - len(known),
    }


@torch.no_grad()
def parameter_sync_diagnostics(context, agent, scaler):
    """Low-frequency per-parameter checksums, not an elementwise error bound."""
    moments = []
    for parameter in agent.parameters():
        value = parameter.detach().float()
        moments.extend((value.sum(), value.square().sum()))
    checksum = torch.stack(moments) if moments else torch.empty(0, device=context.device)
    scale = float(scaler.get_scale()) if scaler is not None else 1.0
    counters = checksum.new_tensor([float(context.optimizer_steps), float(context.amp_skipped_steps), scale])
    vector = torch.cat((checksum, counters))
    gathered = [torch.empty_like(vector) for _ in range(context.world_size)]
    if context.enabled:
        dist.all_gather(gathered, vector)
    else:
        gathered[0].copy_(vector)
    stacked = torch.stack(gathered)
    difference = stacked.amax(0) - stacked.amin(0)
    values = torch.stack((difference[:-3].amax() if moments else vector.new_zeros(()), *difference[-3:])).cpu().tolist()
    return {
        'parameter_sync_checksum_max_diff': values[0],
        'optimizer_steps_rank_max_diff': values[1],
        'amp_skipped_steps_rank_max_diff': values[2],
        'amp_scale_rank_max_diff': values[3],
        'amp_scale': scale,
        'parameter_sync_scope': 'per-parameter FP32 sum and squared-norm checksums; not a full elementwise parameter comparison',
    }
