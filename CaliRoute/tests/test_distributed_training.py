from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from offline2online.distributed import DistributedContext, rank_seed, capture_local_training_state, restore_local_training_state


class MaskedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.optional = torch.nn.Parameter(torch.tensor([.5], dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    context = DistributedContext(rank=rank, world_size=2, device='cpu')
    model = MaskedModel()
    with torch.no_grad():
        model.weight.add_(rank * 10)
    context.broadcast_module(model)
    assert torch.equal(model.weight, torch.tensor([1., -2.], dtype=torch.float64))
    x = torch.tensor([[1., 2.], [2., 1.], [3., -1.]], dtype=torch.float64)
    mask = torch.tensor([True, False, False]) if rank == 0 else torch.tensor([True, True, True])
    loss = (x[mask] @ model.weight).square().mean()
    if rank == 1:
        loss = loss + model.optional.square().sum()
    loss.backward()
    assert context.synchronize_gradients(model, bucket_bytes=8)
    reference = MaskedModel()
    explicit_objective = ((x[:1] @ reference.weight).square().mean() + (x @ reference.weight).square().mean() + reference.optional.square().sum()) / 2
    explicit_objective.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=0, atol=1e-14)
    torch.testing.assert_close(model.optional.grad, reference.optional.grad, rtol=0, atol=1e-14)
    assert model.unused.grad is None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    optimizer.step()
    expected = torch.optim.AdamW(reference.parameters(), lr=.01)
    expected.step()
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-14)
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        model.weight.sum().backward()
    assert context.synchronize_gradients(model)
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, .5))
    assert model.optional.grad is None
    optimizer.zero_grad(set_to_none=True)
    assert not context.synchronize_gradients(model)
    from offline2online.trainer import _optimizer_step
    model._distributed_context = context
    scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    before = model.weight.detach().clone()
    if rank == 0:
        scaler.scale(model.weight.sum() * float('inf')).backward()
    _optimizer_step(optimizer, model, 1., scaler, True)
    assert torch.equal(model.weight, before)
    assert scaler.get_scale() == 64.
    assert context.amp_skipped_steps == 1
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        scaler.scale(model.weight.sum()).backward()
    _optimizer_step(optimizer, model, 1., scaler, True)
    weights = [torch.zeros_like(model.weight) for _ in range(2)]
    dist.all_gather(weights, model.weight.detach())
    assert torch.equal(weights[0], weights[1])
    assert context.optimizer_steps == 1
    assert not torch.equal(model.weight, before)
    ids = np.random.default_rng(rank_seed(3009, rank)).choice(5000, size=64, replace=False).tolist()
    gathered = context.gather_objects(ids)
    assert gathered[0] != gathered[1]
    assert context.broadcast_object({'evaluation': 123} if rank == 0 else None) == {'evaluation': 123}
    metrics = context.metrics(num_envs=64, n_traj=50, effective_instances=16, train_seconds=1+rank, learning_rate=1e-4, samples_seen=64)
    assert metrics['global_num_envs'] == 128
    assert metrics['global_trajectories_per_rollout'] == 6400
    assert metrics['global_effective_instances_per_optimizer_step'] == 32
    assert metrics['global_instances_per_sec'] == 64
    assert metrics['global_samples_seen'] == 128
    from offline2online.distributed import rollout_instance_coverage, parameter_sync_diagnostics
    coverage = rollout_instance_coverage(context, ['shared', f'unique{rank}'])
    assert coverage['global_unique_instances_per_rollout'] == 3
    assert coverage['global_duplicate_instances_per_rollout'] == 1
    assert coverage['global_instance_ids_missing'] == 0
    checks = parameter_sync_diagnostics(context, model, scaler)
    assert checks['parameter_sync_checksum_max_diff'] == 0
    assert checks['optimizer_steps_rank_max_diff'] == 0
    assert checks['amp_scale_rank_max_diff'] == 0
    Path(output, f'rank{rank}.ok').write_text('ok')
    dist.destroy_process_group()


def test_two_rank_masked_objective_unused_parameters_zero_grad_and_amp(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / 'rank0.ok').exists()
    assert (tmp_path / 'rank1.ok').exists()


def test_rank_seed_is_reproducible_and_distinct():
    assert rank_seed(3009, 0) == 3009
    assert rank_seed(3009, 1) != rank_seed(3009, 0)
    assert rank_seed(3009, 1) == rank_seed(3009, 1)


def test_epoch_boundary_state_roundtrip_restores_rng_sampler_memory_and_scaler():
    from offline2online.instance_adapter import AdaptedFixedDatasetInstancePool
    from types import SimpleNamespace
    pool = AdaptedFixedDatasetInstancePool.__new__(AdaptedFixedDatasetInstancePool)
    pool.rng = np.random.default_rng(999)
    pool.order = np.arange(8)
    pool.cursor = 3
    pool.sample_count = 43
    expert = SimpleNamespace(rng=np.random.default_rng(1000))
    context = DistributedContext(optimizer_steps=72, amp_skipped_steps=2)
    random.seed(35)
    np.random.seed(36)
    torch.manual_seed(37)
    state = capture_local_training_state(context, pool, expert, None, {'one': 12.3}, None, 100)
    expected = (random.random(), np.random.random(), torch.rand(1), pool.rng.random(), expert.rng.random())
    pool.cursor = 0
    pool.sample_count = 0
    memory = {}
    assert restore_local_training_state(state, context, pool, expert, None, memory, None) == 100
    actual = (random.random(), np.random.random(), torch.rand(1), pool.rng.random(), expert.rng.random())
    for left, right in zip(expected, actual):
        assert left == right
    assert pool.cursor == 3
    assert pool.sample_count == 43
    assert memory == {'one': 12.3}
    assert context.optimizer_steps == 72


def test_only_primary_writes_atomic_checkpoint_with_resume_state(tmp_path):
    from offline2online.trainer import save_checkpoint, _load_training_checkpoint
    model = MaskedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    path = tmp_path / 'checkpoint.pt'
    model._distributed_context = DistributedContext(rank=1, world_size=2)
    save_checkpoint(path, model, optimizer, {}, epoch=7, seed=3009)
    assert not path.exists()
    model._distributed_context = DistributedContext(rank=0, world_size=2)
    model._training_resume_state = {'world_size': 2, 'ranks': [{'rng_numpy': np.random.get_state()}, {'rank': 1}]}
    save_checkpoint(path, model, optimizer, {}, epoch=7, seed=3009)
    assert path.exists()
    assert not path.with_name('.checkpoint.pt.tmp').exists()
    restored = MaskedModel()
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=.02)
    info = _load_training_checkpoint(restored, new_optimizer, path, 'cpu')
    assert info['epoch'] == 7 and info['optimizer_loaded']
    assert new_optimizer.param_groups[0]['lr'] == .01
    assert restored._pending_training_resume_state['world_size'] == 2
    assert restored._pending_training_resume_state['ranks'][1]['rank'] == 1


def test_gradient_monitor_counts_nonfinite_and_clipping_over_finite_attempts():
    context = DistributedContext()
    context.record_step(float('inf'), skipped=True, max_grad_norm=1.)
    context.record_step(2., skipped=False, max_grad_norm=1.)
    context.record_step(.5, skipped=False, max_grad_norm=1.)
    values = context.metrics(num_envs=2, n_traj=3, effective_instances=1, train_seconds=1, learning_rate=1e-4, samples_seen=2)
    assert values['grad_norm_nonfinite_count'] == 1
    assert values['grad_norm_finite_mean'] == 1.25
    assert values['grad_clipped_fraction'] == .5
    assert values['amp_skipped_steps_epoch'] == 1
    context.reset_epoch()
    assert context.epoch_finite_grad_clipped == []
