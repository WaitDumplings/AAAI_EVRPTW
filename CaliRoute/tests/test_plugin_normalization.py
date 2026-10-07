from __future__ import annotations

from datetime import timedelta
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from caliroute.plugins.normalization import (
    ActorAdvantageScale,
    MaskedMoments,
    ScalarPopArt,
    masked_moments,
    merge_moments,
)


def test_masked_moments_ignore_padding_and_match_unequal_chunk_merge():
    x = torch.tensor([[1., 3., float('nan')], [5., 7., float('inf')]], dtype=torch.float64)
    valid = torch.tensor([[True, True, False], [True, True, False]])
    whole = masked_moments(x, valid)
    parts = [masked_moments(x[:1, :1]), masked_moments(x[0, 1:2]), masked_moments(x[1], valid[1])]
    empty = masked_moments(x, torch.zeros_like(valid))
    merged = merge_moments([empty, *parts, empty])
    assert whole.count.item() == 4
    assert whole.mean.item() == 4
    assert whole.variance.item() == 5
    for field in ('count', 'mean', 'variance'):
        torch.testing.assert_close(getattr(whole, field), getattr(merged, field))


def test_centered_variance_retains_small_spread_around_large_offset():
    x = 1e12 + torch.arange(4, dtype=torch.float64)
    moments = masked_moments(x)
    assert moments.mean.item() == 1e12 + 1.5
    assert moments.variance.item() == 1.25


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_valid_nonfinite_rejected_without_partial_normalizer_update(invalid):
    value = torch.tensor([1., invalid])
    popart, actor, head = ScalarPopArt(), ActorAdvantageScale(), nn.Linear(2, 1)
    original_weight = head.weight.detach().clone()
    with pytest.raises(ValueError, match='finite'):
        popart.update(value, torch.ones_like(value, dtype=torch.bool), head)
    with pytest.raises(ValueError, match='finite'):
        actor.update(value)
    assert popart.update_count.item() == actor.update_count.item() == 0
    torch.testing.assert_close(head.weight, original_weight)


def test_empty_mask_does_not_update_counts_statistics_or_head():
    head = nn.Linear(2, 1)
    popart, actor = ScalarPopArt(), ActorAdvantageScale()
    targets = torch.tensor([float('nan'), float('inf')])
    mask = torch.zeros_like(targets, dtype=torch.bool)
    before = head.weight.detach().clone()
    popart.update(targets, mask, head)
    actor.update(targets, mask)
    assert popart.update_count.item() == actor.update_count.item() == 0
    assert popart.sample_count.item() == actor.sample_count.item() == 0
    assert popart.mean.item() == 0
    assert popart.std.item() == actor.snapshot().item() == 1
    torch.testing.assert_close(head.weight, before)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_popart_preserves_physical_value_on_first_and_later_updates(dtype):
    torch.manual_seed(114)
    head = nn.Linear(7, 1, dtype=dtype)
    x = torch.randn(11, 7, dtype=dtype)
    popart = ScalarPopArt(beta=.2, min_std=.01)
    expected = popart.denormalize(head(x)).detach()
    parameter_ids = [id(p) for p in head.parameters()]
    for targets in (torch.tensor([-100., -150., -200.]), torch.tensor([-1000., -250., 500.])):
        popart.update(targets, None, head)
        torch.testing.assert_close(popart.denormalize(head(x)), expected, atol=5e-5 if dtype == torch.float32 else 1e-12, rtol=1e-5)
        assert [id(p) for p in head.parameters()] == parameter_ids
    assert popart.sample_count.item() == 6
    assert popart.update_count.item() == 2
    assert popart.mean.item() == pytest.approx(-170.)
    # Mixture variance includes the between-batch mean shift.
    expected_variance = .8 * (10000 / 6) + .2 * 375000 + .2 * .8 * 100**2
    assert popart.variance.item() == pytest.approx(expected_variance)


def test_popart_initializes_from_first_batch_and_has_positive_constant_target_floor():
    popart = ScalarPopArt(beta=.001, min_std=.25)
    head = nn.Linear(2, 1, dtype=torch.float64)
    x = torch.randn(4, 2, dtype=torch.float64)
    old = popart.denormalize(head(x)).detach()
    targets = torch.full((6,), -50., dtype=torch.float64)
    popart.update(targets, None, head)
    assert popart.mean.item() == -50
    assert popart.variance.item() == 0
    assert popart.std.item() == .25
    torch.testing.assert_close(popart.normalize(targets), torch.zeros_like(targets))
    torch.testing.assert_close(popart.denormalize(head(x)), old)


def test_actor_snapshot_is_frozen_shared_uncentered_and_not_per_instance():
    actor = ActorAdvantageScale(beta=.25)
    raw = torch.tensor([[1., 2.], [10., 20.]])
    first = actor.snapshot()
    actor.update(raw)
    assert first.item() == 1  # Calibration did not mutate an existing snapshot.
    frozen = actor.snapshot()
    assert frozen.item() == pytest.approx((505 / 4)**.5)
    actual = actor.normalize(raw, frozen)
    assert actual[1, 0].item() / actual[0, 0].item() == pytest.approx(10.)
    assert actual.mean().item() > 0  # RMS scaling does not subtract a mean.
    actor.update(raw * 100)
    torch.testing.assert_close(actor.normalize(raw, frozen), actual)
    assert actor.snapshot().item() > frozen.item()


def test_actor_masked_padding_has_zero_output_and_gradient():
    actor = ActorAdvantageScale()
    raw = torch.tensor([3., 4., float('nan')], requires_grad=True)
    mask = torch.tensor([True, True, False])
    actor.update(raw, mask)
    out = actor.normalize(raw, actor.snapshot(), mask)
    assert out[-1].item() == 0
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert raw.grad[-1].item() == 0
    assert torch.isfinite(raw.grad).all()
    assert not actor.snapshot().requires_grad


@pytest.mark.parametrize('multiplier', [1e-3, 1e3, 1e6])
def test_normalized_actor_and_critic_targets_invariant_to_common_units(multiplier):
    x = torch.tensor([-20., -5., 2., 13.], dtype=torch.float64)
    base_actor, scaled_actor = ActorAdvantageScale(), ActorAdvantageScale()
    base_popart, scaled_popart = ScalarPopArt(), ScalarPopArt()
    head = nn.Linear(2, 1, dtype=torch.float64)
    scaled_head = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        scaled_head.weight.copy_(head.weight * multiplier)
        scaled_head.bias.copy_(head.bias * multiplier)
    base_actor.update(x)
    scaled_actor.update(x * multiplier)
    torch.testing.assert_close(base_actor.normalize(x, base_actor.snapshot()), scaled_actor.normalize(x * multiplier, scaled_actor.snapshot()))
    base_popart.update(x, None, head)
    scaled_popart.update(x * multiplier, None, scaled_head)
    torch.testing.assert_close(base_popart.normalize(x), scaled_popart.normalize(x * multiplier))
    torch.testing.assert_close(head.weight, scaled_head.weight)
    torch.testing.assert_close(head.bias, scaled_head.bias)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_inputs_promote_arithmetic_and_statistics_stay_double(dtype):
    actor, popart = ActorAdvantageScale(), ScalarPopArt()
    head = nn.Linear(2, 1)
    value = torch.tensor([-40000., -20000., 10000.], dtype=dtype)
    actor.update(value)
    popart.update(value, None, head)
    actor.to(dtype=dtype)
    popart.to(dtype=dtype)
    assert actor.second_moment.dtype == popart.variance.dtype == torch.float64
    assert torch.isfinite(actor.second_moment) and torch.isfinite(popart.variance)
    normalized = popart.normalize(value)
    assert normalized.dtype == actor.normalize(value, actor.snapshot()).dtype == torch.float32
    restored = popart.denormalize(normalized)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, value.float(), atol=.01, rtol=1e-5)
    x = torch.randn(4, 2)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        prediction = popart.denormalize(head(x))
        loss = popart.normalize(prediction).square().mean()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


def test_checkpoint_restores_statistics_configuration_head_and_next_update():
    popart = ScalarPopArt(beta=.2, min_std=.05)
    actor = ActorAdvantageScale(beta=.3, min_scale=.06)
    head = nn.Linear(3, 1)
    sample = torch.tensor([-20., 5., 6.])
    popart.update(sample, None, head)
    actor.update(sample)
    stream = io.BytesIO()
    torch.save({'popart': popart.state_dict(), 'actor': actor.state_dict(), 'head': head.state_dict()}, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored_popart, restored_actor, restored_head = ScalarPopArt(), ActorAdvantageScale(), nn.Linear(3, 1)
    restored_popart.load_state_dict(state['popart'])
    restored_actor.load_state_dict(state['actor'])
    restored_head.load_state_dict(state['head'])
    for normalizer, restored in ((popart, restored_popart), (actor, restored_actor)):
        for key, value in normalizer.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
    popart.update(sample * 3, None, head)
    restored_popart.update(sample * 3, None, restored_head)
    actor.update(sample * 3)
    restored_actor.update(sample * 3)
    torch.testing.assert_close(actor.snapshot(), restored_actor.snapshot())
    torch.testing.assert_close(head.weight, restored_head.weight)
    torch.testing.assert_close(head.bias, restored_head.bias)
    torch.testing.assert_close(popart.std, restored_popart.std)


@pytest.mark.parametrize('head', [nn.Linear(2, 2), nn.Linear(2, 1, bias=False), nn.Linear(2, 1).half()])
def test_popart_rejects_unsupported_heads_without_changing_statistics(head):
    popart = ScalarPopArt()
    with pytest.raises(ValueError):
        popart.update(torch.tensor([1., 2.]), None, head)
    assert popart.update_count.item() == 0


@pytest.mark.parametrize('beta', [0., -1., 1.1, float('nan')])
def test_invalid_ema_rate_rejected(beta):
    with pytest.raises(ValueError):
        ScalarPopArt(beta=beta)
    with pytest.raises(ValueError):
        ActorAdvantageScale(beta=beta)


def test_invalid_masks_scales_and_external_moments_fail_explicitly():
    with pytest.raises(ValueError, match='same shape'):
        masked_moments(torch.ones(3), torch.ones(1, dtype=torch.bool))
    with pytest.raises(TypeError, match='boolean'):
        masked_moments(torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError):
        ActorAdvantageScale(min_scale=0)
    with pytest.raises(ValueError):
        ScalarPopArt(min_std=float('inf'))
    with pytest.raises(ValueError):
        ActorAdvantageScale().normalize(torch.ones(3), torch.tensor(0.))
    with pytest.raises(ValueError):
        ActorAdvantageScale().update_from_moments(MaskedMoments(torch.tensor(1.), torch.tensor(0.), torch.tensor(-1.)))
    with pytest.raises(ValueError):
        merge_moments([])
    if not dist.is_initialized():
        with pytest.raises(RuntimeError, match='initialized'):
            masked_moments(torch.ones(3), distributed=True)


def _distributed_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        value = torch.tensor([1., 3., float('nan')]) if rank == 0 else torch.tensor([4., 5., 6.])
        mask = torch.tensor([True, True, False]) if rank == 0 else torch.ones(3, dtype=torch.bool)
        actor, popart = ActorAdvantageScale(), ScalarPopArt()
        torch.manual_seed(12)
        head = nn.Linear(2, 1)
        before = head(torch.ones(1, 2)).detach()
        moments = popart.update(value, mask, head, distributed=True)
        actor.update(value, mask, distributed=True)
        empty_rank = masked_moments(torch.tensor([4., 6.]), torch.full((2,), rank == 1, dtype=torch.bool), distributed=True)
        rejected = False
        try:
            masked_moments(torch.tensor([float('nan') if rank == 0 else 1.]), distributed=True)
        except ValueError:
            rejected = True
        torch.testing.assert_close(popart.denormalize(head(torch.ones(1, 2))), before, atol=1e-6, rtol=1e-5)
        payload = {'count': moments.count.item(), 'mean': moments.mean.item(), 'variance': moments.variance.item(), 'actor_scale': actor.snapshot().item(), 'empty_count': empty_rank.count.item(), 'empty_mean': empty_rank.mean.item(), 'empty_variance': empty_rank.variance.item(), 'rejected': rejected}
        Path(output_dir, f'rank_{rank}.json').write_text(json.dumps(payload))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo is unavailable')
def test_ddp_uses_global_valid_moments_including_empty_ranks_and_shared_rejection(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'rank_{rank}.json').read_text()) for rank in range(2)]
    assert rows[0] == rows[1]
    assert rows[0]['count'] == 5
    assert rows[0]['mean'] == pytest.approx(3.8)
    assert rows[0]['variance'] == pytest.approx(2.96)
    assert rows[0]['actor_scale'] == pytest.approx((87 / 5)**.5)
    assert rows[0]['empty_count'] == 2
    assert rows[0]['empty_mean'] == 5
    assert rows[0]['empty_variance'] == 1
    assert rows[0]['rejected']
