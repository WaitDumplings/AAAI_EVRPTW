"""Portable scalar value and policy normalization, independent of problem size.

Physical rewards/returns remain the host's responsibility. PopArt normalizes
critic regression targets while preserving the head's unnormalized predictions.
The actor uses one *uncentered*, historical RMS for every valid rollout sample.
Neither component divides by an instance size, trajectory length, or group std.

Statistics are float64 checkpoint buffers; low precision arithmetic is promoted
to float32. DDP statistics are opt-in and collective: every rank must call with
the same process group, including ranks whose local mask is empty. Normalizers
must have identical previous state on those ranks. No collective runs implicitly
in forward/normalize/snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.distributed as dist


@dataclass(frozen=True)
class MaskedMoments:
    """Population moments of all valid samples, stored as detached scalars."""

    count: Tensor
    mean: Tensor
    variance: Tensor


def _working_float(value: Tensor) -> Tensor:
    if not value.is_floating_point():
        raise TypeError("normalization values must be floating point")
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _mask_for(value: Tensor, valid: Tensor | None) -> Tensor:
    if valid is None:
        return torch.ones_like(value, dtype=torch.bool)
    if valid.shape != value.shape:
        raise ValueError("valid mask must have the same shape as values")
    if valid.dtype != torch.bool:
        raise TypeError("valid mask must have boolean dtype")
    return valid.to(device=value.device)


@torch.no_grad()
def masked_moments(
    values: Tensor,
    valid: Tensor | None = None,
    *,
    distributed: bool = False,
    group=None,
) -> MaskedMoments:
    """Centered population variance, optionally over all distributed ranks.

    Padded nonfinite values are ignored. A nonfinite *valid* value raises rather
    than silently changing the effective objective. With DDP, that rejection is
    synchronized so one rank cannot leave its peers waiting in the next reduce.
    Two reductions compute a global mean followed by centered squared residuals;
    averaging per-rank variances or subtracting E[x]^2 from E[x^2] is avoided.
    """
    _working_float(values)
    selected = values.detach().to(torch.float64)[_mask_for(values, valid)]
    if distributed and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("distributed moments require an initialized process group")
    finite = torch.isfinite(selected)
    safe = torch.where(finite, selected, 0.0)
    totals = torch.stack((selected.new_tensor(selected.numel()), safe.sum(), (~finite).sum()))
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=group)
    if totals[2].item() != 0 or not torch.isfinite(totals).all().item():
        raise ValueError("valid normalization samples must be finite")
    count = totals[0]
    mean = totals[1] / count.clamp_min(1)
    residual_sum = (selected - mean).square().sum()
    if distributed:
        dist.all_reduce(residual_sum, op=dist.ReduceOp.SUM, group=group)
    variance = residual_sum / count.clamp_min(1)
    if not torch.isfinite(variance).item():
        raise ValueError("normalization variance overflowed float64")
    return MaskedMoments(count.detach(), mean.detach(), variance.detach())


def _checked_moments(moments: MaskedMoments, device: torch.device) -> MaskedMoments:
    fields = []
    for value in (moments.count, moments.mean, moments.variance):
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise ValueError("moments must contain scalar tensors")
        fields.append(value.detach().to(device=device, dtype=torch.float64).reshape(()))
    count, mean, variance = fields
    if not torch.isfinite(torch.stack(fields)).all().item() or count.item() < 0 or variance.item() < 0:
        raise ValueError("moments must be finite with nonnegative count and variance")
    return MaskedMoments(count, mean, variance)


@torch.no_grad()
def merge_moments(parts: Iterable[MaskedMoments]) -> MaskedMoments:
    """Merge unequal-size chunk/worker summaries using count-weighted moments.

    Supply at least one summary (it may be empty). This is the noncollective
    interface for hosts that already orchestrate their own global aggregation.
    """
    parts = iter(parts)
    try:
        first = next(parts)
    except StopIteration as exc:
        raise ValueError("at least one moments summary is required") from exc
    state = _checked_moments(first, first.count.device)
    count, mean = state.count.clone(), state.mean.clone()
    square = state.variance * count
    for part in parts:
        other = _checked_moments(part, count.device)
        total = count + other.count
        delta = other.mean - mean
        square = square + other.variance * other.count + delta.square() * count * other.count / total.clamp_min(1)
        mean = mean + delta * other.count / total.clamp_min(1)
        count = total
    return _checked_moments(MaskedMoments(count, mean, square / count.clamp_min(1)), count.device)


class _ScalarStatistics(nn.Module):
    """Keep saved numeric state in float64 even when the host calls .half()."""

    def _apply(self, fn, recurse=True):
        originals = {name: value for name, value in self._buffers.items() if value is not None and value.is_floating_point()}
        result = super()._apply(fn, recurse=recurse)
        for name, value in originals.items():
            self._buffers[name] = value.to(device=self._buffers[name].device, dtype=torch.float64)
        return result

    def _initialize(self, beta: float, minimum: float) -> None:
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("beta must be finite and in (0, 1]")
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("minimum scale must be finite and positive")
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float64))
        self.register_buffer("minimum", torch.tensor(minimum, dtype=torch.float64))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.int64))


class ScalarPopArt(_ScalarStatistics):
    """EMA scalar return moments with output-preserving affine compensation.

    The external ``value_head`` must be a biased nn.Linear with one output and
    float32/float64 master parameters (AMP activations are fine). Its output is
    the normalized value. Call ``denormalize`` before physical-unit TD/GAE, and
    ``normalize`` on physical-unit targets for the critic regression loss.

    Call update once for the complete rollout, between optimizer updates, on
    every DDP rank with globally aggregated moments. Do not update on minibatches
    or evaluation data. The first nonempty observation initializes statistics
    directly; later batches use EMA with weight beta *per rollout*, not per row.

    Parameters are preserved in place, but optimizer moments are not transformed.
    The host must choose/document its optimizer-state treatment on warm starts;
    output preservation alone is not optimizer or policy-gradient invariance.
    """

    def __init__(self, beta: float = 0.01, min_std: float = 1e-4):
        super().__init__()
        self._initialize(beta, min_std)
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("variance", torch.ones((), dtype=torch.float64))

    @property
    def std(self) -> Tensor:
        return self.variance.clamp_min(0).sqrt().clamp_min(self.minimum)

    def normalize(self, value: Tensor) -> Tensor:
        work = _working_float(value)
        return (work - self.mean.to(work)) / self.std.to(work)

    def denormalize(self, value: Tensor) -> Tensor:
        work = _working_float(value)
        return work * self.std.to(work) + self.mean.to(work)

    @torch.no_grad()
    def update(
        self,
        targets: Tensor,
        valid: Tensor | None,
        value_head: nn.Linear,
        *,
        distributed: bool = False,
        group=None,
    ) -> MaskedMoments:
        moments = masked_moments(targets, valid, distributed=distributed, group=group)
        self.update_from_moments(moments, value_head)
        return moments

    @torch.no_grad()
    def update_from_moments(self, moments: MaskedMoments, value_head: nn.Linear) -> None:
        moments = _checked_moments(moments, self.mean.device)
        if moments.count.item() == 0:
            return
        if not isinstance(value_head, nn.Linear) or value_head.out_features != 1 or value_head.bias is None:
            raise ValueError("PopArt requires a biased scalar nn.Linear value head")
        if value_head.weight.dtype not in (torch.float32, torch.float64) or value_head.bias.dtype != value_head.weight.dtype:
            raise ValueError("PopArt value head requires float32 or float64 master parameters")
        if self.update_count.item() == 0:
            new_mean, new_variance = moments.mean, moments.variance
        else:
            delta = moments.mean - self.mean
            new_mean = self.mean + self.beta * delta
            new_variance = (1 - self.beta) * self.variance + self.beta * moments.variance + self.beta * (1 - self.beta) * delta.square()
        new_std = new_variance.sqrt().clamp_min(self.minimum)
        old_std = self.std
        weight = value_head.weight.detach().to(torch.float64)
        bias = value_head.bias.detach().to(torch.float64)
        adjusted_weight = (weight * (old_std / new_std).to(weight)).to(value_head.weight)
        adjusted_bias = ((old_std.to(bias) * bias + self.mean.to(bias) - new_mean.to(bias)) / new_std.to(bias)).to(value_head.bias)
        if not torch.isfinite(adjusted_weight).all().item() or not torch.isfinite(adjusted_bias).all().item() or not torch.isfinite(new_variance).item():
            raise ValueError("PopArt update would produce nonfinite value parameters or moments")
        value_head.weight.copy_(adjusted_weight)
        value_head.bias.copy_(adjusted_bias)
        self.mean.copy_(new_mean)
        self.variance.copy_(new_variance)
        self.sample_count.add_(moments.count)
        self.update_count.add_(1)


class ActorAdvantageScale(_ScalarStatistics):
    """One historical RMS, shared across all tasks/instances/steps in an update.

    Typical protocol: ``scale = normalizer.snapshot()``; normalize the entire
    raw rollout with that immutable scale; reuse it for all PPO passes; then
    ``update(raw_advantages, valid)`` to prepare the *next* rollout. Calibration
    with a frozen policy before the first update is permitted and must be logged.
    No centering, clipping, return transformation, or length weighting is applied.
    A shared scalar preserves within-batch advantage ratios, not relative weights
    against independently scaled entropy/value/SL losses.
    """

    def __init__(self, beta: float = 0.01, min_scale: float = 1e-4):
        super().__init__()
        self._initialize(beta, min_scale)
        self.register_buffer("second_moment", torch.ones((), dtype=torch.float64))

    def snapshot(self) -> Tensor:
        """Detached scalar copy; later statistics updates cannot change it."""
        return self.second_moment.clamp_min(0).sqrt().clamp_min(self.minimum).detach().clone()

    def normalize(self, advantages: Tensor, scale: Tensor, valid: Tensor | None = None) -> Tensor:
        """Apply the supplied snapshot without updating or retaining its graph."""
        work = _working_float(advantages)
        mask = _mask_for(advantages, valid)
        if not isinstance(scale, Tensor) or scale.numel() != 1 or not torch.isfinite(scale).all().item() or scale.item() <= 0:
            raise ValueError("scale must be one finite positive scalar tensor")
        safe = torch.where(mask, work, 0.0)
        return safe / scale.detach().to(work).reshape(())

    @torch.no_grad()
    def update(
        self,
        advantages: Tensor,
        valid: Tensor | None = None,
        *,
        distributed: bool = False,
        group=None,
    ) -> MaskedMoments:
        moments = masked_moments(advantages, valid, distributed=distributed, group=group)
        self.update_from_moments(moments)
        return moments

    @torch.no_grad()
    def update_from_moments(self, moments: MaskedMoments) -> None:
        moments = _checked_moments(moments, self.second_moment.device)
        if moments.count.item() == 0:
            return
        observed = moments.variance + moments.mean.square()
        updated = observed if self.update_count.item() == 0 else (1 - self.beta) * self.second_moment + self.beta * observed
        if not torch.isfinite(updated).item():
            raise ValueError("actor second moment overflowed float64")
        self.second_moment.copy_(updated)
        self.sample_count.add_(moments.count)
        self.update_count.add_(1)
