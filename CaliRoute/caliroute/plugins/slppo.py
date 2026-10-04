"""Backbone-independent solution-level PPO tensor operations.

The host supplies selected-action log probabilities [time, ...routes], a valid
step mask, detached route advantages and optional route feasibility. No routing
instance, encoder, environment or optimizer dependency is required. Ratios use
mean log probability per route, preserving the original length normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def _stable_dtype(value: Tensor) -> Tensor:
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


@dataclass
class RoutePPOResult:
    loss: Tensor
    weights: Tensor
    mean_logratio: Tensor
    route_mask: Tensor
    active_mask: Tensor
    valid_counts: Tensor
    diagnostics: dict[str, Tensor]


def normalized_route_advantages(
    objectives: Tensor,
    feasible: Tensor,
    reference: Tensor | None = None,
    *,
    floor_mode: str = "relative",
    relative_floor: float = 0.01,
    absolute_floor: float = 5.0,
    clip: float = 3.0,
    include_reference: bool = True,
) -> Tensor:
    """Within-instance standardized improvement for minimizing objectives.

    The last dimension indexes sampled routes. A finite feasible reference may
    participate in the group mean/std. Relative floors scale with the reference
    (or feasible mean absolute objective when absent), giving invariance to
    positive changes in objective units. This is NOT translation invariant: a
    reference-relative floor deliberately preserves a meaningful objective zero.
    Infeasible/nonfinite routes receive zero and never influence group moments.
    """
    if floor_mode not in {"relative", "absolute"}:
        raise ValueError("floor_mode must be relative or absolute")
    if relative_floor < 0 or absolute_floor < 0 or clip <= 0:
        raise ValueError("advantage floors must be nonnegative and clip positive")
    objective = _stable_dtype(objectives.detach())
    valid = feasible.bool() & torch.isfinite(objective)
    safe = torch.where(valid, objective, torch.zeros_like(objective))
    count = valid.sum(-1, keepdim=True).to(objective.dtype)
    ref = None if reference is None else _stable_dtype(reference.detach()).unsqueeze(-1)
    ref_ok = torch.zeros_like(count, dtype=torch.bool) if ref is None else torch.isfinite(ref) & (ref > 0)
    ref_safe = torch.zeros_like(count) if ref is None else torch.where(ref_ok, ref, torch.zeros_like(ref))
    add_ref = ref_ok & include_reference
    n = count + add_ref.to(objective.dtype)
    mean = (safe.sum(-1, keepdim=True) + torch.where(add_ref, ref_safe, 0.0)) / n.clamp_min(1)
    square = torch.where(valid, (safe - mean).square(), 0.0).sum(-1, keepdim=True)
    square = square + torch.where(add_ref, (ref_safe - mean).square(), 0.0)
    std = (square / n.clamp_min(1)).sqrt()
    objective_scale = torch.where(ref_ok, ref_safe.abs(), safe.abs().sum(-1, keepdim=True) / count.clamp_min(1))
    floor = objective_scale * relative_floor if floor_mode == "relative" else torch.full_like(std, absolute_floor)
    denom = torch.maximum(std, floor).clamp_min(torch.finfo(objective.dtype).tiny)
    return torch.where(valid, ((mean - safe) / denom).clamp(-clip, clip), 0.0)


def clipped_route_surrogate(
    mean_logratio: Tensor,
    advantages: Tensor,
    route_mask: Tensor,
    *,
    clip_coef: float = 0.2,
    valid_counts: Tensor | None = None,
) -> RoutePPOResult:
    """Clipped route PPO objective and its detached log-probability weights.

    `weights / valid_counts` are exactly d(-loss)/d(step logprob), so a host
    can reuse/chunk a PPO forward without constructing another full graph.
    Empty batches have differentiable zero loss. Invalid routes are excluded
    and counted; exponentiation is float32 or float64, never AMP float16.
    """
    if not math.isfinite(clip_coef) or not 0 < clip_coef < 1:
        raise ValueError("clip_coef must be finite and between zero and one")
    delta = _stable_dtype(mean_logratio)
    adv = advantages.detach().to(dtype=delta.dtype, device=delta.device)
    requested = route_mask.bool() & (adv != 0)
    finite = torch.isfinite(delta) & torch.isfinite(adv)
    safe_delta = torch.where(finite, delta, torch.zeros_like(delta))
    safe_adv = torch.where(finite, adv, torch.zeros_like(adv))
    lower, upper = math.log1p(-clip_coef), math.log1p(clip_coef)
    selected = torch.where(safe_adv >= 0, safe_delta.clamp(max=upper), safe_delta.clamp(min=lower))
    max_log = math.log(torch.finfo(delta.dtype).max) - 2.0
    limit = max_log - safe_adv.abs().clamp_min(1.0).log()
    finite = finite & (selected <= limit)
    mask = requested & finite
    selected = torch.where(mask, selected, torch.zeros_like(selected))
    safe_adv = torch.where(mask, safe_adv, torch.zeros_like(safe_adv))
    count = mask.sum().to(delta.dtype)
    weighted = selected.exp() * safe_adv
    loss = -weighted.sum() / count.clamp_min(1)
    active = mask & torch.where(adv >= 0, safe_delta <= upper, safe_delta >= lower)
    weights = torch.where(active, weighted.detach() / count.clamp_min(1), 0.0)
    with torch.no_grad():
        diagnostic_delta = safe_delta.detach().double()
        diagnostic_adv = adv.double()
        bounded_ratio = diagnostic_delta.clamp(-80, 80).exp()
        r = bounded_ratio[mask]
        clipped = (safe_delta < lower) | (safe_delta > upper)
        def average(value):
            return torch.where(mask, value, 0.0).sum() / count.clamp_min(1)
        ratio_mean = average(bounded_ratio)
        advantage_mean = average(torch.where(finite, diagnostic_adv, 0.0))
        quantiles = torch.quantile(r, r.new_tensor([0.05, 0.5, 0.95])) if r.numel() else delta.new_ones(3)
        diagnostics = {
            "loss": loss.detach(),
            "ratio_mean": torch.where(count > 0, ratio_mean, 1.0),
            "ratio_std": average((bounded_ratio - ratio_mean).square()).sqrt(),
            "ratio_p05": quantiles[0], "ratio_p50": quantiles[1], "ratio_p95": quantiles[2],
            "clip_frac": average(clipped.to(delta.dtype)),
            "active_frac": active.sum().to(delta.dtype) / count.clamp_min(1),
            "adv_mean": advantage_mean,
            "adv_std": average(torch.where(finite, (diagnostic_adv - advantage_mean).square(), 0.0)).sqrt(),
            "adv_positive_frac": average((adv > 0).to(delta.dtype)),
            "num_routes_used": count,
            "rejected_nonfinite": (requested & ~finite).sum().to(delta.dtype),
            "approx_kl": average(bounded_ratio - 1.0 - safe_delta.detach()),
        }
    counts = torch.ones_like(delta) if valid_counts is None else valid_counts.detach().to(delta.dtype)
    return RoutePPOResult(loss, weights, delta, mask, active, counts, diagnostics)


def solution_level_ppo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    valid: Tensor,
    advantages: Tensor,
    *,
    feasible: Tensor | None = None,
    clip_coef: float = 0.2,
) -> RoutePPOResult:
    """Portable full route loss, also usable with arbitrary nonrouting models."""
    new, old = _stable_dtype(logprobs), _stable_dtype(old_logprobs.detach())
    mask = valid.bool()
    step_finite = torch.isfinite(new) & torch.isfinite(old)
    safe_new = torch.where(mask & step_finite, new, 0.0)
    safe_old = torch.where(mask & step_finite, old, 0.0)
    counts = mask.sum(0).to(new.dtype)
    mean_delta = (safe_new - safe_old).sum(0) / counts.clamp_min(1)
    route_mask = (counts > 0) & ((~mask) | step_finite).all(0)
    if feasible is not None:
        route_mask = route_mask & feasible.bool()
    result = clipped_route_surrogate(mean_delta, advantages, route_mask, clip_coef=clip_coef, valid_counts=counts)
    result.diagnostics["invalid_logprob_routes"] = (((~mask) | step_finite).all(0).logical_not() & (counts > 0)).sum().to(new.dtype)
    return result


def replay_weight_schedule(epoch: int, weight: float, warmup_epochs: int = 0, ramp_epochs: int = 0) -> float:
    """Collect verified memory during warmup; smoothly activate its loss later."""
    if not math.isfinite(weight) or weight < 0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("Replay schedule requires finite nonnegative weight and durations")
    if epoch <= warmup_epochs:
        return 0.0
    return weight * min(1.0, (epoch - warmup_epochs) / max(ramp_epochs, 1))
