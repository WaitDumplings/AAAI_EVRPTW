"""Detached training diagnostics; tensor collection never synchronizes the GPU.

Convert a complete diagnostic dictionary with tensors_to_floats at an epoch or
minibatch boundary, rather than transferring a scalar on each decoding step.
"""
from __future__ import annotations

import numpy as np
import torch


def tensors_to_floats(values):
    if not values:
        return {}
    keys = list(values)
    packed = torch.stack([values[key].detach().float().reshape(()) for key in keys]).cpu().tolist()
    return dict(zip(keys, map(float, packed)))


def numpy_distribution(values, prefix, mask=None):
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values.reshape(-1)
    finite = values[np.isfinite(values)]
    out = {f"{prefix}_count": float(values.size), f"{prefix}_finite_frac": float(finite.size / max(values.size, 1))}
    for key, value in {
        "mean": finite.mean() if finite.size else 0.0,
        "std": finite.std() if finite.size else 0.0,
        "positive_frac": (finite > 0).mean() if finite.size else 0.0,
        "negative_frac": (finite < 0).mean() if finite.size else 0.0,
        "abs_mean": np.abs(finite).mean() if finite.size else 0.0,
        "p05": np.quantile(finite, .05) if finite.size else 0.0,
        "p50": np.quantile(finite, .50) if finite.size else 0.0,
        "p95": np.quantile(finite, .95) if finite.size else 0.0,
    }.items():
        out[f"{prefix}_{key}"] = float(value)
    return out


@torch.no_grad()
def value_and_advantage_diagnostics(values, returns, advantages, valid):
    """Epoch-level critic fit and step-advantage moments, all on current device."""
    finite = valid.bool() & torch.isfinite(values) & torch.isfinite(returns) & torch.isfinite(advantages)
    count = finite.sum().float()
    def mean(value):
        return torch.where(finite, value.float(), 0.0).sum() / count.clamp_min(1)
    return_mean = mean(returns)
    residual = returns.float() - values.float()
    return_var = mean((returns.float() - return_mean).square())
    residual_var = mean((residual - mean(residual)).square())
    advantage_mean = mean(advantages)
    return {
        "value_explained_variance": torch.where(return_var > 1e-12, 1.0 - residual_var / return_var.clamp_min(1e-12), 0.0),
        "value_target_variance": return_var,
        "value_rmse": mean(residual.square()).sqrt(),
        "step_adv_mean": advantage_mean,
        "step_adv_std": mean((advantages.float() - advantage_mean).square()).sqrt(),
        "step_adv_positive_frac": mean((advantages > 0).float()),
        "step_valid_finite_frac": count / valid.sum().clamp_min(1),
    }


def gradient_component_diagnostics(losses, parameters):
    """Optional sampled pre-backward gradient balance (no mutation, no CPU sync).

    Host should call only on a small common parameter subset and infrequently;
    extra autograd work is intentional, so this must not run each time step.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients = {}
    for name, loss in losses.items():
        if not parameters or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads))
    out = {}
    for name, grads in gradients.items():
        out[f"grad_{name}_norm"] = torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
    names = list(gradients)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            dot = torch.stack([(a.float() * b.float()).sum() for a, b in zip(gradients[left], gradients[right])]).sum()
            denom = out[f"grad_{left}_norm"] * out[f"grad_{right}_norm"]
            out[f"grad_{left}_{right}_cosine"] = dot / denom.clamp_min(1e-12)
            out[f"grad_{left}_to_{right}_ratio"] = out[f"grad_{left}_norm"] / out[f"grad_{right}_norm"].clamp_min(1e-12)
    return out
