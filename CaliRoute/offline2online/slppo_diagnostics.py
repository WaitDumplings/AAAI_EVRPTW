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


def detached_component_gradients(loss, parameters):
    """Sample a small common head before its *normal* training backward.

    The caller must still backward this loss/graph to release its saved tensors.
    Only detached head gradients escape, so different loss graphs can be sampled
    sequentially without keeping their activations alive together.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters or not loss.requires_grad:
        return ()
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads))


def gradient_diagnostics_from_components(gradients):
    """Summarize detached head gradients; never retain a model forward graph."""
    gradients = {name: grads for name, grads in gradients.items() if grads}
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


def gradient_component_diagnostics(losses, parameters):
    """Sample already-live training graphs, all of which must be backwarded.

    For sequential losses, prefer detached_component_gradients before each
    normal backward, then gradient_diagnostics_from_components afterwards.
    """
    parameters = tuple(parameters)
    return gradient_diagnostics_from_components({
        name: detached_component_gradients(loss, parameters)
        for name, loss in losses.items()
    })


class DetachedGradientAccumulator:
    """Sum detached common-head gradients across sequential loss chunks.

    Norms of chunk sums represent the actual minibatch contribution; summing
    chunk norms would incorrectly hide cancellation. Sampling retains only the
    tiny parameter-gradient vectors, never their forward activation graphs.
    """
    def __init__(self, parameters, names=('ppo_minibatch', 'expert', 'replay')):
        self.parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        self.gradients = {name: tuple(torch.zeros_like(p, dtype=torch.float32) for p in self.parameters)
                          for name in names}

    def sample(self, name, loss):
        gradients = detached_component_gradients(loss, self.parameters)
        if gradients:
            if name not in self.gradients:
                self.gradients[name] = tuple(torch.zeros_like(g, dtype=torch.float32) for g in gradients)
            with torch.no_grad():
                for total, gradient in zip(self.gradients[name], gradients):
                    total.add_(gradient.float())
        return gradients

    def diagnostics(self):
        values = gradient_diagnostics_from_components(self.gradients)
        # Report only the intended comparisons. Inactive replay has zero norm,
        # not a meaningful cosine or an enormous expert/replay denominator ratio.
        output = {key: value for key, value in values.items() if key.endswith('_norm')}
        reference = values.get('grad_ppo_minibatch_norm')
        if reference is not None:
            undefined = reference.new_full((), float('nan'))  # JSON monitor emits null.
            for name in ('expert', 'replay'):
                norm = values.get(f'grad_{name}_norm')
                if norm is None:
                    continue
                output[f'grad_{name}_to_ppo_minibatch_ratio'] = torch.where(
                    reference > 1e-12, norm / reference.clamp_min(1e-12), undefined)
                output[f'grad_{name}_ppo_minibatch_cosine'] = torch.where(
                    (norm > 1e-12) & (reference > 1e-12),
                    values[f'grad_ppo_minibatch_{name}_cosine'], undefined)
        return output
